from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from draftverifybench.models import detect_device, load_model_bundle
from draftverifybench.utils import ensure_parent, set_seed, synchronize_device


def _base_model(model: Any) -> Any:
    base = getattr(model, "model", None)
    if base is None:
        raise TypeError(
            f"Cannot find transformer body at model.model for {type(model).__name__}. "
            "This profiler currently supports Llama/Qwen-style CausalLM modules."
        )
    return base


def _lm_head(model: Any) -> Any:
    head = getattr(model, "lm_head", None)
    if head is None:
        raise TypeError(
            f"Cannot find lm_head for {type(model).__name__}. "
            "This profiler currently supports Llama/Qwen-style CausalLM modules."
        )
    return head


def _timed_call(device: torch.device, fn):
    synchronize_device(device)
    start = time.perf_counter()
    value = fn()
    synchronize_device(device)
    return (time.perf_counter() - start) * 1000.0, value


@torch.inference_mode()
def profile_model(
    *,
    model_name: str,
    prompt: str,
    device_name: str,
    dtype: str,
    seed: int,
    decode_steps: int,
    local_files_only: bool,
) -> dict[str, Any]:
    set_seed(seed)
    bundle = load_model_bundle(
        model_name,
        device=device_name,
        dtype=dtype,
        seed=seed,
        local_files_only=local_files_only,
    )
    device = detect_device(device_name)
    model = bundle.model
    body = _base_model(model)
    lm_head = _lm_head(model)
    encoded = bundle.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # Prefill: full prompt through transformer body, then final-position LM head.
    prefill_body_ms, body_outputs = _timed_call(
        device,
        lambda: body(input_ids=input_ids, use_cache=True),
    )
    last_hidden = body_outputs.last_hidden_state[:, -1:, :]
    prefill_lm_head_ms, logits = _timed_call(device, lambda: lm_head(last_hidden))
    next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
    past_key_values = body_outputs.past_key_values

    decode_body_times: list[float] = []
    decode_lm_head_times: list[float] = []
    for _ in range(decode_steps):
        body_ms, body_outputs = _timed_call(
            device,
            lambda: body(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            ),
        )
        past_key_values = body_outputs.past_key_values
        last_hidden = body_outputs.last_hidden_state[:, -1:, :]
        head_ms, logits = _timed_call(device, lambda: lm_head(last_hidden))
        next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        decode_body_times.append(body_ms)
        decode_lm_head_times.append(head_ms)

    vocab_size = int(getattr(model.config, "vocab_size", 0))
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    lm_head_params = sum(param.numel() for param in lm_head.parameters())
    mean_decode_body_ms = sum(decode_body_times) / max(len(decode_body_times), 1)
    mean_decode_lm_head_ms = sum(decode_lm_head_times) / max(len(decode_lm_head_times), 1)
    mean_decode_total_ms = mean_decode_body_ms + mean_decode_lm_head_ms

    return {
        "model_name": model_name,
        "device": str(device),
        "dtype": str(bundle.dtype),
        "parameter_count": bundle.parameter_count,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "lm_head_params": lm_head_params,
        "prompt_tokens": int(input_ids.shape[-1]),
        "decode_steps": decode_steps,
        "prefill_body_ms": prefill_body_ms,
        "prefill_lm_head_ms": prefill_lm_head_ms,
        "prefill_lm_head_share": prefill_lm_head_ms
        / max(prefill_body_ms + prefill_lm_head_ms, 1e-9),
        "mean_decode_body_ms": mean_decode_body_ms,
        "mean_decode_lm_head_ms": mean_decode_lm_head_ms,
        "mean_decode_total_ms": mean_decode_total_ms,
        "decode_lm_head_share": mean_decode_lm_head_ms / max(mean_decode_total_ms, 1e-9),
        "decode_body_times_ms": decode_body_times,
        "decode_lm_head_times_ms": decode_lm_head_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile transformer-body vs LM-head time for Llama/Qwen CausalLM models."
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--prompt",
        default='Write a JSON object with keys "name", "age", and "city":',
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--out", default="results/lm_head_overhead_profile.csv")
    parser.add_argument("--raw-out", default="results/lm_head_overhead_profile.json")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    rows = [
        profile_model(
            model_name=model_name,
            prompt=args.prompt,
            device_name=args.device,
            dtype=args.dtype,
            seed=args.seed,
            decode_steps=args.decode_steps,
            local_files_only=args.local_files_only,
        )
        for model_name in args.models
    ]

    ensure_parent(args.raw_out)
    Path(args.raw_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"decode_body_times_ms", "decode_lm_head_times_ms"}
        }
        for row in rows
    ]
    ensure_parent(args.out)
    with Path(args.out).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote LM-head overhead profile to {args.out}")


if __name__ == "__main__":
    main()
