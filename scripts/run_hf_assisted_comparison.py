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
from transformers import AutoModelForCausalLM, AutoTokenizer

from draftverifybench.datasets import get_prompts
from draftverifybench.models import detect_device, select_dtype
from draftverifybench.utils import ensure_parent, set_seed, synchronize_device


def _load(model_name: str, *, device: torch.device, dtype: torch.dtype, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def _generate_once(
    *,
    tokenizer: Any,
    verifier_model: Any,
    assistant_model: Any | None,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs.update({"do_sample": False})
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model

    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = verifier_model.generate(**inputs, **generate_kwargs)
    synchronize_device(device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0, prompt_len:].detach().cpu().tolist()
    return {
        "latency_ms": latency_ms,
        "generated_tokens": len(generated_ids),
        "tokens_per_second": len(generated_ids) / (latency_ms / 1000.0) if latency_ms > 0 else 0.0,
        "output_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "output_token_ids": generated_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Transformers generate baseline with assisted generation."
    )
    parser.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--verifier-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-types", nargs="+", default=["structured_json", "code_completion"])
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/hf_assisted_comparison.csv")
    parser.add_argument("--raw-out", default="results/hf_assisted_comparison_raw.jsonl")
    parser.add_argument("--metadata-out", default="results/hf_assisted_comparison_metadata.json")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = detect_device(args.device)
    dtype = select_dtype(device, args.dtype)
    tokenizer, verifier = _load(
        args.verifier_model,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    _, assistant = _load(
        args.draft_model,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )

    prompts = get_prompts(args.prompt_types, max_prompts=args.max_prompts)
    rows: list[dict[str, Any]] = []
    raw_path = Path(args.raw_out)
    ensure_parent(raw_path)
    raw_path.write_text("", encoding="utf-8")

    for prompt in prompts:
        for temperature in args.temperatures:
            for repetition in range(args.repetitions):
                baseline = _generate_once(
                    tokenizer=tokenizer,
                    verifier_model=verifier,
                    assistant_model=None,
                    prompt=prompt.prompt_text,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    seed=args.seed + repetition,
                )
                assisted = _generate_once(
                    tokenizer=tokenizer,
                    verifier_model=verifier,
                    assistant_model=assistant,
                    prompt=prompt.prompt_text,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    seed=args.seed + repetition,
                )
                for mode, result in [("hf_generate", baseline), ("hf_assisted", assisted)]:
                    row = {
                        "mode": mode,
                        "prompt_id": prompt.prompt_id,
                        "prompt_type": prompt.prompt_type,
                        "temperature": temperature,
                        "repetition": repetition,
                        "draft_model": args.draft_model if mode == "hf_assisted" else "",
                        "verifier_model": args.verifier_model,
                        "device_type": str(device),
                        "dtype": str(dtype),
                        "latency_ms": result["latency_ms"],
                        "generated_tokens": result["generated_tokens"],
                        "tokens_per_second": result["tokens_per_second"],
                        "speedup_vs_hf_generate": (
                            baseline["latency_ms"] / assisted["latency_ms"]
                            if mode == "hf_assisted" and assisted["latency_ms"] > 0
                            else None
                        ),
                        "exact_output_match_with_hf_generate": (
                            result["output_token_ids"] == baseline["output_token_ids"]
                            if temperature == 0.0
                            else None
                        ),
                    }
                    rows.append(row)
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    **row,
                                    "output_text": result["output_text"],
                                    "output_token_ids": result["output_token_ids"],
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )

    ensure_parent(args.out)
    with Path(args.out).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "draft_model": args.draft_model,
        "verifier_model": args.verifier_model,
        "device": str(device),
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(prompts),
        "temperatures": args.temperatures,
        "repetitions": args.repetitions,
        "note": "Uses Transformers generate(..., assistant_model=...) for assisted generation.",
    }
    ensure_parent(args.metadata_out)
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
