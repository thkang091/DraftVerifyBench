from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.datasets import get_prompts
from draftverifybench.utils import ensure_parent


def _load_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "vLLM is not installed. Install vLLM on a CUDA Linux instance before running this "
            "comparison. See docs/Next_GPU_Experiments.md."
        ) from exc
    return LLM, SamplingParams


def _generate(llm: Any, sampling_params: Any, prompts: list[str]) -> tuple[float, list[Any]]:
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return latency_ms, outputs


def _output_token_count(output: Any) -> int:
    if not output.outputs:
        return 0
    token_ids = getattr(output.outputs[0], "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    text = getattr(output.outputs[0], "text", "")
    return len(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vLLM baseline and draft-model speculative decoding."
    )
    parser.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--verifier-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--prompt-types", nargs="+", default=["structured_json", "code_completion"])
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", default="results/vllm_comparison.csv")
    parser.add_argument("--raw-out", default="results/vllm_comparison_raw.jsonl")
    parser.add_argument("--metadata-out", default="results/vllm_comparison_metadata.json")
    args = parser.parse_args()

    LLM, SamplingParams = _load_vllm()
    prompt_rows = get_prompts(args.prompt_types, max_prompts=args.max_prompts)
    prompts = [prompt.prompt_text for prompt in prompt_rows]
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    common_kwargs = {
        "model": args.verifier_model,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    baseline_llm = LLM(**common_kwargs)
    baseline_latency_ms, baseline_outputs = _generate(baseline_llm, sampling_params, prompts)
    del baseline_llm

    speculative_config = {
        "method": "draft_model",
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    speculative_llm = LLM(**common_kwargs, speculative_config=speculative_config)
    speculative_latency_ms, speculative_outputs = _generate(
        speculative_llm, sampling_params, prompts
    )

    baseline_tokens = sum(_output_token_count(output) for output in baseline_outputs)
    speculative_tokens = sum(_output_token_count(output) for output in speculative_outputs)
    rows = [
        {
            "mode": "vllm_baseline",
            "draft_model": "",
            "verifier_model": args.verifier_model,
            "prompt_count": len(prompts),
            "temperature": args.temperature,
            "num_speculative_tokens": "",
            "latency_ms": baseline_latency_ms,
            "generated_tokens": baseline_tokens,
            "tokens_per_second": baseline_tokens / (baseline_latency_ms / 1000.0),
            "speedup_vs_vllm_baseline": None,
        },
        {
            "mode": "vllm_speculative",
            "draft_model": args.draft_model,
            "verifier_model": args.verifier_model,
            "prompt_count": len(prompts),
            "temperature": args.temperature,
            "num_speculative_tokens": args.num_speculative_tokens,
            "latency_ms": speculative_latency_ms,
            "generated_tokens": speculative_tokens,
            "tokens_per_second": speculative_tokens / (speculative_latency_ms / 1000.0),
            "speedup_vs_vllm_baseline": baseline_latency_ms / speculative_latency_ms,
        },
    ]

    ensure_parent(args.out)
    with Path(args.out).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ensure_parent(args.raw_out)
    with Path(args.raw_out).open("w", encoding="utf-8") as handle:
        for mode, outputs in [
            ("vllm_baseline", baseline_outputs),
            ("vllm_speculative", speculative_outputs),
        ]:
            for prompt, output in zip(prompt_rows, outputs, strict=True):
                text = output.outputs[0].text if output.outputs else ""
                handle.write(
                    json.dumps(
                        {
                            "mode": mode,
                            "prompt_id": prompt.prompt_id,
                            "prompt_type": prompt.prompt_type,
                            "output_text": text,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    metadata = {
        "backend": "vllm",
        "draft_model": args.draft_model,
        "verifier_model": args.verifier_model,
        "speculative_config": speculative_config,
        "note": "Uses vLLM offline LLM API with draft-model speculative_config.",
    }
    ensure_parent(args.metadata_out)
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote vLLM comparison to {args.out}")


if __name__ == "__main__":
    main()
