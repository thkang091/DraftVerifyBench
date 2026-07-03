from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from draftverifybench.adaptive import adaptive_speculative_decode
from draftverifybench.decoding import baseline_decode
from draftverifybench.models import load_model_bundle
from draftverifybench.speculative import speculative_decode


def _device_activities(require_cuda: bool) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif require_cuda:
        raise SystemExit("CUDA profiling requested, but torch.cuda.is_available() is false.")
    return activities


def _run_profiled_decode(args: argparse.Namespace):
    if args.mode == "baseline":
        verifier = load_model_bundle(
            args.verifier_model,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        return baseline_decode(
            verifier,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )

    draft = load_model_bundle(
        args.draft_model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    verifier = load_model_bundle(
        args.verifier_model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    if args.mode == "speculative":
        return speculative_decode(
            draft,
            verifier,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            draft_k=args.draft_k,
            temperature=args.temperature,
            seed=args.seed,
        )
    if args.mode == "adaptive":
        return adaptive_speculative_decode(
            draft,
            verifier,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            policy=args.adaptive_policy,
            seed=args.seed,
        )
    raise ValueError(f"Unknown mode: {args.mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one DraftVerifyBench decode path.")
    parser.add_argument("--mode", choices=["baseline", "speculative", "adaptive"], required=True)
    parser.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--verifier-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--prompt",
        default='Write a JSON object with keys "name", "age", and "city":',
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--adaptive-policy", default="confidence_threshold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/profiles")
    parser.add_argument("--trace-name")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.trace_name or f"{args.mode}_k{args.draft_k}_temp{args.temperature}".replace(
        ".", "_"
    )
    trace_path = out_dir / f"{name}_trace.json"
    table_path = out_dir / f"{name}_profile.txt"
    result_path = out_dir / f"{name}_result.txt"

    activities = _device_activities(args.require_cuda)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        result = _run_profiled_decode(args)

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
    prof.export_chrome_trace(str(trace_path))
    table_path.write_text(table, encoding="utf-8")
    result_path.write_text(str(result), encoding="utf-8")

    print(f"Wrote trace: {trace_path}")
    print(f"Wrote table: {table_path}")
    print(f"Wrote result: {result_path}")


if __name__ == "__main__":
    main()
