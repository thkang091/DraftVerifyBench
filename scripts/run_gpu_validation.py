from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.config import load_config
from draftverifybench.runner import run_benchmark
from draftverifybench.utils import cuda_hardware_metadata, ensure_parent


def _write_summary(path: str | Path, payload: dict) -> None:
    ensure_parent(path)
    lines = [
        "# GPU Validation Summary",
        "",
        f"Config: `{payload['config']}`",
        f"CUDA available: `{payload['cuda_available']}`",
        f"Status: `{payload['status']}`",
        f"Rows: `{payload.get('rows', 0)}`",
        "",
        "## Hardware",
        "",
        "```json",
        json.dumps(payload["hardware"], indent=2),
        "```",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_gpu_validation(
    config_path: str,
    *,
    out: str,
    raw_out: str,
    metadata_out: str,
    summary_out: str,
    max_prompts: int | None = None,
    require_cuda: bool = False,
) -> int:
    config = load_config(config_path)
    cuda_available = torch.cuda.is_available()
    hardware = cuda_hardware_metadata()
    print(f"CUDA available: {cuda_available}")
    print(json.dumps(hardware, indent=2))

    if require_cuda and not cuda_available:
        payload = {
            "config": config_path,
            "cuda_available": False,
            "status": "failed_require_cuda",
            "hardware": hardware,
        }
        ensure_parent(metadata_out)
        Path(metadata_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_summary(summary_out, payload)
        print("CUDA is required but unavailable.", file=sys.stderr)
        return 2

    if config.device == "cuda" and not cuda_available:
        payload = {
            "config": config_path,
            "cuda_available": False,
            "status": "skipped_cuda_unavailable",
            "hardware": hardware,
            "note": "GPU validation config requested CUDA, but this machine has no CUDA device.",
        }
        ensure_parent(metadata_out)
        Path(metadata_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ensure_parent(out)
        Path(out).write_text("", encoding="utf-8")
        ensure_parent(raw_out)
        Path(raw_out).write_text("", encoding="utf-8")
        _write_summary(summary_out, payload)
        return 0

    rows = run_benchmark(
        config_path,
        out=out,
        raw_out=raw_out,
        metadata_out=metadata_out,
        max_prompts=max_prompts,
    )
    payload = {
        "config": config_path,
        "cuda_available": cuda_available,
        "status": "completed",
        "hardware": hardware,
        "rows": len(rows),
    }
    _write_summary(summary_out, payload)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA GPU validation benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/gpu_validation_results.csv")
    parser.add_argument("--raw-out", default="results/gpu_validation_raw.jsonl")
    parser.add_argument("--metadata-out", default="results/gpu_validation_metadata.json")
    parser.add_argument("--summary-out", default="results/gpu_validation_summary.md")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_gpu_validation(
            args.config,
            out=args.out,
            raw_out=args.raw_out,
            metadata_out=args.metadata_out,
            summary_out=args.summary_out,
            max_prompts=args.max_prompts,
            require_cuda=args.require_cuda,
        )
    )


if __name__ == "__main__":
    main()

