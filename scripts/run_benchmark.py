from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DraftVerifyBench.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(
        args.config,
        out=args.out,
        raw_out=args.raw_out,
        metadata_out=args.metadata_out,
        max_prompts=args.max_prompts,
    )


if __name__ == "__main__":
    main()
