from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.decoding import baseline_decode
from draftverifybench.models import load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one baseline decode.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    bundle = load_model_bundle(args.model)
    result = baseline_decode(
        bundle,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(result.output_text)
    print(result)


if __name__ == "__main__":
    main()
