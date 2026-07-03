from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.models import load_model_bundle
from draftverifybench.speculative import speculative_decode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one greedy speculative decode.")
    parser.add_argument("--draft-model", default="distilgpt2")
    parser.add_argument("--verifier-model", default="gpt2")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    draft = load_model_bundle(args.draft_model)
    verifier = load_model_bundle(args.verifier_model)
    result = speculative_decode(
        draft,
        verifier,
        args.prompt,
        draft_k=args.draft_k,
        max_new_tokens=args.max_new_tokens,
    )
    print(result.output_text)
    print(result)


if __name__ == "__main__":
    main()
