from __future__ import annotations

from draftverifybench.speculative import speculative_decode
from tests.helpers import TinyBundle


def test_speculative_accepts_all_tokens_when_models_agree() -> None:
    draft = TinyBundle("draft", [7, 8, 99])
    verifier = TinyBundle("verifier", [7, 8, 99])
    result = speculative_decode(draft, verifier, "prompt", max_new_tokens=8, draft_k=2)
    assert result.output_token_ids == [7, 8, 99]
    assert result.draft_tokens_accepted == 3
    assert result.draft_tokens_rejected == 0
    assert result.acceptance_rate == 1.0


def test_speculative_rejects_when_draft_differs() -> None:
    draft = TinyBundle("draft", [7, 42, 99])
    verifier = TinyBundle("verifier", [7, 8, 99])
    result = speculative_decode(draft, verifier, "prompt", max_new_tokens=3, draft_k=3)
    assert result.output_token_ids[:2] == [7, 8]
    assert result.draft_tokens_rejected >= 1
    assert result.acceptance_rate < 1.0

