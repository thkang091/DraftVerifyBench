from __future__ import annotations

import torch

from draftverifybench.decoding import baseline_decode, sample_from_logits
from tests.helpers import TinyBundle


def test_greedy_sampling_returns_argmax() -> None:
    logits = torch.tensor([[0.1, 2.0, 0.3]])
    assert sample_from_logits(logits, temperature=0.0) == 1


def test_baseline_decoder_returns_expected_fields() -> None:
    result = baseline_decode(TinyBundle("tiny", [4, 5, 99]), "prompt", max_new_tokens=8)
    assert result.output_token_ids == [4, 5, 99]
    assert result.generated_tokens == 3
    assert result.verifier_forward_calls == 3
    assert result.tokens_per_second >= 0.0

