from __future__ import annotations

from draftverifybench.decoding import BaselineDecodeResult
from draftverifybench.metrics import BenchmarkMetrics, acceptance_rate, result_to_metrics, speedup


def test_acceptance_rate_calculation() -> None:
    assert acceptance_rate(3, 4) == 0.75
    assert acceptance_rate(0, 0) == 0.0


def test_speedup_calculation() -> None:
    assert speedup(100.0, 50.0) == 2.0


def test_metrics_schema_validation() -> None:
    baseline = BaselineDecodeResult(
        output_token_ids=[1, 2],
        output_text="1 2",
        generated_tokens=2,
        total_latency_ms=100.0,
        time_to_first_token_ms=10.0,
        tokens_per_second=20.0,
        per_token_latency_ms=[10.0, 20.0],
        verifier_forward_calls=2,
    )
    metrics = result_to_metrics(
        baseline,
        baseline_result=None,
        model_pair="a->b",
        prompt_type="factual_qa",
        temperature=0.0,
        draft_k=None,
        repetition=0,
    )
    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.verifier_calls_per_output_token == 1.0

