from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BenchmarkMetrics(BaseModel):
    total_latency_ms: float
    time_to_first_token_ms: float | None = None
    tokens_per_second: float
    generated_tokens: int
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    draft_tokens_rejected: int = 0
    acceptance_rate: float = 0.0
    verifier_forward_calls: int
    draft_forward_calls: int = 0
    verifier_calls_per_output_token: float
    draft_overhead_ms: float | None = None
    speedup_vs_baseline: float | None = None
    slowdown_flag: bool = False
    output_length: int
    exact_output_match_with_baseline: bool | None = None
    model_pair: str
    prompt_type: str
    temperature: float
    draft_k: int | None = None
    repetition: int
    schedule_type: str = "static"
    static_draft_k: int | None = None
    adaptive_policy: str | None = None
    average_selected_k: float | None = None
    min_selected_k: int | None = None
    max_selected_k: int | None = None
    wasted_draft_tokens: int | None = None


def acceptance_rate(accepted: int, proposed: int) -> float:
    return accepted / proposed if proposed else 0.0


def speedup(baseline_latency_ms: float, candidate_latency_ms: float) -> float | None:
    if candidate_latency_ms <= 0:
        return None
    return baseline_latency_ms / candidate_latency_ms


def result_to_metrics(
    result: Any,
    *,
    baseline_result: Any | None,
    model_pair: str,
    prompt_type: str,
    temperature: float,
    draft_k: int | None,
    repetition: int,
    schedule_type: str = "static",
    adaptive_policy: str | None = None,
) -> BenchmarkMetrics:
    baseline_latency = getattr(baseline_result, "total_latency_ms", None)
    candidate_latency = result.total_latency_ms
    speed = speedup(baseline_latency, candidate_latency) if baseline_latency is not None else None
    generated = result.generated_tokens
    return BenchmarkMetrics(
        total_latency_ms=candidate_latency,
        time_to_first_token_ms=getattr(result, "time_to_first_token_ms", None),
        tokens_per_second=result.tokens_per_second,
        generated_tokens=generated,
        draft_tokens_proposed=getattr(result, "draft_tokens_proposed", 0),
        draft_tokens_accepted=getattr(result, "draft_tokens_accepted", 0),
        draft_tokens_rejected=getattr(result, "draft_tokens_rejected", 0),
        acceptance_rate=getattr(result, "acceptance_rate", 0.0),
        verifier_forward_calls=result.verifier_forward_calls,
        draft_forward_calls=getattr(result, "draft_forward_calls", 0),
        verifier_calls_per_output_token=(
            result.verifier_forward_calls / generated if generated else 0.0
        ),
        draft_overhead_ms=getattr(result, "draft_overhead_ms", None),
        speedup_vs_baseline=speed,
        slowdown_flag=(speed is not None and speed < 1.0),
        output_length=len(result.output_token_ids),
        exact_output_match_with_baseline=(
            result.output_token_ids == baseline_result.output_token_ids
            if baseline_result is not None and temperature == 0.0
            else None
        ),
        model_pair=model_pair,
        prompt_type=prompt_type,
        temperature=temperature,
        draft_k=draft_k,
        repetition=repetition,
        schedule_type=schedule_type,
        static_draft_k=draft_k if schedule_type == "static" else None,
        adaptive_policy=adaptive_policy,
        average_selected_k=getattr(result, "average_selected_k", None),
        min_selected_k=getattr(result, "min_selected_k", None),
        max_selected_k=getattr(result, "max_selected_k", None),
        wasted_draft_tokens=getattr(result, "wasted_draft_tokens", None),
    )
