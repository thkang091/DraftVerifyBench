from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from draftverifybench.decoding import _decode, _device_of, _encode, next_token_id
from draftverifybench.speculative import _append, _greedy_tokens_for_contexts
from draftverifybench.utils import elapsed_ms, now_perf, set_seed, synchronize_device


class AdaptivePolicyConfig(BaseModel):
    confidence_thresholds: dict[float, int] = Field(
        default_factory=lambda: {0.85: 8, 0.70: 4, 0.50: 2}
    )
    entropy_thresholds: dict[float, int] = Field(default_factory=lambda: {2.0: 8, 4.0: 4, 6.0: 2})
    rolling_acceptance_window: int = 8
    rolling_acceptance_min_k: int = 1
    rolling_acceptance_max_k: int = 8


class AdaptiveSpeculativeDecodeResult(BaseModel):
    output_token_ids: list[int]
    output_text: str
    generated_tokens: int
    total_latency_ms: float
    tokens_per_second: float
    verifier_forward_calls: int
    draft_forward_calls: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    draft_tokens_rejected: int
    acceptance_rate: float
    draft_overhead_ms: float
    schedule_type: str
    adaptive_policy: str
    selected_k_per_step: list[int]
    average_selected_k: float
    min_selected_k: int
    max_selected_k: int
    confidence_per_step: list[float | None]
    entropy_per_step: list[float | None]
    recent_acceptance_per_step: list[float | None]
    accepted_tokens_by_k: dict[int, int]
    rejected_tokens_by_k: dict[int, int]
    wasted_draft_tokens: int


def confidence_threshold_k(confidence: float, thresholds: dict[float, int]) -> int:
    for threshold, draft_k in sorted(thresholds.items(), reverse=True):
        if confidence >= float(threshold):
            return int(draft_k)
    return 1


def entropy_threshold_k(entropy: float, thresholds: dict[float, int]) -> int:
    for threshold, draft_k in sorted(thresholds.items()):
        if entropy <= float(threshold):
            return int(draft_k)
    return 1


def rolling_acceptance_k(
    recent_acceptance: float,
    *,
    current_k: int,
    min_k: int = 1,
    max_k: int = 8,
) -> int:
    if recent_acceptance >= 0.85:
        return min(max_k, current_k * 2)
    if recent_acceptance <= 0.50:
        return max(min_k, max(1, current_k // 2))
    return current_k


@torch.inference_mode()
def draft_distribution_stats(
    bundle: Any,
    input_ids: torch.Tensor,
) -> tuple[float | None, float | None]:
    if hasattr(bundle.model, "next_token"):
        return 1.0, 0.0
    device = _device_of(bundle)
    outputs = bundle.model(input_ids=input_ids.to(device))
    logits = outputs.logits[:, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    confidence = float(probs.max(dim=-1).values.item())
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).item())
    if math.isnan(confidence) or math.isnan(entropy):
        return None, None
    return confidence, entropy


def choose_adaptive_k(
    policy: str,
    *,
    confidence: float | None,
    entropy: float | None,
    recent_acceptance: float | None,
    current_k: int,
    config: AdaptivePolicyConfig,
) -> int:
    if policy == "confidence_threshold":
        return confidence_threshold_k(confidence or 0.0, config.confidence_thresholds)
    if policy == "entropy_threshold":
        return entropy_threshold_k(
            entropy if entropy is not None else float("inf"),
            config.entropy_thresholds,
        )
    if policy == "rolling_acceptance":
        return rolling_acceptance_k(
            recent_acceptance if recent_acceptance is not None else 1.0,
            current_k=current_k,
            min_k=config.rolling_acceptance_min_k,
            max_k=config.rolling_acceptance_max_k,
        )
    raise ValueError(f"Unknown adaptive policy: {policy}")


@torch.inference_mode()
def adaptive_speculative_decode(
    draft_bundle: Any,
    verifier_bundle: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    policy: str = "confidence_threshold",
    policy_config: AdaptivePolicyConfig | None = None,
    seed: int | None = None,
) -> AdaptiveSpeculativeDecodeResult:
    set_seed(seed)
    config = policy_config or AdaptivePolicyConfig()
    device = _device_of(verifier_bundle)
    input_ids = _encode(verifier_bundle.tokenizer, prompt, device)
    prompt_len = input_ids.shape[-1]
    eos_token_id = getattr(verifier_bundle.tokenizer, "eos_token_id", None)

    verifier_forward_calls = 0
    draft_forward_calls = 0
    proposed = 0
    accepted = 0
    rejected = 0
    wasted = 0
    draft_overhead_ms = 0.0
    current_k = 1
    selected_k_per_step: list[int] = []
    confidence_per_step: list[float | None] = []
    entropy_per_step: list[float | None] = []
    recent_acceptance_per_step: list[float | None] = []
    recent_window: deque[int] = deque(maxlen=config.rolling_acceptance_window)
    accepted_by_k: dict[int, int] = {}
    rejected_by_k: dict[int, int] = {}

    synchronize_device(device)
    start = now_perf()
    while (input_ids.shape[-1] - prompt_len) < max_new_tokens:
        draft_context = input_ids.to(_device_of(draft_bundle))
        confidence, entropy = draft_distribution_stats(draft_bundle, draft_context)
        recent_acceptance = (
            sum(recent_window) / len(recent_window) if len(recent_window) > 0 else None
        )
        current_k = choose_adaptive_k(
            policy,
            confidence=confidence,
            entropy=entropy,
            recent_acceptance=recent_acceptance,
            current_k=current_k,
            config=config,
        )
        selected_k_per_step.append(current_k)
        confidence_per_step.append(confidence)
        entropy_per_step.append(entropy)
        recent_acceptance_per_step.append(recent_acceptance)

        draft_ids: list[int] = []
        for _ in range(current_k):
            if (input_ids.shape[-1] - prompt_len + len(draft_ids)) >= max_new_tokens:
                break
            synchronize_device(_device_of(draft_bundle))
            draft_start = now_perf()
            draft_token = next_token_id(draft_bundle, draft_context, temperature=0.0)
            synchronize_device(_device_of(draft_bundle))
            draft_overhead_ms += elapsed_ms(draft_start)
            draft_forward_calls += 1
            proposed += 1
            draft_ids.append(draft_token)
            draft_context = _append(draft_context, draft_token, _device_of(draft_bundle))
            if eos_token_id is not None and draft_token == eos_token_id:
                break

        if not draft_ids:
            break

        synchronize_device(device)
        verifier_tokens = _greedy_tokens_for_contexts(verifier_bundle, input_ids, draft_ids)
        synchronize_device(device)
        verifier_forward_calls += 1

        step_accepted = 0
        for index, (draft_token, verifier_token) in enumerate(
            zip(draft_ids, verifier_tokens, strict=True)
        ):
            if draft_token == verifier_token:
                accepted += 1
                step_accepted += 1
                recent_window.append(1)
                accepted_by_k[current_k] = accepted_by_k.get(current_k, 0) + 1
                input_ids = _append(input_ids, draft_token, device)
                if eos_token_id is not None and draft_token == eos_token_id:
                    break
            else:
                rejected += 1
                recent_window.append(0)
                rejected_by_k[current_k] = rejected_by_k.get(current_k, 0) + 1
                wasted += len(draft_ids) - index
                input_ids = _append(input_ids, verifier_token, device)
                break
            if (input_ids.shape[-1] - prompt_len) >= max_new_tokens:
                break

        if step_accepted == len(draft_ids):
            wasted += 0
        if eos_token_id is not None and int(input_ids[0, -1].item()) == eos_token_id:
            break

    synchronize_device(device)
    total_ms = elapsed_ms(start)
    generated_ids = input_ids[0, prompt_len:].detach().cpu().tolist()
    generated = len(generated_ids)
    average_selected_k = (
        sum(selected_k_per_step) / len(selected_k_per_step) if selected_k_per_step else 0.0
    )
    return AdaptiveSpeculativeDecodeResult(
        output_token_ids=generated_ids,
        output_text=_decode(verifier_bundle.tokenizer, generated_ids),
        generated_tokens=generated,
        total_latency_ms=total_ms,
        tokens_per_second=(generated / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        verifier_forward_calls=verifier_forward_calls,
        draft_forward_calls=draft_forward_calls,
        draft_tokens_proposed=proposed,
        draft_tokens_accepted=accepted,
        draft_tokens_rejected=rejected,
        acceptance_rate=accepted / proposed if proposed else 0.0,
        draft_overhead_ms=draft_overhead_ms,
        schedule_type="adaptive",
        adaptive_policy=policy,
        selected_k_per_step=selected_k_per_step,
        average_selected_k=average_selected_k,
        min_selected_k=min(selected_k_per_step) if selected_k_per_step else 0,
        max_selected_k=max(selected_k_per_step) if selected_k_per_step else 0,
        confidence_per_step=confidence_per_step,
        entropy_per_step=entropy_per_step,
        recent_acceptance_per_step=recent_acceptance_per_step,
        accepted_tokens_by_k=accepted_by_k,
        rejected_tokens_by_k=rejected_by_k,
        wasted_draft_tokens=wasted,
    )
