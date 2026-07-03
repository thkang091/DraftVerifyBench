from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from draftverifybench.decoding import _encode


@dataclass(frozen=True)
class PromptFeatures:
    prompt_length_tokens: int
    prompt_type: str
    punctuation_ratio: float
    code_like_markers: int
    json_like_markers: int
    first_token_entropy: float | None = None
    draft_top1_probability: float | None = None
    draft_top5_probability_mass: float | None = None
    dry_run_entropy_3: float | None = None


CODE_PATTERNS = [r"\bdef\b", r"\bclass\b", r"\bSELECT\b", r"\bFROM\b", r"[{}();]"]
JSON_PATTERNS = [r"\bJSON\b", r"\{", r"\}", r":", r"\b invoice\b", r"\b receipt\b"]


def lexical_prompt_features(prompt: str, prompt_type: str) -> PromptFeatures:
    tokens = prompt.split()
    punctuation = sum(1 for char in prompt if not char.isalnum() and not char.isspace())
    code_markers = sum(1 for pattern in CODE_PATTERNS if re.search(pattern, prompt, re.I))
    json_markers = sum(1 for pattern in JSON_PATTERNS if re.search(pattern, prompt, re.I))
    return PromptFeatures(
        prompt_length_tokens=len(tokens),
        prompt_type=prompt_type,
        punctuation_ratio=punctuation / max(len(prompt), 1),
        code_like_markers=code_markers,
        json_like_markers=json_markers,
    )


def _base_feature_dict(features: PromptFeatures) -> dict[str, Any]:
    data = features.__dict__.copy()
    for key in [
        "first_token_entropy",
        "draft_top1_probability",
        "draft_top5_probability_mass",
        "dry_run_entropy_3",
    ]:
        data.pop(key, None)
    return data


@torch.inference_mode()
def draft_distribution_features(
    draft_bundle: Any,
    prompt: str,
    prompt_type: str,
    *,
    dry_run_tokens: int = 0,
) -> PromptFeatures:
    base = lexical_prompt_features(prompt, prompt_type)
    if hasattr(draft_bundle.model, "next_token"):
        return PromptFeatures(
            **_base_feature_dict(base),
            first_token_entropy=0.0,
            draft_top1_probability=1.0,
            draft_top5_probability_mass=1.0,
            dry_run_entropy_3=0.0 if dry_run_tokens else None,
        )
    device = getattr(draft_bundle, "device", torch.device("cpu"))
    input_ids = _encode(draft_bundle.tokenizer, prompt, device)
    entropies: list[float] = []
    top1 = None
    top5 = None
    for index in range(max(1, dry_run_tokens)):
        outputs = draft_bundle.model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).item())
        values, indices = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
        if index == 0:
            top1 = float(values[0, 0].item())
            top5 = float(values[0].sum().item())
        entropies.append(entropy)
        next_token = indices[:, :1].to(input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return PromptFeatures(
        **_base_feature_dict(base),
        first_token_entropy=entropies[0],
        draft_top1_probability=top1,
        draft_top5_probability_mass=top5,
        dry_run_entropy_3=float(sum(entropies[:3]) / min(len(entropies), 3))
        if dry_run_tokens
        else None,
    )


def feature_threshold_decision(
    features: PromptFeatures,
    *,
    entropy_threshold: float = 5.0,
    top1_threshold: float = 0.25,
    long_prompt_tokens: int = 96,
) -> str:
    entropy = features.dry_run_entropy_3 or features.first_token_entropy
    top1 = features.draft_top1_probability
    if entropy is not None and entropy > entropy_threshold:
        return "baseline"
    if top1 is not None and top1 < top1_threshold:
        return "baseline"
    if features.prompt_length_tokens > long_prompt_tokens and features.punctuation_ratio > 0.12:
        return "baseline"
    if features.code_like_markers or features.json_like_markers:
        return "static_speculative"
    return "adaptive_speculative"


def route_prompt(
    policy: str,
    features: PromptFeatures,
    *,
    entropy_threshold: float = 5.0,
    top1_threshold: float = 0.25,
) -> str:
    if policy == "always_baseline":
        return "baseline"
    if policy == "always_speculative_best_static":
        return "static_speculative"
    if policy == "always_adaptive":
        return "adaptive_speculative"
    if policy == "feature_threshold_router":
        return feature_threshold_decision(
            features,
            entropy_threshold=entropy_threshold,
            top1_threshold=top1_threshold,
        )
    if policy == "logistic_regression_router":
        raise NotImplementedError(
            "Logistic regression router is not enabled; use threshold router."
        )
    raise ValueError(f"Unknown router policy: {policy}")


def router_risk_score(features: PromptFeatures) -> float:
    entropy = features.dry_run_entropy_3 or features.first_token_entropy or 0.0
    top1 = features.draft_top1_probability or 0.0
    score = 0.0
    score += min(entropy / 10.0, 1.0) * 0.5
    score += max(0.0, 0.5 - top1) * 0.8
    score += min(features.punctuation_ratio, 0.25)
    score += 0.05 * math.log1p(features.prompt_length_tokens)
    return min(score, 1.0)
