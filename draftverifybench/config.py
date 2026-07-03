from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"^\$\{([^}:]+)(?::-([^}]+))?\}$")


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_PATTERN.match(value)
        if match:
            name, default = match.groups()
            return os.environ.get(name, default or value)
        return value
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    return value


class BenchmarkConfig(BaseModel):
    seed: int = 42
    device: str = "auto"
    dtype: str = "auto"
    draft_model: str = "distilgpt2"
    verifier_model: str = "gpt2"
    max_new_tokens: int = 64
    temperatures: list[float] = Field(default_factory=lambda: [0.0])
    draft_ks: list[int] = Field(default_factory=lambda: [1, 2, 4])
    prompt_types: list[str] = Field(default_factory=list)
    repetitions: int = 1
    batch_size: int = 1
    warmup_runs: int = 0
    torch_compile: bool = False
    local_files_only: bool = False
    validation_level: str = "local_debug"
    schedules: list[str] = Field(default_factory=lambda: ["static"])
    adaptive_policies: list[str] = Field(default_factory=list)
    adaptive_variants: list[dict[str, Any]] = Field(default_factory=list)
    router_policy: str = "feature_threshold_router"
    router_policies: list[str] = Field(default_factory=list)
    router_static_draft_k: int = 4
    router_adaptive_policy: str = "confidence_threshold"
    router_acceptance_threshold: float = 0.6
    router_entropy_threshold: float = 5.0
    router_top1_threshold: float = 0.25
    confidence_thresholds: dict[str, int] = Field(
        default_factory=lambda: {"0.85": 8, "0.70": 4, "0.50": 2}
    )
    entropy_thresholds: dict[str, int] = Field(
        default_factory=lambda: {"2.0": 8, "4.0": 4, "6.0": 2}
    )
    rolling_acceptance_window: int = 8
    rolling_acceptance_min_k: int = 1
    rolling_acceptance_max_k: int = 8


def load_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data = _expand_env_placeholders(data)
    return BenchmarkConfig(**data)
