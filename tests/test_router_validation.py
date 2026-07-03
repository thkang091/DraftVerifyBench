from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_router_from_grid import (
    POST_GENERATION_FIELDS,
    ROUTER_FEATURES,
    validate_router,
)

RESULT_PATHS = {
    "qwen_seed42": Path("results/gpu_qwen_full_static_results.csv"),
    "qwen_seed43": Path("results/gpu_qwen_full_static_seed43_results.csv"),
    "llama_seed42": Path("results/gpu_llama_full_static_results.csv"),
    "llama_seed43": Path("results/gpu_llama_full_static_seed43_results.csv"),
}


def test_router_features_do_not_use_post_generation_fields() -> None:
    assert not set(ROUTER_FEATURES).intersection(POST_GENERATION_FIELDS)
    assert ROUTER_FEATURES == ["model_family", "prompt_type"]


def test_router_summary_reconciles_with_routed_rows() -> None:
    missing = [str(path) for path in RESULT_PATHS.values() if not path.exists()]
    if missing:
        pytest.skip(f"Missing GH200 result files: {missing}")

    summary, _by_prompt, _decisions, routed = validate_router(RESULT_PATHS)

    for row in summary.itertuples(index=False):
        subset = routed[
            (routed["model_family"] == row.model_family)
            & (routed["router_policy"] == row.router_policy)
        ]
        assert len(subset) == row.rows
        assert abs((subset["router_speedup"] < 1.0).mean() - row.slowdown_rate) < 1e-12
        assert abs(subset["router_used_speculation"].mean() - row.speculative_share) < 1e-12

        speculated = subset[subset["router_used_speculation"]]
        expected_speculated_slowdown = (
            float((speculated["router_speedup"] < 1.0).mean()) if not speculated.empty else 0.0
        )
        assert abs(expected_speculated_slowdown - row.speculated_slowdown_rate) < 1e-12


def test_llama_router_is_ev_positive_not_per_decision_positive() -> None:
    missing = [str(path) for path in RESULT_PATHS.values() if not path.exists()]
    if missing:
        pytest.skip(f"Missing GH200 result files: {missing}")

    summary, _by_prompt, _decisions, _routed = validate_router(RESULT_PATHS)
    llama = summary[
        (summary["model_family"] == "llama")
        & (summary["router_policy"] == "trained_regime_router")
    ].iloc[0]

    assert llama["mean_speedup"] > 1.0
    assert llama["speculated_mean_speedup"] > 1.0
    assert llama["speculated_slowdown_rate"] > 0.5
    assert llama["slowdown_rate"] < 0.45
