from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from draftverifybench.adaptive import (
    AdaptivePolicyConfig,
    adaptive_speculative_decode,
    confidence_threshold_k,
    entropy_threshold_k,
    rolling_acceptance_k,
)
from draftverifybench.speculative import speculative_decode
from tests.helpers import TinyBundle


def test_confidence_threshold_chooses_correct_k() -> None:
    thresholds = {0.85: 8, 0.70: 4, 0.50: 2}
    assert confidence_threshold_k(0.9, thresholds) == 8
    assert confidence_threshold_k(0.75, thresholds) == 4
    assert confidence_threshold_k(0.55, thresholds) == 2
    assert confidence_threshold_k(0.2, thresholds) == 1


def test_entropy_threshold_chooses_correct_k() -> None:
    thresholds = {2.0: 8, 4.0: 4, 6.0: 2}
    assert entropy_threshold_k(1.0, thresholds) == 8
    assert entropy_threshold_k(3.0, thresholds) == 4
    assert entropy_threshold_k(5.0, thresholds) == 2
    assert entropy_threshold_k(8.0, thresholds) == 1


def test_rolling_acceptance_updates_k() -> None:
    assert rolling_acceptance_k(0.9, current_k=2, max_k=8) == 4
    assert rolling_acceptance_k(0.4, current_k=4, min_k=1) == 2
    assert rolling_acceptance_k(0.7, current_k=4) == 4


def test_adaptive_result_schema_includes_selected_k_history() -> None:
    result = adaptive_speculative_decode(
        TinyBundle("draft", [7, 8, 99]),
        TinyBundle("verifier", [7, 8, 99]),
        "prompt",
        max_new_tokens=4,
        policy="confidence_threshold",
        policy_config=AdaptivePolicyConfig(),
    )
    assert result.selected_k_per_step
    assert result.average_selected_k >= 1
    assert isinstance(result.accepted_tokens_by_k, dict)


def test_static_schedule_still_works() -> None:
    result = speculative_decode(
        TinyBundle("draft", [7, 8, 99]),
        TinyBundle("verifier", [7, 8, 99]),
        "prompt",
        max_new_tokens=4,
        draft_k=2,
    )
    assert result.output_token_ids == [7, 8, 99]


def test_adaptive_analysis_works_on_fake_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "adaptive.csv"
    rows = [
        {
            "mode": "speculative",
            "prompt_id": "p1",
            "prompt_type": "factual_qa",
            "temperature": 0.0,
            "draft_k": 4,
            "schedule_type": "static",
            "adaptive_policy": "",
            "total_latency_ms": 100.0,
            "speedup_vs_baseline": 1.2,
            "slowdown_flag": False,
            "wasted_draft_tokens": 2,
            "acceptance_rate": 0.7,
            "verifier_calls_per_output_token": 0.5,
            "draft_forward_calls": 10,
            "exact_output_match_with_baseline": True,
            "selected_k_per_step": "",
            "error": "",
        },
        {
            "mode": "speculative",
            "prompt_id": "p2",
            "prompt_type": "factual_qa",
            "temperature": 0.0,
            "draft_k": "",
            "schedule_type": "adaptive",
            "adaptive_policy": "confidence_threshold",
            "total_latency_ms": 90.0,
            "speedup_vs_baseline": 1.3,
            "slowdown_flag": False,
            "wasted_draft_tokens": 1,
            "acceptance_rate": 0.8,
            "verifier_calls_per_output_token": 0.4,
            "draft_forward_calls": 9,
            "exact_output_match_with_baseline": True,
            "selected_k_per_step": "[1, 2, 4]",
            "error": "",
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_adaptive.py",
            "--input",
            str(csv_path),
            "--out",
            str(tmp_path / "analysis.md"),
            "--comparison-out",
            str(tmp_path / "comparison.csv"),
            "--best-out",
            str(tmp_path / "best.csv"),
            "--failure-out",
            str(tmp_path / "failures.md"),
            "--plots-dir",
            str(tmp_path / "plots"),
        ],
        check=True,
    )
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "plots" / "adaptive_speedup_vs_static.png").exists()
