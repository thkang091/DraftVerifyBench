from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from draftverifybench.cli import build_parser
from draftverifybench.config import load_config
from draftverifybench.routing import (
    PromptFeatures,
    feature_threshold_decision,
    lexical_prompt_features,
    route_prompt,
)


def test_routing_feature_extraction() -> None:
    features = lexical_prompt_features("Complete this function: def add(a, b):", "code_completion")
    assert features.prompt_length_tokens > 0
    assert features.code_like_markers >= 1
    assert features.punctuation_ratio > 0


def test_router_threshold_decisions() -> None:
    risky = PromptFeatures(
        prompt_length_tokens=12,
        prompt_type="open_ended",
        punctuation_ratio=0.01,
        code_like_markers=0,
        json_like_markers=0,
        first_token_entropy=8.0,
        draft_top1_probability=0.05,
    )
    assert feature_threshold_decision(risky) == "baseline"
    assert route_prompt("always_speculative_best_static", risky) == "static_speculative"
    assert route_prompt("always_baseline", risky) == "baseline"


def test_cli_help_works() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "draftverifybench" in help_text


def test_quickstart_config_loads() -> None:
    config = load_config("examples/quickstart_config.yaml")
    assert config.draft_model == "distilgpt2"
    assert config.max_new_tokens == 16


def test_backend_stubs_raise() -> None:
    from draftverifybench.backends.sglang_backend_stub import SGLangBackend
    from draftverifybench.backends.vllm_backend_stub import VLLMBackend

    with pytest.raises(NotImplementedError):
        VLLMBackend()
    with pytest.raises(NotImplementedError):
        SGLangBackend()


def test_ablation_analysis_on_fake_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "ablation.csv"
    pd.DataFrame(
        [
            {
                "mode": "speculative",
                "prompt_type": "factual_qa",
                "expected_entropy_level": "medium",
                "model_pair": "a->b",
                "temperature": 0.0,
                "draft_k": 4,
                "schedule_type": "static",
                "adaptive_policy": "",
                "total_latency_ms": 100.0,
                "speedup_vs_baseline": 1.1,
                "slowdown_flag": False,
                "wasted_draft_tokens": 2,
                "acceptance_rate": 0.7,
                "exact_output_match_with_baseline": True,
                "error": "",
            }
        ]
    ).to_csv(csv_path, index=False)
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_ablations.py",
            "--input",
            str(csv_path),
            "--out",
            str(tmp_path / "analysis.md"),
            "--summary-out",
            str(tmp_path / "summary.csv"),
            "--failure-out",
            str(tmp_path / "failures.md"),
            "--recommendations-out",
            str(tmp_path / "recommendations.md"),
        ],
        check=True,
    )
    assert (tmp_path / "summary.csv").exists()


def test_router_analysis_on_fake_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "router.csv"
    pd.DataFrame(
        [
            {
                "mode": "router",
                "router_policy": "feature_threshold_router",
                "router_decision": "baseline",
                "total_latency_ms": 100.0,
                "speedup_vs_baseline": 1.0,
                "slowdown_flag": False,
                "error": "",
            }
        ]
    ).to_csv(csv_path, index=False)
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_router.py",
            "--input",
            str(csv_path),
            "--out",
            str(tmp_path / "router.md"),
            "--summary-out",
            str(tmp_path / "router.csv"),
            "--examples-out",
            str(tmp_path / "examples.md"),
        ],
        check=True,
    )
    assert (tmp_path / "router.csv").exists()


def test_cli_run_with_mocked_runner(monkeypatch, tmp_path: Path) -> None:
    from draftverifybench import cli

    called = {}

    def fake_run(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return []

    monkeypatch.setattr(cli, "run_benchmark", fake_run)
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            "examples/quickstart_config.yaml",
            "--out",
            str(tmp_path / "out.csv"),
        ]
    )
    args.func(args)
    assert called["kwargs"]["out"].endswith("out.csv")


def test_package_import_works() -> None:
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", "import draftverifybench; print(draftverifybench.__all__)"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "routing" in result.stdout or "adaptive" in result.stdout

