from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from draftverifybench.analysis import (
    build_summary_tables,
    correlation_analysis,
    load_results,
    output_equivalence,
    slowdown_rates,
)


def _fake_results(path: Path) -> None:
    rows = [
        {
            "mode": "baseline",
            "prompt_id": "p1",
            "prompt_type": "structured_json",
            "temperature": 0.0,
            "draft_k": "",
            "repetition": 0,
            "total_latency_ms": 100.0,
            "tokens_per_second": 10.0,
            "generated_tokens": 4,
            "output_length": 4,
            "verifier_forward_calls": 4,
            "verifier_calls_per_output_token": 1.0,
            "speedup_vs_baseline": "",
            "acceptance_rate": 0.0,
            "draft_overhead_ms": "",
            "slowdown_flag": False,
            "exact_output_match_with_baseline": "",
            "error": "",
        },
        {
            "mode": "speculative",
            "prompt_id": "p1",
            "prompt_type": "structured_json",
            "temperature": 0.0,
            "draft_k": 2,
            "repetition": 0,
            "total_latency_ms": 80.0,
            "tokens_per_second": 12.5,
            "generated_tokens": 4,
            "output_length": 4,
            "verifier_forward_calls": 2,
            "verifier_calls_per_output_token": 0.5,
            "speedup_vs_baseline": 1.25,
            "acceptance_rate": 0.75,
            "draft_overhead_ms": 20.0,
            "slowdown_flag": False,
            "exact_output_match_with_baseline": True,
            "error": "",
        },
        {
            "mode": "speculative",
            "prompt_id": "p2",
            "prompt_type": "open_ended",
            "temperature": 1.0,
            "draft_k": 4,
            "repetition": 0,
            "total_latency_ms": 120.0,
            "tokens_per_second": 8.0,
            "generated_tokens": 4,
            "output_length": 4,
            "verifier_forward_calls": 2,
            "verifier_calls_per_output_token": 0.5,
            "speedup_vs_baseline": 0.8,
            "acceptance_rate": 0.25,
            "draft_overhead_ms": 60.0,
            "slowdown_flag": True,
            "exact_output_match_with_baseline": "",
            "error": "",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_summary_table_generation(tmp_path: Path) -> None:
    csv_path = tmp_path / "fake.csv"
    _fake_results(csv_path)
    df = load_results([csv_path])
    tables = build_summary_tables(df)
    assert "01_overall_baseline_vs_speculative" in tables
    assert "08_slowdown_cases" in tables


def test_correlation_analysis_and_slowdown_detection(tmp_path: Path) -> None:
    csv_path = tmp_path / "fake.csv"
    _fake_results(csv_path)
    df = load_results([csv_path])
    corr = correlation_analysis(df)
    slowdowns = slowdown_rates(df, "prompt_type")
    assert "acceptance_rate_vs_speedup" in set(corr["metric"])
    assert slowdowns["slowdown_rate"].max() == 1.0


def test_output_equivalence_check(tmp_path: Path) -> None:
    csv_path = tmp_path / "fake.csv"
    _fake_results(csv_path)
    raw_path = tmp_path / "fake.jsonl"
    rows = [
        {
            "mode": "baseline",
            "prompt_id": "p1",
            "temperature": 0.0,
            "repetition": 0,
            "output_token_ids": [1, 2, 3, 4],
        },
        {
            "mode": "speculative",
            "prompt_id": "p1",
            "temperature": 0.0,
            "repetition": 0,
            "draft_k": 2,
            "output_token_ids": [1, 2, 3, 4],
        },
    ]
    raw_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary = output_equivalence(raw_path, csv_path)
    assert summary["exact_match_rate"] == 1.0
    assert summary["token_level_match_rate"] == 1.0


def test_plots_script_runs_on_tiny_fake_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "fake.csv"
    out_dir = tmp_path / "plots"
    _fake_results(csv_path)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    subprocess.run(
        [
            sys.executable,
            "scripts/make_plots.py",
            "--inputs",
            str(csv_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        env=env,
    )
    assert (out_dir / "speedup_by_draft_k.png").exists()
    assert (out_dir / "acceptance_vs_speedup.png").exists()

