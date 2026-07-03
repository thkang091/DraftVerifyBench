from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_results(inputs: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in inputs:
        frame = pd.read_csv(path)
        frame["source_file"] = Path(path).name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""].copy()
    numeric_cols = [
        "acceptance_rate",
        "draft_k",
        "draft_overhead_ms",
        "generated_tokens",
        "output_length",
        "speedup_vs_baseline",
        "temperature",
        "tokens_per_second",
        "total_latency_ms",
        "verifier_calls_per_output_token",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _speculative(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["mode"] == "speculative"].copy()


def _agg_spec(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    spec = _speculative(df)
    if spec.empty:
        return pd.DataFrame()
    return (
        spec.groupby(group_cols, dropna=False)
        .agg(
            rows=("total_latency_ms", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_latency_ms=("total_latency_ms", "mean"),
            mean_draft_overhead_ms=("draft_overhead_ms", "mean"),
            verifier_calls_per_output_token=("verifier_calls_per_output_token", "mean"),
            slowdown_rate=("slowdown_flag", "mean"),
        )
        .reset_index()
    )


def summarize_results(csv_path: str | Path) -> pd.DataFrame:
    df = load_results([csv_path])
    group_cols = ["prompt_type", "temperature", "draft_k"]
    return _agg_spec(df, group_cols)


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    spec = _speculative(df)
    baseline = df[df["mode"] == "baseline"].copy()
    overall = pd.DataFrame(
        [
            {
                "mode": "baseline",
                "rows": len(baseline),
                "mean_latency_ms": baseline["total_latency_ms"].mean(),
                "mean_tokens_per_second": baseline["tokens_per_second"].mean(),
            },
            {
                "mode": "speculative",
                "rows": len(spec),
                "mean_latency_ms": spec["total_latency_ms"].mean(),
                "mean_tokens_per_second": spec["tokens_per_second"].mean(),
                "mean_speedup": spec["speedup_vs_baseline"].mean(),
                "slowdown_rate": spec["slowdown_flag"].mean(),
            },
        ]
    )

    tables = {
        "01_overall_baseline_vs_speculative": overall,
        "02_speedup_by_prompt_type": _agg_spec(df, ["prompt_type"]),
        "03_speedup_by_temperature": _agg_spec(df, ["temperature"]),
        "04_speedup_by_draft_k": _agg_spec(df, ["draft_k"]),
        "05_acceptance_rate_by_prompt_type": _agg_spec(df, ["prompt_type"])[
            ["prompt_type", "rows", "mean_acceptance_rate"]
        ],
        "06_acceptance_rate_by_temperature": _agg_spec(df, ["temperature"])[
            ["temperature", "rows", "mean_acceptance_rate"]
        ],
        "07_acceptance_rate_by_draft_k": _agg_spec(df, ["draft_k"])[
            ["draft_k", "rows", "mean_acceptance_rate"]
        ],
        "08_slowdown_cases": spec.sort_values("speedup_vs_baseline", ascending=True).head(20),
        "09_best_speedup_cases": spec.sort_values("speedup_vs_baseline", ascending=False).head(20),
        "10_acceptance_rate_vs_speedup_correlation": pd.DataFrame(
            [{"correlation": safe_corr(spec, "acceptance_rate", "speedup_vs_baseline")}]
        ),
        "11_verifier_calls_per_output_token": _agg_spec(df, ["draft_k"])[
            ["draft_k", "rows", "verifier_calls_per_output_token"]
        ],
        "12_draft_overhead_vs_speedup": pd.DataFrame(
            [{"correlation": safe_corr(spec, "draft_overhead_ms", "speedup_vs_baseline")}]
        ),
    }
    return tables


def safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    subset = df[[left, right]].dropna()
    if len(subset) < 2 or subset[left].nunique() < 2 or subset[right].nunique() < 2:
        return float("nan")
    return float(subset[left].corr(subset[right]))


def slowdown_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    spec = _speculative(df)
    if spec.empty:
        return pd.DataFrame()
    return (
        spec.groupby(group_col, dropna=False)
        .agg(
            rows=("slowdown_flag", "count"),
            slowdown_rate=("slowdown_flag", "mean"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            mean_acceptance_rate=("acceptance_rate", "mean"),
        )
        .reset_index()
    )


def acceptance_threshold(df: pd.DataFrame) -> float | None:
    spec = _speculative(df).dropna(subset=["acceptance_rate", "speedup_vs_baseline"])
    if spec.empty or not (spec["speedup_vs_baseline"] > 1.0).any():
        return None
    rounded = np.arange(0.0, 1.01, 0.05)
    candidates: list[tuple[float, float, int]] = []
    for threshold in rounded:
        subset = spec[spec["acceptance_rate"] >= threshold]
        if len(subset) >= 3:
            candidates.append(
                (
                    float(threshold),
                    float((subset["speedup_vs_baseline"] > 1.0).mean()),
                    len(subset),
                )
            )
    viable = [item for item in candidates if item[1] >= 0.5]
    return viable[0][0] if viable else None


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    spec = _speculative(df)
    rows: list[dict[str, Any]] = [
        {
            "metric": "acceptance_rate_vs_speedup",
            "value": safe_corr(spec, "acceptance_rate", "speedup_vs_baseline"),
        },
        {
            "metric": "draft_k_vs_speedup",
            "value": safe_corr(spec, "draft_k", "speedup_vs_baseline"),
        },
        {
            "metric": "output_length_vs_speedup",
            "value": safe_corr(spec, "output_length", "speedup_vs_baseline"),
        },
    ]
    for group_col in ["temperature", "prompt_type", "draft_k"]:
        for row in slowdown_rates(df, group_col).to_dict(orient="records"):
            rows.append(
                {
                    "metric": f"slowdown_rate_by_{group_col}",
                    "condition": row[group_col],
                    "value": row["slowdown_rate"],
                    "rows": row["rows"],
                }
            )
    threshold = acceptance_threshold(df)
    rows.append({"metric": "acceptance_threshold_for_positive_speedup", "value": threshold})
    return pd.DataFrame(rows)


def output_equivalence(raw_jsonl: str | Path, results_csv: str | Path) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    with Path(raw_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_rows.append(json.loads(line))
    raw = pd.DataFrame(raw_rows)
    results = load_results([results_csv])
    greedy = results[results["temperature"] == 0.0]
    exact = greedy[greedy["mode"] == "speculative"]["exact_output_match_with_baseline"].dropna()

    token_rates = []
    diffs = []
    baseline_by_key: dict[tuple[Any, ...], list[int]] = {}
    for row in raw[raw["mode"] == "baseline"].to_dict(orient="records"):
        key = (row["prompt_id"], row["temperature"], row["repetition"])
        baseline_by_key[key] = row.get("output_token_ids", [])
    for row in raw[raw["mode"] == "speculative"].to_dict(orient="records"):
        if float(row["temperature"]) != 0.0:
            continue
        key = (row["prompt_id"], row["temperature"], row["repetition"])
        baseline_ids = baseline_by_key.get(key, [])
        spec_ids = row.get("output_token_ids", [])
        pairs = zip(baseline_ids, spec_ids, strict=False)
        denom = max(len(baseline_ids), len(spec_ids), 1)
        matches = sum(1 for left, right in pairs if left == right)
        token_rates.append(matches / denom)
        if baseline_ids != spec_ids:
            diffs.append(
                {
                    "prompt_id": row["prompt_id"],
                    "draft_k": row["draft_k"],
                    "repetition": row["repetition"],
                    "baseline_len": len(baseline_ids),
                    "speculative_len": len(spec_ids),
                }
            )
    return {
        "greedy_speculative_rows": int(len(exact)),
        "exact_match_rate": float(exact.astype(bool).mean()) if len(exact) else float("nan"),
        "token_level_match_rate": float(np.mean(token_rates)) if token_rates else float("nan"),
        "differences": diffs[:25],
    }
