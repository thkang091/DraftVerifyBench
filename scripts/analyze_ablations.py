from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.analysis import load_results


def _policy_label(row: pd.Series) -> str:
    if row.get("schedule_type") == "adaptive":
        return str(row.get("adaptive_policy") or "adaptive")
    if pd.notna(row.get("draft_k")):
        return f"static_k={int(row['draft_k'])}"
    return str(row.get("mode"))


def _table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    spec = df[df["mode"] == "speculative"].copy()
    spec["policy"] = spec.apply(_policy_label, axis=1)
    groups = group_cols or ["policy"]
    return (
        spec.groupby(groups, dropna=False)
        .agg(
            rows=("mode", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            p50_latency_ms=("total_latency_ms", "median"),
            p95_latency_ms=("total_latency_ms", lambda values: values.quantile(0.95)),
            slowdown_rate=("slowdown_flag", "mean"),
            wasted_draft_tokens=("wasted_draft_tokens", "mean"),
            acceptance_rate=("acceptance_rate", "mean"),
            output_equivalence_rate=("exact_output_match_with_baseline", "mean"),
        )
        .reset_index()
        .sort_values(["median_speedup", "mean_speedup"], ascending=False)
    )


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows)
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze(input_path: str | Path) -> tuple[pd.DataFrame, str, str]:
    df = load_results([input_path])
    summary = _table(df, ["policy"])
    by_prompt = _table(df, ["prompt_type", "policy"])
    by_temp = _table(df, ["temperature", "policy"])
    entropy_col = "expected_entropy_level"
    by_entropy = (
        _table(df, [entropy_col, "policy"]) if entropy_col in df.columns else pd.DataFrame()
    )
    by_model = _table(df, ["model_pair", "policy"])
    spec = df[df["mode"] == "speculative"].copy()
    spec["policy"] = spec.apply(_policy_label, axis=1)
    failures = spec.sort_values("speedup_vs_baseline", ascending=True).head(30)
    md = "\n\n".join(
        [
            "# Ablation Analysis",
            "No result should be treated as a claim unless generated from a completed CSV.",
            "## Overall Policy Summary",
            _md_table(summary),
            "## Best Policy By Prompt Type",
            _md_table(by_prompt),
            "## Best Policy By Temperature",
            _md_table(by_temp),
            "## Best Policy By Entropy Level",
            _md_table(by_entropy),
            "## Best Policy By Model Pair",
            _md_table(by_model),
        ]
    )
    failure_md = "\n\n".join(["# Ablation Failure Cases", _md_table(failures)])
    rec_md = "\n\n".join(
        [
            "# Ablation Recommendations",
            "Use the top policy only for conditions where repeated runs show speedup and "
            "acceptable tail latency. Prefer disabling speculation when slowdown rate remains "
            "high.",
            _md_table(summary.head(10)),
        ]
    )
    return summary, md, failure_md + "\n\n" + rec_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze static/adaptive ablation results.")
    parser.add_argument("--input")
    parser.add_argument("--results")
    parser.add_argument("--out", default="results/ablation_analysis.md")
    parser.add_argument("--summary-out", default="results/ablation_summary.csv")
    parser.add_argument("--failure-out", default="results/ablation_failure_cases.md")
    parser.add_argument("--recommendations-out", default="results/ablation_recommendations.md")
    args = parser.parse_args()
    input_path = args.results or args.input
    if input_path is None:
        raise SystemExit("--results or --input is required")
    summary, md, combined = analyze(input_path)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    Path(args.out).write_text(md, encoding="utf-8")
    failure, recommendations = combined.split("# Ablation Recommendations", maxsplit=1)
    Path(args.failure_out).write_text(failure.strip() + "\n", encoding="utf-8")
    Path(args.recommendations_out).write_text(
        "# Ablation Recommendations" + recommendations,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
