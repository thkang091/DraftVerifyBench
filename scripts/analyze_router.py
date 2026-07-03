from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.analysis import load_results


def router_summary(df: pd.DataFrame) -> pd.DataFrame:
    routed = df[df["mode"] == "router"].copy()
    summary = (
        routed.groupby(["router_policy", "router_decision"], dropna=False)
        .agg(
            rows=("mode", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            p95_latency_ms=("total_latency_ms", lambda values: values.quantile(0.95)),
            slowdown_rate=("slowdown_flag", "mean"),
            routed_share=("mode", lambda values: len(values) / max(len(routed), 1)),
        )
        .reset_index()
    )
    total_by_policy = routed.groupby("router_policy")["mode"].count().to_dict()
    summary["routed_share_within_policy"] = summary.apply(
        lambda row: row["rows"] / max(total_by_policy.get(row["router_policy"], 1), 1),
        axis=1,
    )
    summary["false_positive_route_rate"] = summary.apply(
        lambda row: row["slowdown_rate"] if row["router_decision"] != "baseline" else 0.0,
        axis=1,
    )
    # False negatives require paired counterfactual rows; computed in aggregate below when possible.
    summary["false_negative_route_rate"] = 0.0
    baseline_rows = routed[routed["router_decision"] == "baseline"]
    speculative_rows = routed[routed["router_decision"] != "baseline"]
    if not baseline_rows.empty and not speculative_rows.empty:
        key = ["prompt_id", "temperature", "repetition"]
        helped = speculative_rows.groupby(key)["speedup_vs_baseline"].max() > 1.0
        baseline_keys = baseline_rows.set_index(key).index
        false_negatives = sum(bool(helped.get(item, False)) for item in baseline_keys)
        false_negative_rate = false_negatives / max(len(baseline_rows), 1)
        mask = summary["router_decision"] == "baseline"
        summary.loc[mask, "false_negative_route_rate"] = false_negative_rate
    return summary


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]) for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prompt-router results.")
    parser.add_argument("--input")
    parser.add_argument("--results")
    parser.add_argument("--out", default="results/router_analysis.md")
    parser.add_argument("--summary-out", default="results/router_summary.csv")
    parser.add_argument("--examples-out", default="results/router_decision_examples.md")
    args = parser.parse_args()
    input_path = args.results or args.input
    if input_path is None:
        raise SystemExit("--results or --input is required")
    df = load_results([input_path])
    summary = router_summary(df)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    Path(args.out).write_text(
        "# Router Analysis\n\n"
        "Compare always-baseline, always-speculative, always-adaptive, and threshold router "
        "runs only after their CSVs exist.\n\n"
        + _md_table(summary),
        encoding="utf-8",
    )
    examples = df[df["mode"] == "router"].head(30)
    Path(args.examples_out).write_text(
        "# Router Decision Examples\n\n" + _md_table(examples),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
