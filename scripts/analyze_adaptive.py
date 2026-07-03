from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.analysis import load_results


def _policy_label(row: pd.Series) -> str:
    if row.get("schedule_type") == "adaptive":
        return str(row.get("adaptive_policy"))
    return f"static_k={int(row.get('draft_k'))}"


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    spec = df[df["mode"] == "speculative"].copy()
    spec["policy"] = spec.apply(_policy_label, axis=1)
    return (
        spec.groupby("policy", dropna=False)
        .agg(
            rows=("mode", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            p95_latency_ms=("total_latency_ms", lambda values: values.quantile(0.95)),
            slowdown_rate=("slowdown_flag", "mean"),
            wasted_draft_tokens=("wasted_draft_tokens", "mean"),
            acceptance_rate=("acceptance_rate", "mean"),
            verifier_calls_per_output_token=("verifier_calls_per_output_token", "mean"),
            draft_forward_calls=("draft_forward_calls", "mean"),
            exact_match_rate=("exact_output_match_with_baseline", "mean"),
        )
        .reset_index()
        .sort_values("median_speedup", ascending=False)
    )


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_plots(spec: pd.DataFrame, comparison: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.set_index("policy")["median_speedup"].plot(kind="bar", title="Adaptive vs Static")
    plt.ylabel("Median speedup")
    plt.tight_layout()
    plt.savefig(out_dir / "adaptive_speedup_vs_static.png", dpi=160)
    plt.close()

    comparison.set_index("policy")["slowdown_rate"].plot(kind="bar", title="Slowdown Rate")
    plt.ylabel("Slowdown rate")
    plt.tight_layout()
    plt.savefig(out_dir / "adaptive_slowdown_rate.png", dpi=160)
    plt.close()

    comparison.set_index("policy")["wasted_draft_tokens"].plot(
        kind="bar",
        title="Wasted Draft Tokens",
    )
    plt.ylabel("Mean wasted draft tokens")
    plt.tight_layout()
    plt.savefig(out_dir / "wasted_tokens_by_policy.png", dpi=160)
    plt.close()

    if "selected_k_per_step" in spec.columns:
        adaptive = spec[spec["schedule_type"] == "adaptive"].dropna(subset=["selected_k_per_step"])
        if not adaptive.empty:
            history = ast.literal_eval(str(adaptive.iloc[0]["selected_k_per_step"]))
            pd.Series(history).plot(marker="o", title="Selected k Over Time Example")
            plt.xlabel("Speculative step")
            plt.ylabel("selected k")
            plt.tight_layout()
            plt.savefig(out_dir / "selected_k_over_time_example.png", dpi=160)
            plt.close()

    by_prompt = spec.groupby(["policy", "prompt_type"])["acceptance_rate"].mean().unstack(0)
    by_prompt.plot(kind="bar", title="Adaptive Acceptance by Prompt Type")
    plt.ylabel("Acceptance rate")
    plt.tight_layout()
    plt.savefig(out_dir / "adaptive_acceptance_by_prompt_type.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze adaptive DraftVerifyBench runs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="results/adaptive_analysis.md")
    parser.add_argument("--comparison-out", default="results/adaptive_comparison.csv")
    parser.add_argument("--best-out", default="results/adaptive_best_conditions.csv")
    parser.add_argument("--failure-out", default="results/adaptive_failure_cases.md")
    parser.add_argument("--plots-dir", default="results/plots")
    args = parser.parse_args()

    df = load_results([args.input])
    spec = df[df["mode"] == "speculative"].copy()
    if spec.empty:
        raise SystemExit("No speculative rows found.")
    spec["policy"] = spec.apply(_policy_label, axis=1)
    comparison = _summarize(df)
    Path(args.comparison_out).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison_out, index=False)

    best = spec.sort_values("speedup_vs_baseline", ascending=False).head(25)
    best.to_csv(args.best_out, index=False)
    failures = spec.sort_values("speedup_vs_baseline", ascending=True).head(25)

    md = [
        "# Adaptive Draft Length Analysis",
        "",
        "This report compares static draft lengths with adaptive scheduling policies.",
        "",
        "## Policy Comparison",
        "",
        _markdown_table(comparison),
    ]
    Path(args.out).write_text("\n".join(md), encoding="utf-8")

    failure_lines = ["# Adaptive Failure Cases", "", _markdown_table(failures)]
    Path(args.failure_out).write_text("\n".join(failure_lines), encoding="utf-8")
    _write_plots(spec, comparison, Path(args.plots_dir))


if __name__ == "__main__":
    main()

