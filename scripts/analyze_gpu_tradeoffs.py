from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUNS = {
    "qwen_0.5b_to_7b": "results/gpu_qwen_reduced_results.csv",
    "llama_1b_to_8b": "results/gpu_llama_reduced_results.csv",
}


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._\n"
    display = df.head(max_rows).copy()
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def _read(path: str | Path, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run"] = run_name
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()
    for column in [
        "acceptance_rate",
        "draft_forward_calls",
        "draft_k",
        "draft_overhead_ms",
        "generated_tokens",
        "latency_ms",
        "speedup_vs_baseline",
        "temperature",
        "total_latency_ms",
        "verifier_calls_per_output_token",
        "verifier_forward_calls",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _attach_baseline(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["run", "prompt_id", "temperature", "repetition"]
    baseline = (
        df[df["mode"] == "baseline"][key_cols + ["total_latency_ms", "tokens_per_second"]]
        .rename(
            columns={
                "total_latency_ms": "baseline_latency_ms",
                "tokens_per_second": "baseline_tokens_per_second",
            }
        )
        .copy()
    )
    spec = df[df["mode"] == "speculative"].copy()
    return spec.merge(baseline, on=key_cols, how="left", validate="many_to_one")


def build_tradeoff_rows(paths: dict[str, str | Path]) -> pd.DataFrame:
    frames = [_read(path, run_name) for run_name, path in paths.items()]
    spec = _attach_baseline(pd.concat(frames, ignore_index=True))
    spec["candidate_latency_ms"] = spec["total_latency_ms"]
    spec["latency_gap_ms"] = spec["candidate_latency_ms"] - spec["baseline_latency_ms"]
    spec["draft_overhead_share"] = spec["draft_overhead_ms"] / spec["candidate_latency_ms"]
    spec["non_draft_latency_ms"] = spec["candidate_latency_ms"] - spec["draft_overhead_ms"]
    spec["oracle_speedup_if_draft_free"] = (
        spec["baseline_latency_ms"] / spec["non_draft_latency_ms"].clip(lower=1e-9)
    )
    spec["break_even_draft_overhead_ms"] = (
        spec["baseline_latency_ms"] - spec["non_draft_latency_ms"]
    )
    spec["break_even_gap_ms"] = (
        spec["draft_overhead_ms"] - spec["break_even_draft_overhead_ms"]
    )
    spec["draft_overhead_reduction_needed_ms"] = spec["latency_gap_ms"].clip(lower=0)
    spec["draft_overhead_reduction_needed_pct"] = (
        spec["draft_overhead_reduction_needed_ms"] / spec["draft_overhead_ms"]
    )
    spec["would_win_if_draft_free"] = spec["oracle_speedup_if_draft_free"] >= 1.0
    spec["is_faster_than_baseline"] = spec["speedup_vs_baseline"] >= 1.0
    spec["static_or_adaptive"] = spec["schedule_type"].fillna("unknown")
    return spec


def build_summary(tradeoffs: pd.DataFrame) -> pd.DataFrame:
    return (
        tradeoffs.groupby("run", dropna=False)
        .agg(
            rows=("speedup_vs_baseline", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            best_speedup=("speedup_vs_baseline", "max"),
            slowdown_rate=("is_faster_than_baseline", lambda values: float((~values).mean())),
            mean_draft_overhead_ms=("draft_overhead_ms", "mean"),
            mean_draft_overhead_share=("draft_overhead_share", "mean"),
            mean_latency_gap_ms=("latency_gap_ms", "mean"),
            mean_oracle_speedup_if_draft_free=("oracle_speedup_if_draft_free", "mean"),
            oracle_win_rate_if_draft_free=("would_win_if_draft_free", "mean"),
            mean_break_even_gap_ms=("break_even_gap_ms", "mean"),
        )
        .reset_index()
    )


def write_outputs(
    *,
    qwen_results: str | Path,
    llama_results: str | Path,
    out_dir: str | Path,
    report_out: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "qwen_0.5b_to_7b": qwen_results,
        "llama_1b_to_8b": llama_results,
    }
    tradeoffs = build_tradeoff_rows(paths)
    summary = build_summary(tradeoffs)

    columns = [
        "run",
        "prompt_id",
        "prompt_type",
        "temperature",
        "draft_k",
        "schedule_type",
        "adaptive_policy",
        "baseline_latency_ms",
        "candidate_latency_ms",
        "speedup_vs_baseline",
        "latency_gap_ms",
        "draft_overhead_ms",
        "draft_overhead_share",
        "oracle_speedup_if_draft_free",
        "break_even_draft_overhead_ms",
        "break_even_gap_ms",
        "acceptance_rate",
        "verifier_calls_per_output_token",
        "draft_forward_calls",
        "verifier_forward_calls",
        "exact_output_match_with_baseline",
    ]
    available = [column for column in columns if column in tradeoffs.columns]

    tradeoffs[available].to_csv(out / "gpu_tradeoff_decomposition.csv", index=False)
    summary.to_csv(out / "gpu_tradeoff_summary.csv", index=False)
    tradeoffs.sort_values("break_even_gap_ms").head(30)[available].to_csv(
        out / "gpu_closest_to_break_even_cases.csv", index=False
    )
    tradeoffs.sort_values("oracle_speedup_if_draft_free", ascending=False).head(30)[
        available
    ].to_csv(out / "gpu_oracle_best_cases.csv", index=False)

    by_policy = (
        tradeoffs.groupby(["run", "schedule_type", "draft_k"], dropna=False)
        .agg(
            rows=("speedup_vs_baseline", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            mean_draft_overhead_ms=("draft_overhead_ms", "mean"),
            mean_draft_overhead_share=("draft_overhead_share", "mean"),
            mean_oracle_speedup_if_draft_free=("oracle_speedup_if_draft_free", "mean"),
            mean_break_even_gap_ms=("break_even_gap_ms", "mean"),
            oracle_win_rate_if_draft_free=("would_win_if_draft_free", "mean"),
        )
        .reset_index()
    )
    by_policy.to_csv(out / "gpu_tradeoff_by_policy.csv", index=False)

    report = [
        "# GPU Tradeoff Decomposition",
        "",
        "This report decomposes the A100 Qwen/Llama reduced runs using existing result CSVs. It "
        "does not require a GPU to regenerate.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## By Policy",
        "",
        _markdown_table(by_policy, max_rows=40),
        "",
        "## Closest To Break-Even",
        "",
        "Rows with the smallest `break_even_gap_ms` are closest to crossing `1.0x`. Negative "
        "values mean the row already beat baseline.",
        "",
        _markdown_table(
            tradeoffs.sort_values("break_even_gap_ms")[
                [
                    "run",
                    "prompt_id",
                    "prompt_type",
                    "temperature",
                    "draft_k",
                    "schedule_type",
                    "speedup_vs_baseline",
                    "latency_gap_ms",
                    "draft_overhead_ms",
                    "break_even_gap_ms",
                    "oracle_speedup_if_draft_free",
                    "acceptance_rate",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## Oracle Draft-Free Upper Bound",
        "",
        "`oracle_speedup_if_draft_free` estimates how fast the speculative row would have been if "
        "measured draft overhead were removed while keeping all non-draft costs unchanged. This is "
        "not a claim about achievable serving performance; it is a diagnostic upper bound.",
        "",
        _markdown_table(
            tradeoffs.sort_values("oracle_speedup_if_draft_free", ascending=False)[
                [
                    "run",
                    "prompt_id",
                    "prompt_type",
                    "temperature",
                    "draft_k",
                    "schedule_type",
                    "speedup_vs_baseline",
                    "draft_overhead_share",
                    "oracle_speedup_if_draft_free",
                    "verifier_calls_per_output_token",
                    "acceptance_rate",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## Interpretation",
        "",
        "- Qwen remained below break-even even though acceptance was often high.",
        "- Llama had a small number of strong wins, but most rows still slowed down.",
        "- Draft overhead was a major negative driver in both model families.",
        "- If the draft path were free, many more rows would cross break-even; that gap motivates "
        "profiling and production-backend comparisons.",
        "",
    ]
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose GPU speculative decoding tradeoffs.")
    parser.add_argument("--qwen-results", default=DEFAULT_RUNS["qwen_0.5b_to_7b"])
    parser.add_argument("--llama-results", default=DEFAULT_RUNS["llama_1b_to_8b"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--report-out", default="docs/GPU_Tradeoff_Decomposition.md")
    args = parser.parse_args()
    write_outputs(
        qwen_results=args.qwen_results,
        llama_results=args.llama_results,
        out_dir=args.out_dir,
        report_out=args.report_out,
    )


if __name__ == "__main__":
    main()
