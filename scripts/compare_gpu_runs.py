from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RUNS = {
    "qwen_0.5b_to_7b": "results/gpu_qwen_reduced_results.csv",
    "llama_1b_to_8b": "results/gpu_llama_reduced_results.csv",
}


def _read(path: str | Path, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run"] = run_name
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()
    for column in [
        "acceptance_rate",
        "draft_k",
        "draft_overhead_ms",
        "latency_ms",
        "speedup_vs_baseline",
        "temperature",
        "verifier_calls_per_output_token",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _speculative(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["mode"] == "speculative"].copy()


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    subset = df[[left, right]].dropna()
    if len(subset) < 2 or subset[left].nunique() < 2 or subset[right].nunique() < 2:
        return float("nan")
    return float(subset[left].corr(subset[right]))


def _run_summary(run_name: str, df: pd.DataFrame) -> dict[str, Any]:
    spec = _speculative(df)
    speed = spec["speedup_vs_baseline"]
    fast = spec[speed > 1.0]
    static = spec[spec["schedule_type"] == "static"] if "schedule_type" in spec.columns else spec
    best_static_k = None
    if not static.empty and "draft_k" in static.columns:
        by_k = static.dropna(subset=["draft_k"]).groupby("draft_k")["speedup_vs_baseline"].mean()
        if not by_k.empty:
            best_static_k = int(by_k.idxmax())

    return {
        "run": run_name,
        "rows": int(len(df)),
        "baseline_rows": int((df["mode"] == "baseline").sum()),
        "speculative_rows": int(len(spec)),
        "mean_speedup": float(speed.mean()),
        "median_speedup": float(speed.median()),
        "best_speedup": float(speed.max()),
        "worst_speedup": float(speed.min()),
        "slowdown_rate": float((speed < 1.0).mean()),
        "faster_than_baseline_rows": int(len(fast)),
        "mean_acceptance_rate": float(spec["acceptance_rate"].mean()),
        "acceptance_speedup_corr": _safe_corr(spec, "acceptance_rate", "speedup_vs_baseline"),
        "draft_overhead_speedup_corr": _safe_corr(spec, "draft_overhead_ms", "speedup_vs_baseline"),
        "verifier_calls_speedup_corr": _safe_corr(
            spec, "verifier_calls_per_output_token", "speedup_vs_baseline"
        ),
        "best_static_k_by_mean_speedup": best_static_k,
    }


def _group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    spec = _speculative(df)
    return (
        spec.groupby(group_cols, dropna=False)
        .agg(
            rows=("speedup_vs_baseline", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            best_speedup=("speedup_vs_baseline", "max"),
            worst_speedup=("speedup_vs_baseline", "min"),
            slowdown_rate=("speedup_vs_baseline", lambda values: float((values < 1.0).mean())),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_draft_overhead_ms=("draft_overhead_ms", "mean"),
            verifier_calls_per_output_token=("verifier_calls_per_output_token", "mean"),
        )
        .reset_index()
    )


def _case_table(df: pd.DataFrame, *, fastest: bool, limit: int) -> pd.DataFrame:
    spec = _speculative(df)
    columns = [
        "run",
        "prompt_id",
        "prompt_type",
        "temperature",
        "draft_k",
        "schedule_type",
        "adaptive_policy",
        "latency_ms",
        "speedup_vs_baseline",
        "acceptance_rate",
        "draft_overhead_ms",
        "verifier_calls_per_output_token",
        "draft_forward_calls",
        "verifier_forward_calls",
    ]
    available = [column for column in columns if column in spec.columns]
    return spec.sort_values("speedup_vs_baseline", ascending=not fastest)[available].head(limit)


def write_gpu_comparison(
    *,
    qwen_results: str | Path,
    llama_results: str | Path,
    out_dir: str | Path,
    case_limit: int,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = {
        "qwen_0.5b_to_7b": _read(qwen_results, "qwen_0.5b_to_7b"),
        "llama_1b_to_8b": _read(llama_results, "llama_1b_to_8b"),
    }
    combined = pd.concat(runs.values(), ignore_index=True)

    pd.DataFrame([_run_summary(name, df) for name, df in runs.items()]).to_csv(
        out / "gpu_model_pair_comparison.csv", index=False
    )
    _group_summary(combined, ["run", "prompt_type"]).to_csv(
        out / "gpu_prompt_type_comparison.csv", index=False
    )
    _group_summary(combined, ["run", "draft_k"]).to_csv(
        out / "gpu_draft_k_comparison.csv", index=False
    )
    _group_summary(combined, ["run", "schedule_type"]).to_csv(
        out / "gpu_schedule_comparison.csv", index=False
    )

    correlation_rows = []
    for run_name, df in runs.items():
        spec = _speculative(df)
        for metric in [
            "acceptance_rate",
            "draft_overhead_ms",
            "verifier_calls_per_output_token",
        ]:
            correlation_rows.append(
                {
                    "run": run_name,
                    "metric": f"{metric}_vs_speedup",
                    "correlation": _safe_corr(spec, metric, "speedup_vs_baseline"),
                }
            )
    pd.DataFrame(correlation_rows).to_csv(out / "gpu_correlation_summary.csv", index=False)

    _case_table(runs["llama_1b_to_8b"], fastest=True, limit=case_limit).to_csv(
        out / "gpu_llama_fast_cases.csv", index=False
    )
    _case_table(runs["llama_1b_to_8b"], fastest=False, limit=case_limit).to_csv(
        out / "gpu_llama_slow_cases.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare A100 Qwen and Llama GPU validation runs.")
    parser.add_argument("--qwen-results", default=DEFAULT_RUNS["qwen_0.5b_to_7b"])
    parser.add_argument("--llama-results", default=DEFAULT_RUNS["llama_1b_to_8b"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--case-limit", type=int, default=20)
    args = parser.parse_args()
    write_gpu_comparison(
        qwen_results=args.qwen_results,
        llama_results=args.llama_results,
        out_dir=args.out_dir,
        case_limit=args.case_limit,
    )


if __name__ == "__main__":
    main()
