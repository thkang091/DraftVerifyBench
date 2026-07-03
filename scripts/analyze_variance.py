from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: str | Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run"] = label
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()
    for column in [
        "acceptance_rate",
        "draft_k",
        "draft_overhead_ms",
        "latency_ms",
        "repetition",
        "speedup_vs_baseline",
        "temperature",
        "total_latency_ms",
        "verifier_calls_per_output_token",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _ci95(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) < 2:
        return float("nan")
    return float(1.96 * values.std(ddof=1) / (len(values) ** 0.5))


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    spec = df[df["mode"] == "speculative"].copy()
    return (
        spec.groupby(group_cols, dropna=False)
        .agg(
            rows=("speedup_vs_baseline", "count"),
            repetitions=("repetition", "nunique"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            std_speedup=("speedup_vs_baseline", "std"),
            ci95_speedup=("speedup_vs_baseline", _ci95),
            min_speedup=("speedup_vs_baseline", "min"),
            max_speedup=("speedup_vs_baseline", "max"),
            slowdown_rate=("speedup_vs_baseline", lambda values: float((values < 1.0).mean())),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_draft_overhead_ms=("draft_overhead_ms", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize speedup variance across repeated DraftVerifyBench runs."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    labels = args.labels or [Path(path).stem for path in args.inputs]
    if len(labels) != len(args.inputs):
        raise SystemExit("--labels must have the same length as --inputs")

    df = pd.concat([_read(path, label) for path, label in zip(args.inputs, labels, strict=True)])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _summary(df, ["run"]).to_csv(out / "gpu_variance_summary.csv", index=False)
    _summary(df, ["run", "prompt_type"]).to_csv(
        out / "gpu_variance_by_prompt_type.csv", index=False
    )
    _summary(df, ["run", "draft_k"]).to_csv(out / "gpu_variance_by_draft_k.csv", index=False)
    _summary(df, ["run", "schedule_type"]).to_csv(
        out / "gpu_variance_by_schedule.csv", index=False
    )
    _summary(df, ["run", "prompt_id", "temperature", "draft_k", "schedule_type"]).to_csv(
        out / "gpu_variance_by_case.csv", index=False
    )
    print(f"Wrote variance summaries to {out}")


if __name__ == "__main__":
    main()
