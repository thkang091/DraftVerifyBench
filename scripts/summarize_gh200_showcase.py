from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_RUNS = {
    "qwen_seed42": "results/gpu_qwen_full_static_results.csv",
    "qwen_seed43": "results/gpu_qwen_full_static_seed43_results.csv",
    "llama_seed42": "results/gpu_llama_full_static_results.csv",
    "llama_seed43": "results/gpu_llama_full_static_seed43_results.csv",
}


def _require_run_files(runs: dict[str, str | Path]) -> None:
    missing = [str(path) for path in runs.values() if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Missing GH200 full-static result CSVs required for showcase summaries:\n"
            f"{formatted}\n"
            "These files are committed reproducibility artifacts. If they are absent, recover "
            "them from gh200_results_core_no_profiles.tar.gz or rerun the GH200 full-static "
            "commands in README.md."
        )


def _read(path: str | Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_label"] = label
    df["model_family"] = "qwen" if label.startswith("qwen") else "llama"
    df["seed_label"] = label.rsplit("_", maxsplit=1)[-1]
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()
    for column in [
        "acceptance_rate",
        "draft_k",
        "draft_overhead_ms",
        "repetition",
        "speedup_vs_baseline",
        "temperature",
        "verifier_calls_per_output_token",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _slowdown_rate(values: pd.Series) -> float:
    return float((values < 1.0).mean())


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    spec = df[df["mode"] == "speculative"].copy()
    return (
        spec.groupby(group_cols, dropna=False)
        .agg(
            rows=("speedup_vs_baseline", "count"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            median_speedup=("speedup_vs_baseline", "median"),
            std_speedup=("speedup_vs_baseline", "std"),
            best_speedup=("speedup_vs_baseline", "max"),
            worst_speedup=("speedup_vs_baseline", "min"),
            slowdown_rate=("speedup_vs_baseline", _slowdown_rate),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_draft_overhead_ms=("draft_overhead_ms", "mean"),
            verifier_calls_per_output_token=("verifier_calls_per_output_token", "mean"),
        )
        .reset_index()
    )


def _hf_summary(paths: list[str | Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        assisted = df[df["mode"] == "hf_assisted"].copy()
        label = "qwen" if "qwen" in str(path) else "llama"
        speed = assisted["speedup_vs_hf_generate"]
        rows.append(
            {
                "model_family": label,
                "rows": int(len(df)),
                "assisted_rows": int(len(assisted)),
                "mean_speedup_vs_hf_generate": float(speed.mean()),
                "median_speedup_vs_hf_generate": float(speed.median()),
                "best_speedup_vs_hf_generate": float(speed.max()),
                "slowdown_rate": float((speed < 1.0).mean()),
                "exact_match_rate": float(
                    assisted["exact_output_match_with_hf_generate"].astype(bool).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GH200 showcase summary tables.")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _require_run_files(DEFAULT_RUNS)
    df = pd.concat([_read(path, label) for label, path in DEFAULT_RUNS.items()])
    _summary(df, ["model_family"]).to_csv(out / "gh200_showcase_by_model_family.csv", index=False)
    _summary(df, ["model_family", "run_label"]).to_csv(
        out / "gh200_showcase_by_seed.csv", index=False
    )
    _summary(df, ["model_family", "draft_k"]).to_csv(
        out / "gh200_showcase_by_draft_k.csv", index=False
    )
    _summary(df, ["model_family", "prompt_type"]).to_csv(
        out / "gh200_showcase_by_prompt_type.csv", index=False
    )
    _summary(df, ["model_family", "temperature"]).to_csv(
        out / "gh200_showcase_by_temperature.csv", index=False
    )

    _hf_summary(
        [
            "results/hf_assisted_llama_comparison.csv",
            "results/hf_assisted_qwen_comparison.csv",
        ]
    ).to_csv(out / "gh200_hf_assisted_summary.csv", index=False)

    if Path("results/vllm_llama_comparison.csv").exists():
        pd.read_csv("results/vllm_llama_comparison.csv").to_csv(
            out / "gh200_vllm_llama_summary.csv", index=False
        )

    print(f"Wrote GH200 showcase summaries to {out}")


if __name__ == "__main__":
    main()
