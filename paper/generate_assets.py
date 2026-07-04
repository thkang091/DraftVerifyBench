from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_router_from_grid import validate_router  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
TABLE_DIR = ROOT / "paper" / "tables"

RUNS = {
    "qwen_seed42": ROOT / "results/gpu_qwen_full_static_results.csv",
    "qwen_seed43": ROOT / "results/gpu_qwen_full_static_seed43_results.csv",
    "llama_seed42": ROOT / "results/gpu_llama_full_static_results.csv",
    "llama_seed43": ROOT / "results/gpu_llama_full_static_seed43_results.csv",
}


def _tracked(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _require_tracked(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    untracked = [
        str(path.relative_to(ROOT)) for path in paths if path.exists() and not _tracked(path)
    ]
    if missing or untracked:
        message = []
        if missing:
            message.append("Missing required paper data files:\n" + "\n".join(missing))
        if untracked:
            message.append("Required paper data files are not committed:\n" + "\n".join(untracked))
        raise SystemExit("\n\n".join(message))


def _read_runs() -> pd.DataFrame:
    _require_tracked(list(RUNS.values()))
    frames = []
    for label, path in RUNS.items():
        df = pd.read_csv(path)
        df["run_label"] = label
        df["model_family"] = "qwen" if label.startswith("qwen") else "llama"
        df["seed_label"] = label.rsplit("_", maxsplit=1)[-1]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()
    for column in [
        "acceptance_rate",
        "draft_k",
        "draft_overhead_ms",
        "repetition",
        "speedup_vs_baseline",
        "temperature",
        "total_latency_ms",
        "verifier_calls_per_output_token",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _slowdown(values: pd.Series) -> float:
    return float((values < 1.0).mean())


def _spec_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
            slowdown_rate=("speedup_vs_baseline", _slowdown),
            mean_acceptance=("acceptance_rate", "mean"),
            verifier_calls_per_token=("verifier_calls_per_output_token", "mean"),
            draft_overhead_ms=("draft_overhead_ms", "mean"),
        )
        .reset_index()
    )


def _pct(value: float) -> str:
    return f"{100 * value:.2f}\\%"


def _speed(value: float) -> str:
    return f"{value:.4f}$\\times$"


def _write_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = ["\\begin{tabular}{" + "l" * len(headers) + "}", "\\toprule"]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tables(df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    model = _spec_summary(df, ["model_family"]).sort_values("model_family")
    _write_table(
        TABLE_DIR / "model_pair_summary.tex",
        ["Pair", "Spec rows", "Mean", "Median", "Best", "Slowdown", "Acceptance"],
        [
            [
                (
                    "Llama 1B$\\rightarrow$8B"
                    if row.model_family == "llama"
                    else "Qwen 0.5B$\\rightarrow$7B"
                ),
                str(int(row.rows)),
                _speed(row.mean_speedup),
                _speed(row.median_speedup),
                _speed(row.best_speedup),
                _pct(row.slowdown_rate),
                f"{row.mean_acceptance:.3f}",
            ]
            for row in model.itertuples(index=False)
        ],
    )

    draft = _spec_summary(df, ["model_family", "draft_k"]).sort_values(
        ["model_family", "draft_k"]
    )
    _write_table(
        TABLE_DIR / "draft_k_summary.tex",
        ["Pair", "$k$", "Rows", "Mean", "Median", "Slowdown"],
        [
            [
                "Llama" if row.model_family == "llama" else "Qwen",
                str(int(row.draft_k)),
                str(int(row.rows)),
                _speed(row.mean_speedup),
                _speed(row.median_speedup),
                _pct(row.slowdown_rate),
            ]
            for row in draft.itertuples(index=False)
        ],
    )

    prompts = _spec_summary(df, ["model_family", "prompt_type"]).sort_values(
        ["model_family", "mean_speedup"], ascending=[True, False]
    )
    _write_table(
        TABLE_DIR / "prompt_type_summary.tex",
        ["Pair", "Prompt type", "Rows", "Mean", "Median", "Slowdown"],
        [
            [
                "Llama" if row.model_family == "llama" else "Qwen",
                str(row.prompt_type).replace("_", "\\_"),
                str(int(row.rows)),
                _speed(row.mean_speedup),
                _speed(row.median_speedup),
                _pct(row.slowdown_rate),
            ]
            for row in prompts.itertuples(index=False)
        ],
    )

    seeds = _spec_summary(df, ["model_family", "seed_label"]).sort_values(
        ["model_family", "seed_label"]
    )
    _write_table(
        TABLE_DIR / "seed_summary.tex",
        ["Pair", "Seed", "Rows", "Mean", "Median", "Best", "Slowdown"],
        [
            [
                "Llama" if row.model_family == "llama" else "Qwen",
                str(row.seed_label).replace("seed", ""),
                str(int(row.rows)),
                _speed(row.mean_speedup),
                _speed(row.median_speedup),
                _speed(row.best_speedup),
                _pct(row.slowdown_rate),
            ]
            for row in seeds.itertuples(index=False)
        ],
    )

    router_summary, _router_prompt, _decisions, _routed = validate_router(
        {label: str(path.relative_to(ROOT)) for label, path in RUNS.items()}
    )
    selected = router_summary[
        router_summary["router_policy"].isin(
            ["never_speculate", "always_speculate_k4", "trained_regime_router", "oracle_per_row"]
        )
    ].copy()
    policy_names = {
        "never_speculate": "Never",
        "always_speculate_k4": "Always $k=4$",
        "trained_regime_router": "Trained router",
        "oracle_per_row": "Oracle",
    }
    policy_order = {
        "never_speculate": 0,
        "always_speculate_k4": 1,
        "trained_regime_router": 2,
        "oracle_per_row": 3,
    }
    selected["policy_order"] = selected["router_policy"].map(policy_order)
    selected = selected.sort_values(["model_family", "policy_order"])
    _write_table(
        TABLE_DIR / "router_summary.tex",
        [
            "Pair",
            "Policy",
            "Rows",
            "Mean",
            "Slowdown",
            "Spec share",
            "Spec mean",
            "Spec slowdown",
        ],
        [
            [
                "Llama" if row.model_family == "llama" else "Qwen",
                policy_names[row.router_policy],
                str(int(row.rows)),
                _speed(row.mean_speedup),
                _pct(row.slowdown_rate),
                _pct(row.speculative_share),
                _speed(row.speculated_mean_speedup),
                _pct(row.speculated_slowdown_rate),
            ]
            for row in selected.itertuples(index=False)
        ],
    )

    hf_paths = [
        ROOT / "results/hf_assisted_llama_comparison.csv",
        ROOT / "results/hf_assisted_qwen_comparison.csv",
    ]
    _require_tracked(hf_paths + [ROOT / "results/vllm_llama_comparison.csv"])
    hf_rows = []
    for path in hf_paths:
        hdf = pd.read_csv(path)
        assisted = hdf[hdf["mode"] == "hf_assisted"]
        speed = assisted["speedup_vs_hf_generate"]
        hf_rows.append(
            [
                "Llama" if "llama" in path.name else "Qwen",
                str(len(assisted)),
                _speed(speed.mean()),
                _speed(speed.median()),
                _speed(speed.max()),
                _pct(float((speed < 1.0).mean())),
                _pct(float(assisted["exact_output_match_with_hf_generate"].astype(bool).mean())),
            ]
        )
    vdf = pd.read_csv(ROOT / "results/vllm_llama_comparison.csv")
    baseline = vdf[vdf["mode"] == "vllm_baseline"].iloc[0]
    spec = vdf[vdf["mode"] == "vllm_speculative"].iloc[0]
    hf_rows.append(
        [
            "vLLM Llama",
            str(int(spec.prompt_count)),
            _speed(float(spec.speedup_vs_vllm_baseline)),
            "--",
            "--",
            "100.00\\%" if spec.speedup_vs_vllm_baseline < 1.0 else "0.00\\%",
            f"{float(spec.tokens_per_second):.1f} tok/s vs. {float(baseline.tokens_per_second):.1f}",
        ]
    )
    _write_table(
        TABLE_DIR / "framework_summary.tex",
        [
            "Comparison",
            "Rows/prompts",
            "Mean speedup",
            "Median",
            "Best",
            "Slowdown",
            "Exact match / throughput",
        ],
        hf_rows,
    )


def _plot_bar(ax, labels: list[str], values: list[float], title: str, ylabel: str = "Speedup"):
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(labels)])
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)


def _write_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    draft = _spec_summary(df, ["model_family", "draft_k"]).sort_values(
        ["model_family", "draft_k"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, family in zip(axes, ["llama", "qwen"], strict=True):
        sub = draft[draft["model_family"] == family]
        x = range(len(sub))
        width = 0.38
        ax.bar([i - width / 2 for i in x], sub["mean_speedup"], width, label="Mean")
        ax.bar([i + width / 2 for i in x], sub["median_speedup"], width, label="Median")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(list(x), [str(int(k)) for k in sub["draft_k"]])
        ax.set_title("Llama" if family == "llama" else "Qwen")
        ax.set_xlabel("Draft length $k$")
        ax.legend()
    axes[0].set_ylabel("Speedup vs. baseline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mean_median_by_draft_k.pdf")
    fig.savefig(FIG_DIR / "mean_median_by_draft_k.png", dpi=200)
    plt.close(fig)

    prompts = _spec_summary(df, ["model_family", "prompt_type"]).sort_values(
        ["model_family", "mean_speedup"], ascending=[True, False]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, family in zip(axes, ["llama", "qwen"], strict=True):
        sub = prompts[prompts["model_family"] == family]
        labels = [str(v).replace("_", "\n") for v in sub["prompt_type"]]
        _plot_bar(ax, labels, list(sub["mean_speedup"]), "Llama" if family == "llama" else "Qwen")
    axes[0].set_ylabel("Mean speedup vs. baseline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "speedup_by_prompt_type.pdf")
    fig.savefig(FIG_DIR / "speedup_by_prompt_type.png", dpi=200)
    plt.close(fig)

    router_summary, _router_prompt, _decisions, _routed = validate_router(
        {label: str(path.relative_to(ROOT)) for label, path in RUNS.items()}
    )
    router = router_summary[
        (router_summary["model_family"] == "llama")
        & (
            router_summary["router_policy"].isin(
                ["never_speculate", "always_speculate_k4", "trained_regime_router", "oracle_per_row"]
            )
        )
    ].copy()
    order = ["never_speculate", "always_speculate_k4", "trained_regime_router", "oracle_per_row"]
    router["order"] = router["router_policy"].map({name: idx for idx, name in enumerate(order)})
    router = router.sort_values("order")
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_bar(
        ax,
        ["Never", "Always\n$k=4$", "Trained\nrouter", "Oracle"],
        list(router["mean_speedup"]),
        "Llama router policies",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "router_policy_comparison.pdf")
    fig.savefig(FIG_DIR / "router_policy_comparison.png", dpi=200)
    plt.close(fig)

    seeds = _spec_summary(df, ["model_family", "seed_label"]).sort_values(
        ["model_family", "seed_label"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [
        ("Llama" if row.model_family == "llama" else "Qwen") + "\n" + row.seed_label
        for row in seeds.itertuples(index=False)
    ]
    _plot_bar(ax, labels, list(seeds["mean_speedup"]), "Seed stability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seed_stability.pdf")
    fig.savefig(FIG_DIR / "seed_stability.png", dpi=200)
    plt.close(fig)


def main() -> None:
    df = _read_runs()
    _write_tables(df)
    _write_figures(df)
    print(f"Wrote paper tables to {TABLE_DIR}")
    print(f"Wrote paper figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
