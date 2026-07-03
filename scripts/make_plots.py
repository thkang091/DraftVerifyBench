from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.analysis import load_results


def _save_bar(series, title: str, ylabel: str, out: Path) -> None:
    ax = series.plot(kind="bar", title=title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _save_scatter(df, x: str, y: str, title: str, out: Path) -> None:
    ax = df.plot(kind="scatter", x=x, y=y, title=title, alpha=0.75)
    ax.axhline(1.0, color="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DraftVerifyBench plots.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.inputs)
    spec = df[df["mode"] == "speculative"].copy()
    if spec.empty:
        raise SystemExit("No speculative rows available for plotting.")

    _save_bar(
        spec.groupby("prompt_type")["speedup_vs_baseline"].mean(),
        "Mean Speedup by Prompt Type",
        "Speedup vs baseline",
        out_dir / "speedup_by_prompt_type.png",
    )
    _save_bar(
        spec.groupby("temperature")["speedup_vs_baseline"].mean(),
        "Mean Speedup by Temperature",
        "Speedup vs baseline",
        out_dir / "speedup_by_temperature.png",
    )
    _save_bar(
        spec.groupby("draft_k")["speedup_vs_baseline"].mean(),
        "Mean Speedup by draft_k",
        "Speedup vs baseline",
        out_dir / "speedup_by_draft_k.png",
    )
    _save_bar(
        spec.groupby("prompt_type")["acceptance_rate"].mean(),
        "Acceptance Rate by Prompt Type",
        "Acceptance rate",
        out_dir / "acceptance_rate_by_prompt_type.png",
    )
    _save_bar(
        spec.groupby("temperature")["acceptance_rate"].mean(),
        "Acceptance Rate by Temperature",
        "Acceptance rate",
        out_dir / "acceptance_rate_by_temperature.png",
    )
    _save_scatter(
        spec,
        "acceptance_rate",
        "speedup_vs_baseline",
        "Acceptance Rate vs Speedup",
        out_dir / "acceptance_vs_speedup.png",
    )
    condition = spec.assign(
        condition=(
            spec["prompt_type"].astype(str)
            + " | temp="
            + spec["temperature"].astype(str)
            + " | k="
            + spec["draft_k"].astype(str)
        )
    )
    slowdown = condition.groupby("condition")["slowdown_flag"].mean().sort_values(ascending=False)
    _save_bar(
        slowdown.head(25),
        "Slowdown Rate by Condition",
        "Slowdown rate",
        out_dir / "slowdown_rate_by_condition.png",
    )


if __name__ == "__main__":
    main()
