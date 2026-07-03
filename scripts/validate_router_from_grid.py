from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROUTER_FEATURES = ["model_family", "prompt_type"]
POST_GENERATION_FIELDS = [
    "acceptance_rate",
    "speedup_vs_baseline",
    "total_latency_ms",
    "latency_ms",
    "generated_text",
    "output_text",
    "verifier_calls",
    "verifier_calls_per_output_token",
]


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
            "Missing GH200 full-static result CSVs required for router validation:\n"
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
        "draft_k",
        "repetition",
        "speedup_vs_baseline",
        "temperature",
        "total_latency_ms",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _speedup_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def speculated_mean_speedup(values: pd.Series) -> float:
        subset = df.loc[values.index]
        speculated = subset[subset["router_used_speculation"]]
        return float(speculated["router_speedup"].mean()) if not speculated.empty else 1.0

    def speculated_slowdown_rate(values: pd.Series) -> float:
        subset = df.loc[values.index]
        speculated = subset[subset["router_used_speculation"]]
        return float((speculated["router_speedup"] < 1.0).mean()) if not speculated.empty else 0.0

    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("router_speedup", "count"),
            mean_speedup=("router_speedup", "mean"),
            median_speedup=("router_speedup", "median"),
            best_speedup=("router_speedup", "max"),
            worst_speedup=("router_speedup", "min"),
            slowdown_rate=("router_speedup", lambda values: float((values < 1.0).mean())),
            mean_latency_ms=("router_latency_ms", "mean"),
            speculative_share=("router_used_speculation", "mean"),
            speculated_mean_speedup=("router_speedup", speculated_mean_speedup),
            speculated_slowdown_rate=("router_speedup", speculated_slowdown_rate),
        )
        .reset_index()
    )


def _action_candidates() -> list[str]:
    return ["baseline", "k1", "k2", "k4", "k8"]


def _action_to_k(action: str) -> float | None:
    if action == "baseline":
        return None
    return float(action.removeprefix("k"))


def _choose_action(train: pd.DataFrame, *, model_family: str, prompt_type: str | None) -> str:
    candidates: list[dict[str, Any]] = []
    subset = train[train["model_family"] == model_family]
    if prompt_type is not None:
        subset = subset[subset["prompt_type"] == prompt_type]
    candidates.append({"action": "baseline", "mean_speedup": 1.0, "rows": len(subset)})
    spec = subset[subset["mode"] == "speculative"]
    for draft_k in [1.0, 2.0, 4.0, 8.0]:
        rows = spec[spec["draft_k"] == draft_k]
        if rows.empty:
            continue
        candidates.append(
            {
                "action": f"k{int(draft_k)}",
                "mean_speedup": float(rows["speedup_vs_baseline"].mean()),
                "rows": len(rows),
            }
        )
    table = pd.DataFrame(candidates)
    return str(table.sort_values(["mean_speedup", "rows"], ascending=False).iloc[0]["action"])


def _prepare_case_rows(validation: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "model_family",
        "seed_label",
        "prompt_id",
        "prompt_type",
        "temperature",
        "repetition",
    ]
    baseline = (
        validation[validation["mode"] == "baseline"][key_cols + ["total_latency_ms"]]
        .rename(columns={"total_latency_ms": "baseline_latency_ms"})
        .copy()
    )
    spec = validation[validation["mode"] == "speculative"][
        key_cols + ["draft_k", "speedup_vs_baseline", "total_latency_ms"]
    ].copy()
    case_rows = baseline.copy()
    for draft_k in [1.0, 2.0, 4.0, 8.0]:
        rows = spec[spec["draft_k"] == draft_k].rename(
            columns={
                "speedup_vs_baseline": f"k{int(draft_k)}_speedup",
                "total_latency_ms": f"k{int(draft_k)}_latency_ms",
            }
        )
        case_rows = case_rows.merge(
            rows.drop(columns=["draft_k"]),
            on=key_cols,
            how="left",
            validate="one_to_one",
        )
    return case_rows


def _evaluate_action(case_rows: pd.DataFrame, action: str, policy: str) -> pd.DataFrame:
    routed = case_rows.copy()
    routed["router_policy"] = policy
    routed["router_action"] = action
    if action == "baseline":
        routed["router_speedup"] = 1.0
        routed["router_latency_ms"] = routed["baseline_latency_ms"]
        routed["router_used_speculation"] = False
        return routed
    routed["router_speedup"] = routed[f"{action}_speedup"]
    routed["router_latency_ms"] = routed[f"{action}_latency_ms"]
    routed["router_used_speculation"] = True
    return routed


def _evaluate_oracle(case_rows: pd.DataFrame) -> pd.DataFrame:
    routed = case_rows.copy()
    action_cols = {
        "baseline": "baseline_latency_ms",
        "k1": "k1_latency_ms",
        "k2": "k2_latency_ms",
        "k4": "k4_latency_ms",
        "k8": "k8_latency_ms",
    }
    speed_cols = {
        "baseline": None,
        "k1": "k1_speedup",
        "k2": "k2_speedup",
        "k4": "k4_speedup",
        "k8": "k8_speedup",
    }
    actions = []
    speeds = []
    latencies = []
    used_speculation = []
    for _, row in routed.iterrows():
        candidates = {"baseline": 1.0}
        for action, speed_col in speed_cols.items():
            if speed_col is None:
                continue
            value = row.get(speed_col)
            if pd.notna(value):
                candidates[action] = float(value)
        action = max(candidates, key=candidates.get)
        actions.append(action)
        speeds.append(candidates[action])
        latencies.append(row[action_cols[action]])
        used_speculation.append(action != "baseline")
    routed["router_policy"] = "oracle_per_row"
    routed["router_action"] = actions
    routed["router_speedup"] = speeds
    routed["router_latency_ms"] = latencies
    routed["router_used_speculation"] = used_speculation
    return routed


def validate_router(
    runs: dict[str, str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate offline routers using held-out seeds.

    The trained router chooses actions from the opposite seed using only ROUTER_FEATURES. Held-out
    realized speedup, latency, acceptance rate, and generated text are not used to make validation
    decisions. The oracle policy below is intentionally separate and is reported as a hindsight
    ceiling, not a deployable router.
    """
    _require_run_files(runs)
    df = pd.concat([_read(path, label) for label, path in runs.items()], ignore_index=True)
    routed_frames: list[pd.DataFrame] = []
    decision_rows: list[dict[str, Any]] = []

    for model_family in sorted(df["model_family"].unique()):
        family = df[df["model_family"] == model_family]
        for validation_seed in sorted(family["seed_label"].unique()):
            train = family[family["seed_label"] != validation_seed]
            validation = family[family["seed_label"] == validation_seed]
            case_rows = _prepare_case_rows(validation)

            routed_frames.append(
                _evaluate_action(case_rows, "baseline", "never_speculate")
            )
            for action in ["k1", "k2", "k4", "k8"]:
                routed_frames.append(
                    _evaluate_action(case_rows, action, f"always_speculate_{action}")
                )
            routed_frames.append(_evaluate_oracle(case_rows))

            global_action = _choose_action(train, model_family=model_family, prompt_type=None)
            routed_frames.append(
                _evaluate_action(case_rows, global_action, "trained_global_router")
            )
            decision_rows.append(
                {
                    "model_family": model_family,
                    "validation_seed": validation_seed,
                    "router_policy": "trained_global_router",
                    "prompt_type": "*",
                    "chosen_action": global_action,
                }
            )

            regime_parts = []
            for prompt_type in sorted(case_rows["prompt_type"].unique()):
                action = _choose_action(train, model_family=model_family, prompt_type=prompt_type)
                part = _evaluate_action(
                    case_rows[case_rows["prompt_type"] == prompt_type],
                    action,
                    "trained_regime_router",
                )
                regime_parts.append(part)
                decision_rows.append(
                    {
                        "model_family": model_family,
                        "validation_seed": validation_seed,
                        "router_policy": "trained_regime_router",
                        "prompt_type": prompt_type,
                        "chosen_action": action,
                    }
                )
            routed_frames.append(pd.concat(regime_parts, ignore_index=True))

    routed = pd.concat(routed_frames, ignore_index=True)
    summary = _speedup_summary(routed, ["model_family", "router_policy"])
    by_prompt = _speedup_summary(routed, ["model_family", "router_policy", "prompt_type"])
    decisions = pd.DataFrame(decision_rows)
    _assert_router_sanity(summary, routed)
    return summary, by_prompt, decisions, routed


def _assert_router_sanity(summary: pd.DataFrame, routed: pd.DataFrame) -> None:
    leaked = set(ROUTER_FEATURES).intersection(POST_GENERATION_FIELDS)
    if leaked:
        raise AssertionError(f"Router features include post-generation fields: {sorted(leaked)}")

    for row in summary.itertuples(index=False):
        subset = routed[
            (routed["model_family"] == row.model_family)
            & (routed["router_policy"] == row.router_policy)
        ]
        if len(subset) != row.rows:
            raise AssertionError(
                f"{row.model_family}/{row.router_policy} row mismatch: "
                f"{len(subset)} != {row.rows}"
            )
        slowdown_rate = float((subset["router_speedup"] < 1.0).mean())
        speculative_share = float(subset["router_used_speculation"].mean())
        if abs(slowdown_rate - row.slowdown_rate) > 1e-12:
            raise AssertionError(
                f"{row.model_family}/{row.router_policy} slowdown mismatch: "
                f"{slowdown_rate} != {row.slowdown_rate}"
            )
        if abs(speculative_share - row.speculative_share) > 1e-12:
            raise AssertionError(
                f"{row.model_family}/{row.router_policy} speculation share mismatch: "
                f"{speculative_share} != {row.speculative_share}"
            )

        speculated = subset[subset["router_used_speculation"]]
        speculated_slowdown_rate = (
            float((speculated["router_speedup"] < 1.0).mean()) if not speculated.empty else 0.0
        )
        if abs(speculated_slowdown_rate - row.speculated_slowdown_rate) > 1e-12:
            raise AssertionError(
                f"{row.model_family}/{row.router_policy} speculated slowdown mismatch: "
                f"{speculated_slowdown_rate} != {row.speculated_slowdown_rate}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate trained routers against always/never speculate using existing grids."
    )
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, by_prompt, decisions, routed = validate_router(DEFAULT_RUNS)
    summary.to_csv(out / "gh200_router_policy_comparison.csv", index=False)
    by_prompt.to_csv(out / "gh200_router_by_prompt_type.csv", index=False)
    decisions.to_csv(out / "gh200_router_decisions.csv", index=False)
    routed.to_csv(out / "gh200_router_case_results.csv", index=False)
    print(f"Wrote router validation outputs to {out}")


if __name__ == "__main__":
    main()
