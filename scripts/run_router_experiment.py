from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.adaptive import AdaptivePolicyConfig, adaptive_speculative_decode
from draftverifybench.config import load_config
from draftverifybench.datasets import get_prompts
from draftverifybench.decoding import baseline_decode
from draftverifybench.metrics import result_to_metrics
from draftverifybench.models import load_model_bundle
from draftverifybench.routing import draft_distribution_features, route_prompt, router_risk_score
from draftverifybench.runner import metadata
from draftverifybench.speculative import speculative_decode
from draftverifybench.utils import append_jsonl, ensure_parent, model_dump_compat


def _write_rows(path: str | Path, rows: list[dict]) -> None:
    ensure_parent(path)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: str | Path, payload: dict) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_router_experiment(
    config_path: str,
    *,
    out: str,
    raw_out: str,
    metadata_out: str,
    max_prompts: int | None = None,
) -> list[dict]:
    config = load_config(config_path)
    draft = load_model_bundle(
        config.draft_model,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        local_files_only=config.local_files_only,
    )
    verifier = load_model_bundle(
        config.verifier_model,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        local_files_only=config.local_files_only,
    )
    prompts = get_prompts(config.prompt_types, max_prompts=max_prompts)
    _write_metadata(metadata_out, metadata(config, draft, verifier, prompt_count=len(prompts)))
    Path(raw_out).unlink(missing_ok=True)
    rows: list[dict] = []
    policy_config = AdaptivePolicyConfig()
    model_pair = f"{config.draft_model}->{config.verifier_model}"
    router_policies = config.router_policies or [config.router_policy]
    for prompt in prompts:
        features = draft_distribution_features(draft, prompt.prompt_text, prompt.prompt_type)
        for router_policy in router_policies:
            decision = route_prompt(
                router_policy,
                features,
                entropy_threshold=config.router_entropy_threshold,
                top1_threshold=config.router_top1_threshold,
            )
            for temperature in config.temperatures:
                for repetition in range(config.repetitions):
                    baseline = baseline_decode(
                        verifier,
                        prompt.prompt_text,
                        max_new_tokens=config.max_new_tokens,
                        temperature=temperature,
                        seed=config.seed + repetition,
                    )
                    if decision == "baseline":
                        result = baseline
                        schedule_type = "baseline"
                        draft_k = None
                        adaptive_policy = None
                    elif decision == "static_speculative":
                        result = speculative_decode(
                            draft,
                            verifier,
                            prompt.prompt_text,
                            max_new_tokens=config.max_new_tokens,
                            draft_k=config.router_static_draft_k,
                            temperature=temperature,
                            seed=config.seed + repetition,
                        )
                        schedule_type = "static"
                        draft_k = config.router_static_draft_k
                        adaptive_policy = None
                    else:
                        result = adaptive_speculative_decode(
                            draft,
                            verifier,
                            prompt.prompt_text,
                            max_new_tokens=config.max_new_tokens,
                            policy=config.router_adaptive_policy,
                            policy_config=policy_config,
                            seed=config.seed + repetition,
                        )
                        schedule_type = "adaptive"
                        draft_k = None
                        adaptive_policy = config.router_adaptive_policy
                    metrics = result_to_metrics(
                        result,
                        baseline_result=baseline,
                        model_pair=model_pair,
                        prompt_type=prompt.prompt_type,
                        temperature=temperature,
                        draft_k=draft_k,
                        repetition=repetition,
                        schedule_type=schedule_type,
                        adaptive_policy=adaptive_policy,
                    )
                    row = {
                        **model_dump_compat(metrics),
                        "mode": "router",
                        "router_policy": router_policy,
                        "router_decision": decision,
                        "router_risk_score": router_risk_score(features),
                        "prompt_id": prompt.prompt_id,
                        "expected_entropy_level": prompt.expected_entropy_level,
                        **features.__dict__,
                        "error": "",
                    }
                    rows.append(row)
                    append_jsonl(raw_out, {**row, "output_text": result.output_text})
    _write_rows(out, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-router benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    run_router_experiment(
        args.config,
        out=args.out,
        raw_out=args.raw_out,
        metadata_out=args.metadata_out,
        max_prompts=args.max_prompts,
    )


if __name__ == "__main__":
    main()
