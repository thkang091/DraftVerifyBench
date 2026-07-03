from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from tqdm import tqdm

from draftverifybench.adaptive import AdaptivePolicyConfig, adaptive_speculative_decode
from draftverifybench.config import BenchmarkConfig, load_config
from draftverifybench.datasets import get_prompts
from draftverifybench.decoding import baseline_decode
from draftverifybench.metrics import result_to_metrics
from draftverifybench.models import load_model_bundle
from draftverifybench.speculative import speculative_decode
from draftverifybench.utils import (
    append_jsonl,
    cuda_hardware_metadata,
    cuda_memory_snapshot,
    ensure_parent,
    model_dump_compat,
)


def metadata(
    config: BenchmarkConfig,
    draft_bundle: Any,
    verifier_bundle: Any,
    *,
    prompt_count: int | None = None,
) -> dict[str, Any]:
    return {
        "device_type": str(verifier_bundle.device),
        **cuda_hardware_metadata(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "draft_model": config.draft_model,
        "verifier_model": config.verifier_model,
        "draft_parameter_count": draft_bundle.parameter_count,
        "verifier_parameter_count": verifier_bundle.parameter_count,
        "draft_dtype": str(draft_bundle.dtype),
        "verifier_dtype": str(verifier_bundle.dtype),
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
        "repetition_count": config.repetitions,
        "batch_size": config.batch_size,
        "prompt_count": prompt_count,
        "warmup_count": config.warmup_runs,
        "temperatures": config.temperatures,
        "draft_ks": config.draft_ks,
        "schedules": config.schedules,
        "adaptive_policies": config.adaptive_policies,
        "torch_compile": config.torch_compile,
        "validation_level": config.validation_level,
        "timing_scope": "cuda_validation"
        if str(verifier_bundle.device).startswith("cuda")
        else "local_dev_only",
        **cuda_memory_snapshot(verifier_bundle.device),
    }


def _write_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _runtime_row_labels(
    config: BenchmarkConfig,
    verifier_bundle: Any,
    *,
    latency_ms: float,
) -> dict[str, Any]:
    memory = cuda_memory_snapshot(verifier_bundle.device)
    allocated = memory["gpu_memory_allocated_bytes"]
    reserved = memory["gpu_memory_reserved_bytes"]
    return {
        "validation_level": config.validation_level,
        "device_type": str(verifier_bundle.device),
        "gpu_name": cuda_hardware_metadata().get("gpu_name"),
        "draft_model": config.draft_model,
        "verifier_model": config.verifier_model,
        "dtype": str(verifier_bundle.dtype),
        "latency_ms": latency_ms,
        **memory,
        "gpu_memory_allocated_mb": allocated / 1e6 if allocated is not None else None,
        "gpu_memory_reserved_mb": reserved / 1e6 if reserved is not None else None,
    }


def _adaptive_policy_runs(config: BenchmarkConfig) -> list[tuple[str, str, AdaptivePolicyConfig]]:
    base_config = AdaptivePolicyConfig(
        confidence_thresholds={float(k): v for k, v in config.confidence_thresholds.items()},
        entropy_thresholds={float(k): v for k, v in config.entropy_thresholds.items()},
        rolling_acceptance_window=config.rolling_acceptance_window,
        rolling_acceptance_min_k=config.rolling_acceptance_min_k,
        rolling_acceptance_max_k=config.rolling_acceptance_max_k,
    )
    runs = [(policy, policy, base_config) for policy in config.adaptive_policies]
    for variant in config.adaptive_variants:
        policy = str(variant.get("policy"))
        name = str(variant.get("name", policy))
        variant_config = AdaptivePolicyConfig(
            confidence_thresholds={
                float(k): v
                for k, v in variant.get(
                    "confidence_thresholds",
                    config.confidence_thresholds,
                ).items()
            },
            entropy_thresholds={
                float(k): v
                for k, v in variant.get("entropy_thresholds", config.entropy_thresholds).items()
            },
            rolling_acceptance_window=int(
                variant.get("rolling_acceptance_window", config.rolling_acceptance_window)
            ),
            rolling_acceptance_min_k=config.rolling_acceptance_min_k,
            rolling_acceptance_max_k=config.rolling_acceptance_max_k,
        )
        runs.append((name, policy, variant_config))
    deduped: list[tuple[str, str, AdaptivePolicyConfig]] = []
    seen: set[str] = set()
    for run in runs:
        if run[0] not in seen:
            deduped.append(run)
            seen.add(run[0])
    return deduped


def run_benchmark(
    config_path: str | Path,
    *,
    out: str | Path,
    raw_out: str | Path,
    metadata_out: str | Path,
    max_prompts: int | None = None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    draft_bundle = load_model_bundle(
        config.draft_model,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        local_files_only=config.local_files_only,
        torch_compile=config.torch_compile,
    )
    verifier_bundle = load_model_bundle(
        config.verifier_model,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        local_files_only=config.local_files_only,
        torch_compile=config.torch_compile,
    )

    Path(raw_out).unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    prompts = get_prompts(config.prompt_types, max_prompts=max_prompts)
    _write_metadata(
        metadata_out,
        metadata(config, draft_bundle, verifier_bundle, prompt_count=len(prompts)),
    )
    if str(verifier_bundle.device).startswith("cuda"):
        _write_metadata(
            "results/gpu_metadata.json",
            metadata(config, draft_bundle, verifier_bundle, prompt_count=len(prompts)),
        )
    model_pair = f"{config.draft_model}->{config.verifier_model}"
    adaptive_runs = _adaptive_policy_runs(config)

    warmup_prompts = prompts[:1]
    for warmup_index in range(config.warmup_runs):
        for prompt in warmup_prompts:
            baseline_decode(
                verifier_bundle,
                prompt.prompt_text,
                max_new_tokens=min(8, config.max_new_tokens),
                temperature=0.0,
                seed=config.seed + warmup_index,
            )
            speculative_decode(
                draft_bundle,
                verifier_bundle,
                prompt.prompt_text,
                max_new_tokens=min(8, config.max_new_tokens),
                draft_k=min(config.draft_ks or [1]),
                seed=config.seed + warmup_index,
            )

    work_items = [
        (prompt, temperature, repetition)
        for prompt in prompts
        for temperature in config.temperatures
        for repetition in range(config.repetitions)
    ]

    for prompt, temperature, repetition in tqdm(work_items, desc="baseline/speculative"):
        try:
            baseline = baseline_decode(
                verifier_bundle,
                prompt.prompt_text,
                max_new_tokens=config.max_new_tokens,
                temperature=temperature,
                seed=config.seed + repetition,
            )
            baseline_metrics = result_to_metrics(
                baseline,
                baseline_result=None,
                model_pair=model_pair,
                prompt_type=prompt.prompt_type,
                temperature=temperature,
                draft_k=None,
                repetition=repetition,
            )
            baseline_row = {
                **model_dump_compat(baseline_metrics),
                "mode": "baseline",
                "prompt_id": prompt.prompt_id,
                "expected_entropy_level": prompt.expected_entropy_level,
                **_runtime_row_labels(
                    config,
                    verifier_bundle,
                    latency_ms=baseline.total_latency_ms,
                ),
                "error": "",
            }
            rows.append(baseline_row)
            append_jsonl(
                raw_out,
                {
                    **baseline_row,
                    "output_text": baseline.output_text,
                    "output_token_ids": baseline.output_token_ids,
                },
            )

            if "static" in config.schedules:
                draft_ks = config.draft_ks
            else:
                draft_ks = []
            for draft_k in draft_ks:
                try:
                    speculative = speculative_decode(
                        draft_bundle,
                        verifier_bundle,
                        prompt.prompt_text,
                        max_new_tokens=config.max_new_tokens,
                        draft_k=draft_k,
                        temperature=temperature,
                        seed=config.seed + repetition,
                    )
                    speculative_metrics = result_to_metrics(
                        speculative,
                        baseline_result=baseline,
                        model_pair=model_pair,
                        prompt_type=prompt.prompt_type,
                        temperature=temperature,
                        draft_k=draft_k,
                        repetition=repetition,
                        schedule_type="static",
                    )
                    row = {
                        **model_dump_compat(speculative_metrics),
                        "mode": "speculative",
                        "prompt_id": prompt.prompt_id,
                        "expected_entropy_level": prompt.expected_entropy_level,
                        **_runtime_row_labels(
                            config,
                            verifier_bundle,
                            latency_ms=speculative.total_latency_ms,
                        ),
                        "error": "",
                    }
                    rows.append(row)
                    append_jsonl(
                        raw_out,
                        {
                            **row,
                            "output_text": speculative.output_text,
                            "output_token_ids": speculative.output_token_ids,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - benchmark should continue per row
                    rows.append(
                        {
                            "mode": "speculative",
                            "prompt_id": prompt.prompt_id,
                            "prompt_type": prompt.prompt_type,
                            "temperature": temperature,
                            "draft_k": draft_k,
                            "repetition": repetition,
                            "model_pair": model_pair,
                            "error": repr(exc),
                        }
                    )
            for adaptive_label, adaptive_policy, policy_config in adaptive_runs:
                try:
                    adaptive = adaptive_speculative_decode(
                        draft_bundle,
                        verifier_bundle,
                        prompt.prompt_text,
                        max_new_tokens=config.max_new_tokens,
                        policy=adaptive_policy,
                        policy_config=policy_config,
                        seed=config.seed + repetition,
                    )
                    adaptive_metrics = result_to_metrics(
                        adaptive,
                        baseline_result=baseline,
                        model_pair=model_pair,
                        prompt_type=prompt.prompt_type,
                        temperature=temperature,
                        draft_k=None,
                        repetition=repetition,
                        schedule_type="adaptive",
                        adaptive_policy=adaptive_label,
                    )
                    row = {
                        **model_dump_compat(adaptive_metrics),
                        "mode": "speculative",
                        "prompt_id": prompt.prompt_id,
                        "expected_entropy_level": prompt.expected_entropy_level,
                        "adaptive_base_policy": adaptive_policy,
                        "selected_k_per_step": adaptive.selected_k_per_step,
                        "confidence_per_step": adaptive.confidence_per_step,
                        "entropy_per_step": adaptive.entropy_per_step,
                        "recent_acceptance_per_step": adaptive.recent_acceptance_per_step,
                        "accepted_tokens_by_k": adaptive.accepted_tokens_by_k,
                        "rejected_tokens_by_k": adaptive.rejected_tokens_by_k,
                        **_runtime_row_labels(
                            config,
                            verifier_bundle,
                            latency_ms=adaptive.total_latency_ms,
                        ),
                        "error": "",
                    }
                    rows.append(row)
                    append_jsonl(
                        raw_out,
                        {
                            **row,
                            "output_text": adaptive.output_text,
                            "output_token_ids": adaptive.output_token_ids,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - benchmark should continue per row
                    rows.append(
                        {
                            "mode": "speculative",
                            "prompt_id": prompt.prompt_id,
                            "prompt_type": prompt.prompt_type,
                            "temperature": temperature,
                            "draft_k": None,
                            "schedule_type": "adaptive",
                            "adaptive_policy": adaptive_label,
                            "adaptive_base_policy": adaptive_policy,
                            "repetition": repetition,
                            "model_pair": model_pair,
                            "error": repr(exc),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - benchmark should continue per row
            rows.append(
                {
                    "mode": "baseline",
                    "prompt_id": prompt.prompt_id,
                    "prompt_type": prompt.prompt_type,
                    "temperature": temperature,
                    "draft_k": None,
                    "repetition": repetition,
                    "model_pair": model_pair,
                    "error": repr(exc),
                }
            )

    _write_rows(out, rows)
    final_metadata = metadata(config, draft_bundle, verifier_bundle, prompt_count=len(prompts))
    _write_metadata(metadata_out, final_metadata)
    if str(verifier_bundle.device).startswith("cuda"):
        _write_metadata("results/gpu_metadata.json", final_metadata)
    return rows
