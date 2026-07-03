# GitHub Showcase Checklist

Use this checklist when publishing DraftVerifyBench.

## Commit These

Core project:

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `draftverifybench/`
- `scripts/`
- `configs/`
- `tests/`
- `examples/`
- `docs/`

Important result summaries and small artifacts:

- `results/gpu_llama_full_static_summary.md`
- `results/gpu_llama_full_static_seed43_summary.md`
- `results/gpu_qwen_full_static_summary.md`
- `results/gpu_qwen_full_static_seed43_summary.md`
- `results/gpu_llama_full_static_metadata.json`
- `results/gpu_llama_full_static_seed43_metadata.json`
- `results/gpu_qwen_full_static_metadata.json`
- `results/gpu_qwen_full_static_seed43_metadata.json`
- `results/gpu_variance_summary.csv`
- `results/gpu_variance_by_draft_k.csv`
- `results/gpu_variance_by_prompt_type.csv`
- `results/gh200_showcase_by_model_family.csv`
- `results/gh200_showcase_by_draft_k.csv`
- `results/gh200_showcase_by_prompt_type.csv`
- `results/gh200_showcase_by_seed.csv`
- `results/gh200_hf_assisted_summary.csv`
- `results/gh200_vllm_llama_summary.csv`
- `results/gh200_router_policy_comparison.csv`
- `results/gh200_router_by_prompt_type.csv`
- `results/gh200_router_decisions.csv`
- `results/gh200_router_case_results.csv`
- `results/vllm_qwen_vocab_mismatch_error.log`

Because `.gitignore` ignores `results/*.csv`, use `git add -f` for selected small result tables.

## Do Not Commit These Directly

Large or noisy files:

- `gh200_profiler_traces.tar.gz`
- `gh200_results_core_no_profiles.tar.gz`
- `results/profiles/`
- full raw JSONL files unless needed for a release artifact
- model weights
- `.venv/`
- Hugging Face cache
- API tokens

Upload large profiler traces as GitHub Release artifacts instead.

## Suggested Commit Commands

```bash
git init
git add README.md pyproject.toml requirements.txt .gitignore
git add draftverifybench scripts configs tests examples docs
git add -f \
  results/gpu_llama_full_static_summary.md \
  results/gpu_llama_full_static_seed43_summary.md \
  results/gpu_qwen_full_static_summary.md \
  results/gpu_qwen_full_static_seed43_summary.md \
  results/gpu_llama_full_static_metadata.json \
  results/gpu_llama_full_static_seed43_metadata.json \
  results/gpu_qwen_full_static_metadata.json \
  results/gpu_qwen_full_static_seed43_metadata.json \
  results/gpu_variance_summary.csv \
  results/gpu_variance_by_draft_k.csv \
  results/gpu_variance_by_prompt_type.csv \
  results/gh200_showcase_by_model_family.csv \
  results/gh200_showcase_by_draft_k.csv \
  results/gh200_showcase_by_prompt_type.csv \
  results/gh200_showcase_by_seed.csv \
  results/gh200_hf_assisted_summary.csv \
  results/gh200_vllm_llama_summary.csv \
  results/gh200_router_policy_comparison.csv \
  results/gh200_router_by_prompt_type.csv \
  results/gh200_router_decisions.csv \
  results/gh200_router_case_results.csv \
  results/vllm_qwen_vocab_mismatch_error.log
git commit -m "Add DraftVerifyBench GH200 speculative decoding study"
```

## Pre-Publish Validation

Run the test suite and regenerate the router tables before publishing:

```bash
python scripts/validate_router_from_grid.py
pytest
```

Expected local validation at packaging time:

```text
41 passed
```

## Suggested GitHub Description

```text
Benchmarking when speculative decoding speeds up or slows down LLM inference. Includes GH200
Qwen/Llama full-grid results, variance summaries, HF assisted generation, vLLM comparison, and
profiler artifacts.
```

## README Lead

Lead with:

> On GH200, Llama 1B -> 8B achieved repeatable positive mean speedup across two seeds, while Qwen
> 0.5B -> 7B was stably slower. Even for Llama, most speculative rows still slowed down, showing
> that speculative decoding is conditional on model pair, workload, draft length, and backend
> overhead.

## Release Artifact Suggestion

Create a GitHub Release named:

```text
GH200 validation artifacts
```

Attach:

- `gh200_results_core_no_profiles.tar.gz`
- `gh200_profiler_traces.tar.gz`

Do this after the repo is public.
