# Next GPU Experiments

This document separates work that is complete without a GPU from work that requires the A100
instance again.

## Completed Without GPU

The existing A100 Qwen/Llama reduced result CSVs were analyzed locally. No new GPU access was
required.

Generated artifacts:

- `docs/A100_Qwen_Llama_Validation_Report.md`
- `docs/GPU_Tradeoff_Decomposition.md`
- `results/gpu_model_pair_comparison.csv`
- `results/gpu_prompt_type_comparison.csv`
- `results/gpu_draft_k_comparison.csv`
- `results/gpu_schedule_comparison.csv`
- `results/gpu_correlation_summary.csv`
- `results/gpu_llama_fast_cases.csv`
- `results/gpu_llama_slow_cases.csv`
- `results/gpu_tradeoff_decomposition.csv`
- `results/gpu_tradeoff_summary.csv`
- `results/gpu_tradeoff_by_policy.csv`
- `results/gpu_closest_to_break_even_cases.csv`
- `results/gpu_oracle_best_cases.csv`

Scripts added:

- `scripts/compare_gpu_runs.py`
- `scripts/analyze_gpu_tradeoffs.py`
- `scripts/analyze_variance.py`
- `scripts/profile_gpu_decode.py`
- `scripts/run_hf_assisted_comparison.py`
- `scripts/run_vllm_comparison.py`

Configs added:

- `configs/gpu_qwen_full_static.yaml`
- `configs/gpu_llama_full_static.yaml`
- `configs/gpu_qwen_reduced_repeats.yaml`
- `configs/gpu_llama_reduced_repeats.yaml`

Public-facing draft:

- `docs/Public_Writeup_Draft.md`

## Why The Next GPU Session Matters

The current reduced A100 result is strong because it shows model-pair contrast:

- Qwen2.5 `0.5B -> 7B` never reached break-even.
- Llama `1B -> 8B` produced a `5.3355x` best case, but still slowed down in most rows.

The next experiments should answer whether the observed behavior is due to:

- algorithmic tradeoffs,
- the custom Hugging Face eager implementation,
- draft-model overhead,
- workload shape,
- or production-serving backend behavior.

## 1. Re-run Validation Checks

```bash
cd ~/DraftVerifyBench
source .venv/bin/activate
pytest
ruff check .
```

## 2. Regenerate Local Analysis

```bash
python scripts/compare_gpu_runs.py
python scripts/analyze_gpu_tradeoffs.py
```

## 3. Torch Profiler Traces

Profile baseline and speculative paths. These traces are CUDA claims, so run them on the GPU
instance.

```bash
python scripts/profile_gpu_decode.py \
  --mode baseline \
  --verifier-model meta-llama/Llama-3.1-8B \
  --device cuda \
  --max-new-tokens 64 \
  --trace-name llama_baseline \
  --require-cuda

python scripts/profile_gpu_decode.py \
  --mode speculative \
  --draft-model meta-llama/Llama-3.2-1B \
  --verifier-model meta-llama/Llama-3.1-8B \
  --device cuda \
  --draft-k 2 \
  --max-new-tokens 64 \
  --trace-name llama_spec_k2 \
  --require-cuda

python scripts/profile_gpu_decode.py \
  --mode speculative \
  --draft-model meta-llama/Llama-3.2-1B \
  --verifier-model meta-llama/Llama-3.1-8B \
  --device cuda \
  --draft-k 4 \
  --max-new-tokens 64 \
  --trace-name llama_spec_k4 \
  --require-cuda
```

Outputs:

- `results/profiles/*_trace.json`
- `results/profiles/*_profile.txt`
- `results/profiles/*_result.txt`

## 4. Qwen LM-Head Overhead Anomaly Test

The full GH200 grid showed Qwen had much higher draft overhead and stable slowdown. A plausible
mechanism is LM-head/vocabulary projection cost: Qwen's draft/verifier pair has tokenizer/vocab
compatibility friction in vLLM, and its vocabulary sizes differ from Llama's. Run this focused
profile to separate transformer-body time from LM-head time.

```bash
python scripts/profile_lm_head_overhead.py \
  --models \
    Qwen/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen2.5-7B-Instruct \
    meta-llama/Llama-3.2-1B \
    meta-llama/Llama-3.1-8B \
  --device cuda \
  --decode-steps 64 \
  --out results/lm_head_overhead_profile.csv \
  --raw-out results/lm_head_overhead_profile.json
```

Inspect:

```bash
cat results/lm_head_overhead_profile.csv
```

If Qwen shows a much larger `decode_lm_head_share`, that supports the hypothesis that vocab/LM-head
projection cost contributes to the Qwen overhead anomaly. If not, the slowdown is likely elsewhere
in the decode loop/backend path.

## 5. Repeat Reduced Runs For Variance

Run this before making the `5.3355x` Llama peak a headline claim in public writing. The goal is to
report mean/std/CI rather than one lucky maximum.

```bash
python scripts/run_gpu_validation.py \
  --config configs/gpu_llama_reduced_repeats.yaml \
  --out results/gpu_llama_reduced_repeats_results.csv \
  --raw-out results/gpu_llama_reduced_repeats_raw.jsonl \
  --metadata-out results/gpu_llama_reduced_repeats_metadata.json \
  --summary-out results/gpu_llama_reduced_repeats_summary.md \
  --require-cuda
```

```bash
python scripts/run_gpu_validation.py \
  --config configs/gpu_qwen_reduced_repeats.yaml \
  --out results/gpu_qwen_reduced_repeats_results.csv \
  --raw-out results/gpu_qwen_reduced_repeats_raw.jsonl \
  --metadata-out results/gpu_qwen_reduced_repeats_metadata.json \
  --summary-out results/gpu_qwen_reduced_repeats_summary.md \
  --require-cuda
```

Then summarize variance:

```bash
python scripts/analyze_variance.py \
  --inputs results/gpu_qwen_reduced_repeats_results.csv results/gpu_llama_reduced_repeats_results.csv \
  --labels qwen_repeats llama_repeats
```

## 6. Hugging Face Assisted Generation Comparison

This checks whether the official Transformers assisted-generation path behaves differently from the
custom transparent decoder.

```bash
python scripts/run_hf_assisted_comparison.py \
  --draft-model meta-llama/Llama-3.2-1B \
  --verifier-model meta-llama/Llama-3.1-8B \
  --device cuda \
  --prompt-types structured_json code_completion open_ended \
  --max-prompts 9 \
  --max-new-tokens 64 \
  --temperatures 0.0 \
  --out results/hf_assisted_llama_comparison.csv \
  --raw-out results/hf_assisted_llama_comparison_raw.jsonl \
  --metadata-out results/hf_assisted_llama_comparison_metadata.json
```

Repeat for Qwen:

```bash
python scripts/run_hf_assisted_comparison.py \
  --draft-model Qwen/Qwen2.5-0.5B-Instruct \
  --verifier-model Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --prompt-types structured_json code_completion open_ended \
  --max-prompts 9 \
  --max-new-tokens 64 \
  --temperatures 0.0 \
  --out results/hf_assisted_qwen_comparison.csv \
  --raw-out results/hf_assisted_qwen_comparison_raw.jsonl \
  --metadata-out results/hf_assisted_qwen_comparison_metadata.json
```

## 7. vLLM Comparison

vLLM's current docs use `--speculative-config` / `speculative_config` with a JSON-style object for
draft-model speculation. The script here uses the offline Python `LLM` API and passes:

```json
{
  "method": "draft_model",
  "model": "<draft-model>",
  "num_speculative_tokens": 4
}
```

Install vLLM only on the CUDA Linux instance.

```bash
pip install vllm
```

Run Llama:

```bash
python scripts/run_vllm_comparison.py \
  --draft-model meta-llama/Llama-3.2-1B \
  --verifier-model meta-llama/Llama-3.1-8B \
  --prompt-types structured_json code_completion open_ended \
  --max-prompts 9 \
  --max-new-tokens 64 \
  --temperature 0.0 \
  --num-speculative-tokens 4 \
  --out results/vllm_llama_comparison.csv \
  --raw-out results/vllm_llama_comparison_raw.jsonl \
  --metadata-out results/vllm_llama_comparison_metadata.json
```

Run Qwen:

```bash
python scripts/run_vllm_comparison.py \
  --draft-model Qwen/Qwen2.5-0.5B-Instruct \
  --verifier-model Qwen/Qwen2.5-7B-Instruct \
  --prompt-types structured_json code_completion open_ended \
  --max-prompts 9 \
  --max-new-tokens 64 \
  --temperature 0.0 \
  --num-speculative-tokens 4 \
  --out results/vllm_qwen_comparison.csv \
  --raw-out results/vllm_qwen_comparison_raw.jsonl \
  --metadata-out results/vllm_qwen_comparison_metadata.json
```

## 8. Full Static Grid

Only run the full static grid after profiler and backend comparisons, unless GPU time is cheap.

```bash
python scripts/run_gpu_validation.py \
  --config configs/gpu_llama_full_static.yaml \
  --out results/gpu_llama_full_static_results.csv \
  --raw-out results/gpu_llama_full_static_raw.jsonl \
  --metadata-out results/gpu_llama_full_static_metadata.json \
  --summary-out results/gpu_llama_full_static_summary.md \
  --require-cuda
```

```bash
python scripts/run_gpu_validation.py \
  --config configs/gpu_qwen_full_static.yaml \
  --out results/gpu_qwen_full_static_results.csv \
  --raw-out results/gpu_qwen_full_static_raw.jsonl \
  --metadata-out results/gpu_qwen_full_static_metadata.json \
  --summary-out results/gpu_qwen_full_static_summary.md \
  --require-cuda
```

## Stop Criteria

Stop the instance after:

1. result CSVs are written,
2. metadata JSON exists,
3. raw JSONL exists,
4. profiler traces, if generated, are archived,
5. a tarball has been copied off the instance.

Suggested archive:

```bash
tar -czf draftverifybench_next_gpu_results.tar.gz \
  results/*llama* \
  results/*qwen* \
  results/profiles
```
