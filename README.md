# DraftVerifyBench

## Profiling when speculative decoding speeds up or slows down local LLM inference

DraftVerifyBench is a local benchmarking framework for comparing standard autoregressive decoding
against speculative decoding across model pairs, prompt types, temperatures, and draft lengths.

**GPU validation caveat:** Local Mac/CPU/MPS runs are useful for correctness and debugging.
Headline inference-systems claims should be based on CUDA GPU validation runs with larger model
pairs such as 1B-class draft models and 7B/8B-class verifier models.

## Primary Finding

### GH200 Full-Static Validation

DraftVerifyBench includes full-static CUDA validation on an `NVIDIA GH200 480GB` with two modern
draft/verifier pairs, two seeds per model family, and `1,800` total benchmark rows:

| model pair | speculative rows | mean speedup | median speedup | best speedup | slowdown rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct -> Qwen2.5-7B-Instruct` | 720 | `0.5936x` | `0.5826x` | `1.5739x` | `99.17%` |
| `Llama-3.2-1B -> Llama-3.1-8B` | 720 | `1.0667x` | `0.8689x` | `12.8418x` | `67.36%` |

The GH200 result is a regime story: Qwen was stably slower across two seeds, while Llama crossed
break-even in specific regimes, especially larger draft lengths and structured/code-like workloads.
Even for Llama, the median row was slower and `67.36%` of speculative rows slowed down, so the
result is conditional rather than a blanket acceleration claim.

A seed-held-out regime router then reduced Llama slowdown risk versus always speculating and learned
to disable speculation for Qwen. This is an expected-value router, not a per-row win classifier:

| model pair | policy | mean speedup | slowdown rate | speculative share | speculated slowdown |
| --- | --- | ---: | ---: | ---: | ---: |
| `Llama-3.2-1B -> Llama-3.1-8B` | always speculate `k=4` | `1.2068x` | `45.00%` | `100.00%` | `45.00%` |
| `Llama-3.2-1B -> Llama-3.1-8B` | trained regime router | `1.2294x` | `37.22%` | `70.00%` | `53.17%` |
| `Llama-3.2-1B -> Llama-3.1-8B` | hindsight oracle | `1.4363x` | `0.00%` | `55.56%` | `0.00%` |
| `Qwen2.5-0.5B -> Qwen2.5-7B` | always speculate `k=4` | `0.6520x` | `98.89%` | `100.00%` | `98.89%` |
| `Qwen2.5-0.5B -> Qwen2.5-7B` | trained regime router | `1.0000x` | `0.00%` | `0.00%` | `0.00%` |

For Llama, the router speculated on `70%` of traffic and that selected speculative traffic averaged
`1.3277x`, even though `53.17%` of selected speculative rows still underperformed baseline. The
wins were large enough to outweigh the more frequent row-level losses. Compared with the hindsight
oracle, the router captured `52.58%` of the available gain above baseline while using only
pre-generation route features: model family and oracle-labeled prompt type.

Each router policy row is `N=180` validation cases per model family: two seed-held-out folds with
`90` validation cases each. A case is one prompt/temperature/repetition condition with baseline and
candidate `k=1`, `k=2`, `k=4`, and `k=8` speculative outcomes.

See [`docs/GH200_Final_Showcase_Report.md`](docs/GH200_Final_Showcase_Report.md).
Router validation is in [`docs/Router_Validation_Report.md`](docs/Router_Validation_Report.md).

The earlier reduced A100 validation is in
[`docs/A100_Qwen_Llama_Validation_Report.md`](docs/A100_Qwen_Llama_Validation_Report.md).
The tradeoff decomposition is in
[`docs/GPU_Tradeoff_Decomposition.md`](docs/GPU_Tradeoff_Decomposition.md), and the next GPU
experiment plan is in [`docs/Next_GPU_Experiments.md`](docs/Next_GPU_Experiments.md).
A public writeup draft is in [`docs/Public_Writeup_Draft.md`](docs/Public_Writeup_Draft.md).

### Local GPT-2/MPS Validation

On Apple MPS with `distilgpt2 -> gpt2`, speculative decoding produced useful speedups only in
narrow conditions where verifier-call reduction outweighed draft-model overhead. Across 480
speculative rows, mean speedup was `0.7875x` and `82.92%` of speculative runs slowed down. The
best case, `factual_qa` prompt `qa_002` at `temperature=0.0` and `draft_k=2`, reached `3.0514x`
speedup with `0.5862` acceptance. The worst case, `structured_json` prompt `json_002` at
`temperature=1.0` and `draft_k=1`, slowed to `0.1059x` despite `0.8125` acceptance.

The key systems result is that acceptance rate alone was not enough: acceptance-rate vs speedup
correlation was only `0.1352`. Draft overhead, verifier-call reduction, prompt type, temperature,
and draft length all affected whether speculative decoding helped.

## Why This Matters

Speculative decoding is an important LLM inference acceleration technique: a smaller draft model
proposes tokens, and a larger verifier model accepts or rejects them. Accepted draft tokens can
reduce expensive verifier calls. Rejected tokens waste draft compute.

The speedup is conditional, not guaranteed. Serving systems need profiling gates that measure
acceptance rate, verifier-call reduction, draft overhead, prompt class, and generation settings
before enabling speculative decoding broadly.

## What I Built

- Local Hugging Face model loader with CUDA, Apple MPS, and CPU device selection
- Standard autoregressive baseline decoder
- Greedy speculative decoder with draft-token acceptance tracking
- Built-in prompt suite across code, JSON extraction, factual QA, summarization, and open-ended text
- YAML-driven benchmark runner
- Latency, token throughput, time-to-first-token, verifier-call, and draft-call metrics
- Acceptance-rate, rejection, slowdown, and draft-overhead logging
- CSV metrics, JSONL raw traces, and hardware metadata
- Correlation analysis and slowdown-case analysis
- A100 Qwen/Llama comparison tables and tradeoff decomposition
- GH200 full-static Qwen/Llama validation with two seeds
- Seed-held-out router validation against always-speculate, never-speculate, and oracle policies
- Hugging Face assisted-generation and vLLM comparisons
- Prepared LM-head overhead profiler for Qwen anomaly analysis
- Greedy output-equivalence check
- Matplotlib plots
- Prepared repeated-run configs and variance analysis for confidence intervals
- Technical report and application package
- Unit tests for decoding, metrics, runner, analysis, plots, and equivalence checks

## Architecture

```text
Prompt Suite
    |
    v
Baseline Decoder
    |
    v
Speculative Decoder
    |
    v
Draft Model + Verifier Model
    |
    v
Latency / Tokens / Acceptance Logs
    |
    v
CSV + JSONL Results
    |
    v
Analysis Tables + Plots + Report
```

## Results

Run setup:

- Device: Apple MPS
- Draft model: `distilgpt2` (`81,912,576` parameters)
- Verifier model: `gpt2` (`124,439,808` parameters)
- Rows: 600 total, 120 baseline, 480 speculative
- Prompt suite: 15 built-in prompts
- Temperatures: `0.0`, `0.3`, `0.7`, `1.0`
- Draft lengths: `1`, `2`, `4`, `8`
- Repetitions: 2
- Max new tokens: 48

The original `local_small` grid was reduced from 64 to 48 max new tokens and from 3 to 2
repetitions after a one-prompt probe showed the full grid was too slow on this local MPS machine.
The medium run (`gpt2 -> gpt2-medium`) was skipped for the same reason. No medium results are
claimed.

These GPT-2-family results are implementation validation and local profiling. The GH200 Qwen/Llama
results above are the current CUDA validation results; the earlier A100 run is retained as a reduced
pilot.

### Best Speedup Cases

| prompt_id | prompt_type | temperature | draft_k | speedup | acceptance | verifier calls/output token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qa_002` | factual_qa | 0.0 | 2 | 3.0514x | 0.5862 | 0.6042 |
| `qa_002` | factual_qa | 0.0 | 4 | 2.9321x | 0.4595 | 0.3958 |
| `qa_002` | factual_qa | 0.0 | 1 | 2.7565x | 0.7083 | 1.0000 |
| `qa_002` | factual_qa | 0.0 | 8 | 2.2033x | 0.2906 | 0.3125 |
| `qa_001` | factual_qa | 0.0 | 2 | 2.0941x | 0.5965 | 0.6042 |

### Worst Slowdown Cases

| prompt_id | prompt_type | temperature | draft_k | speedup | acceptance | draft overhead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `json_002` | structured_json | 1.0 | 1 | 0.1059x | 0.8125 | 818.2419 ms |
| `json_002` | structured_json | 0.7 | 8 | 0.1099x | 0.3980 | 1401.2504 ms |
| `json_002` | structured_json | 0.7 | 4 | 0.1117x | 0.5571 | 1198.2196 ms |
| `json_002` | structured_json | 1.0 | 8 | 0.1146x | 0.3980 | 1321.7959 ms |
| `json_002` | structured_json | 0.7 | 1 | 0.1162x | 0.8125 | 742.0027 ms |

### Speedup By Prompt Type

| prompt_type | mean speedup | slowdown rate | mean acceptance |
| --- | ---: | ---: | ---: |
| code_completion | 0.7003x | 92.71% | 0.4504 |
| factual_qa | 0.7962x | 91.67% | 0.5157 |
| open_ended | 0.8611x | 69.79% | 0.7047 |
| structured_json | 0.7912x | 78.13% | 0.7041 |
| summarization | 0.7887x | 86.46% | 0.6231 |

### Speedup By Temperature

| temperature | mean speedup | slowdown rate | mean acceptance |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.9335x | 70.83% | 0.5996 |
| 0.3 | 0.7774x | 85.00% | 0.5996 |
| 0.7 | 0.7274x | 89.17% | 0.5996 |
| 1.0 | 0.7118x | 90.00% | 0.5996 |

Acceptance is identical across temperatures because the implemented speculative verifier is
greedy-only. Non-zero-temperature rows are diagnostic timing comparisons, not exact speculative
sampling.

### Draft Length Tradeoff

| draft_k | mean speedup | slowdown rate | mean acceptance | verifier calls/output token | draft overhead |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6525x | 95.83% | 0.7736 | 1.0000 | 734.9888 ms |
| 2 | 0.8740x | 91.67% | 0.6892 | 0.5722 | 706.7333 ms |
| 4 | 0.9274x | 60.83% | 0.5572 | 0.3708 | 808.4145 ms |
| 8 | 0.6960x | 86.67% | 0.3783 | 0.2889 | 1253.8565 ms |

`draft_k=8` reduced verifier calls the most but had worse latency than `draft_k=4` because draft
overhead increased enough to erase the verifier-call savings.

### Plots

- `results/plots/speedup_by_prompt_type.png`
- `results/plots/speedup_by_temperature.png`
- `results/plots/speedup_by_draft_k.png`
- `results/plots/acceptance_rate_by_prompt_type.png`
- `results/plots/acceptance_rate_by_temperature.png`
- `results/plots/acceptance_vs_speedup.png`
- `results/plots/slowdown_rate_by_condition.png`

## Reproduce

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Editable install with CLI:

```bash
pip install -e .
draftverifybench --help
```

Run the local small benchmark:

```bash
python scripts/run_benchmark.py \
  --config configs/local_small.yaml \
  --out results/local_small_results.csv \
  --raw-out results/local_small_raw.jsonl \
  --metadata-out results/local_small_metadata.json \
  --max-prompts 25
```

Summarize results:

```bash
python scripts/summarize_results.py \
  --inputs results/local_small_results.csv \
  --out results/local_small_summary.md \
  --tables-out results/local_small_tables \
  --correlation-out results/correlation_analysis.md \
  --correlation-csv results/correlation_analysis.csv \
  --raw-jsonl results/local_small_raw.jsonl \
  --equivalence-out results/output_equivalence_check.md
```

Generate plots:

```bash
python scripts/make_plots.py \
  --inputs results/local_small_results.csv \
  --out-dir results/plots
```

Run validation:

```bash
python -m pytest
python -m ruff check .
```

CLI equivalents:

```bash
draftverifybench run --config configs/local_small.yaml
draftverifybench summarize --results results/local_small_results.csv
draftverifybench plot --results results/local_small_results.csv
draftverifybench adaptive --config configs/adaptive_local.yaml
draftverifybench router --config configs/router_local.yaml
```

## GPU Smoke Test

On a CUDA machine, run a small smoke test first:

```bash
python scripts/run_benchmark.py \
  --config configs/gpu_smoke.yaml \
  --out results/gpu_smoke_results.csv \
  --raw-out results/gpu_smoke_raw.jsonl \
  --metadata-out results/gpu_smoke_metadata.json \
  --max-prompts 5
```

For a Llama-scale run, edit `configs/gpu_llama_1b_8b.yaml` with locally available model names, or
set `DRAFTVERIFY_DRAFT_MODEL` and `DRAFTVERIFY_VERIFIER_MODEL`, then follow
`docs/GPU_Runbook.md`.

## Adaptive Draft Length

DraftVerifyBench also includes an experimental adaptive scheduling path:

- `confidence_threshold`
- `entropy_threshold`
- `rolling_acceptance`

Adaptive scheduling is a new mode, not a replacement for static `draft_k`. It records selected
`k` history, average selected `k`, wasted draft tokens, and accepted/rejected tokens by selected
draft length. Use `configs/adaptive_local.yaml` for local experiments and
`scripts/analyze_adaptive.py` for comparison against static baselines.

## Ablations And Prompt Routing

Month 3 tooling adds:

- ablation configs for static `k`, confidence thresholds, entropy thresholds, and rolling windows
- `scripts/analyze_ablations.py`
- lightweight prompt feature extraction in `draftverifybench/routing.py`
- router configs and `scripts/run_router_experiment.py`
- `scripts/analyze_router.py`

Reduced local Month 3 experiments have now been run. Because CUDA was unavailable, the ablation
and router runs were reduced and should be treated as local small-model evidence only.

Month 3 result:

- Ablation rows: 102, 0 errors
- Router rows: 120, 0 errors
- Best static policy: `static_k=1`, median speedup `0.7642x`, slowdown rate `66.67%`
- Best adaptive policy: `rolling_w2`, median speedup `0.5349x`, slowdown rate `83.33%`
- Feature router: median speedup `0.8110x`, slowdown rate `53.33%`
- Always baseline: median speedup `1.0000x`, slowdown rate `0.00%`

The router reduced slowdown cases relative to always enabling speculative decoding (`100.00%` to
`53.33%`), but it did not beat always-baseline decoding. Adaptive scheduling did not beat the best
static `k` in this reduced run. See `results/month3_primary_claim.md`.

## OSS Tooling

Docs added for standalone usage:

- `docs/Installation.md`
- `docs/Quickstart.md`
- `docs/CLI_Reference.md`
- `docs/Result_Schema.md`
- `docs/Serving_Integration_Notes.md`
- `docs/OSS_PR_Plan.md`

## Limitations

- Completed experiments used local small models only: `distilgpt2 -> gpt2`.
- Latency is hardware-dependent and was measured on Apple MPS, not CUDA.
- CUDA/Llama-scale benchmark support is implemented, but the Llama-scale run is pending.
- The prompt suite is intentionally small and synthetic.
- Greedy speculative decoding is implemented and validated.
- Exact speculative sampling with probability correction is not implemented.
- Non-zero-temperature speculative rows are diagnostic, not exact distribution-preserving sampling.
- This is a local profiler and benchmark, not a production serving engine.
- No GPU kernel-level optimization was implemented.
- These results do not prove universal speedups and should not be generalized to frontier models.

## Relationship To My Other Work

This project complements ReceiptInject, my LLM agent safety/evals infrastructure project.
ReceiptInject evaluates agent safety and tool-boundary failures; DraftVerifyBench profiles
inference optimization tradeoffs.

## Repository Note

DraftVerifyBench currently lives inside a larger local project directory. Before publishing it on
GitHub, move `DraftVerifyBench/` into its own standalone repository and keep `.cache/`, model
weights, virtual environments, and local secret files out of version control.
