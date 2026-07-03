# DraftVerifyBench GH200 Showcase Report

## Executive Summary

DraftVerifyBench is a benchmark and profiler for one inference-systems question:

> When does speculative decoding actually make LLM generation faster, and when does it backfire?

The answer is a regime story, not a single mean-speedup number:

- Llama `k=4` and `k=8` crossed break-even on median.
- Llama structured JSON produced the largest wins, including a `12.8418x` best case.
- Llama factual QA and much of open-ended generation mostly lost.
- Qwen was stably slower across prompt types and seeds.
- A seed-held-out regime router learned to speculate selectively for Llama and disable speculation
  for Qwen.

The final GH200 validation is now much stronger than the earlier reduced A100 pilot. It includes:

- `1,800` full-static benchmark rows
- `1,440` speculative rows
- `2` model families: Qwen and Llama
- `2` seeds per model family
- `0` row errors
- variance summaries
- Hugging Face assisted-generation comparison
- vLLM comparison
- seed-held-out router validation
- CUDA profiler traces archived separately

The main finding:

> Speculative decoding was model-pair and workload dependent. Llama had useful regimes, especially
> larger draft lengths and structured/code-like workloads, while Qwen should be routed to baseline
> on this grid. A simple seed-held-out regime router reduced Llama slowdown risk versus always
> speculating. The selected speculative traffic averaged `1.3277x`, even though `53.17%` of those
> selected rows still lost to baseline.

## Hardware

The full static validation used:

- GPU: `NVIDIA GH200 480GB`
- CUDA version: `13.0`
- driver: `580.105.08`
- platform: `Linux-6.8.0-1046-nvidia-64k-aarch64-with-glibc2.39`
- dtype: `torch.bfloat16`

## Full Static Benchmark Shape

Each model family used:

- 15 prompts across 5 prompt types
- temperatures: `0.0`, `0.3`, `0.7`
- static draft lengths: `1`, `2`, `4`, `8`
- seeds: `42`, `43`
- repetitions per seed config: `2`
- max new tokens: `128`
- schedules: static only

The final full-static files are:

- `results/gpu_llama_full_static_results.csv`
- `results/gpu_llama_full_static_seed43_results.csv`
- `results/gpu_qwen_full_static_results.csv`
- `results/gpu_qwen_full_static_seed43_results.csv`

## Final Model-Pair Result

| model pair | speculative rows | mean speedup | median speedup | std | best | worst | slowdown rate | mean acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B -> Llama 3.1 8B | 720 | `1.0667x` | `0.8689x` | `1.2221` | `12.8418x` | `0.0407x` | `67.36%` | `0.7271` |
| Qwen2.5 0.5B -> Qwen2.5 7B | 720 | `0.5936x` | `0.5826x` | `0.1343` | `1.5739x` | `0.3315x` | `99.17%` | `0.5968` |

The Llama result is positive on mean but negative on median. That mean-vs-median gap is the main
reason this project should lead with regimes rather than a single average. The gains are real but
concentrated. The Qwen result is stable and strongly negative.

## Router Validation

After characterizing the regimes, I validated whether a simple router could select regimes with
positive expected value. The router was trained on one seed and evaluated on the held-out seed.
Each router row has `N=180` validation cases per model family: `90` cases from validating on seed
42 plus `90` cases from validating on seed 43. A case is one prompt/temperature/repetition condition
with baseline plus candidate `k=1`, `k=2`, `k=4`, and `k=8` outcomes.

| model | policy | mean speedup | slowdown rate | speculative share | speculated mean | speculated slowdown |
|---|---|---:|---:|---:|---:|---:|
| Llama | never speculate | `1.0000x` | `0.00%` | `0.00%` | `1.0000x` | `0.00%` |
| Llama | always speculate k4 | `1.2068x` | `45.00%` | `100.00%` | `1.2068x` | `45.00%` |
| Llama | trained regime router | `1.2294x` | `37.22%` | `70.00%` | `1.3277x` | `53.17%` |
| Llama | oracle per row | `1.4363x` | `0.00%` | `55.56%` | `1.7853x` | `0.00%` |
| Qwen | never speculate | `1.0000x` | `0.00%` | `0.00%` | `1.0000x` | `0.00%` |
| Qwen | always speculate k4 | `0.6520x` | `98.89%` | `100.00%` | `0.6520x` | `98.89%` |
| Qwen | trained regime router | `1.0000x` | `0.00%` | `0.00%` | `1.0000x` | `0.00%` |
| Qwen | oracle per row | `1.0062x` | `0.00%` | `1.11%` | `1.5559x` | `0.00%` |

The Llama router result is best stated as risk reduction plus selectivity:

- all-row slowdown rate dropped from `45.00%` with always-`k=4` to `37.22%`
- speculative share dropped from `100.00%` to `70.00%`
- conditional mean speedup on selected speculative traffic was `1.3277x`
- conditional slowdown rate on selected speculative traffic was still `53.17%`
- the router captured `52.58%` of the hindsight-oracle gain above baseline

The `53.17%` selected-row slowdown comes from `67 / 126` selected speculative cases. The router is
positive in expectation because the wins it keeps are large enough to outweigh those more frequent
row-level losses.

This moves the project from "I found where speculation breaks" to "I characterized the regimes and
validated an expected-value routing gate." The Qwen router result is a useful sanity check: it
correctly disables a losing model pair, but the stronger router evidence is within Llama, where the
gate discriminates across prompt regimes rather than skipping a whole family. The router is not a
per-row win classifier; it is profitable on average because the wins it keeps are large enough to
outweigh frequent smaller losses.

See `docs/Router_Validation_Report.md` for the full router analysis.

## Seed Stability

| run | rows | mean speedup | median speedup | best | slowdown rate |
|---|---:|---:|---:|---:|---:|
| Llama seed 42 | 360 | `1.1264x` | `0.8794x` | `12.8418x` | `66.11%` |
| Llama seed 43 | 360 | `1.0070x` | `0.8573x` | `12.6823x` | `68.61%` |
| Qwen seed 42 | 360 | `0.5938x` | `0.5818x` | `1.5739x` | `99.17%` |
| Qwen seed 43 | 360 | `0.5935x` | `0.5827x` | `1.5378x` | `99.17%` |

This directly addresses the small-`n` concern from the reduced pilot. The Llama high-speedup region
repeated across two seeds. Qwen remained almost identical across seeds.

## Draft Length Findings

| model | draft_k | rows | mean speedup | median speedup | best | slowdown rate |
|---|---:|---:|---:|---:|---:|---:|
| Llama | 1 | 180 | `0.8210x` | `0.6593x` | `11.1464x` | `90.00%` |
| Llama | 2 | 180 | `1.1047x` | `0.9360x` | `12.8418x` | `88.33%` |
| Llama | 4 | 180 | `1.2068x` | `1.1270x` | `10.1795x` | `45.00%` |
| Llama | 8 | 180 | `1.1342x` | `1.0675x` | `6.1743x` | `46.11%` |
| Qwen | 1 | 180 | `0.5422x` | `0.5401x` | `0.7036x` | `100.00%` |
| Qwen | 2 | 180 | `0.6590x` | `0.6315x` | `1.5009x` | `98.89%` |
| Qwen | 4 | 180 | `0.6520x` | `0.5994x` | `1.5739x` | `98.89%` |
| Qwen | 8 | 180 | `0.5213x` | `0.4345x` | `1.3988x` | `98.89%` |

For Llama, `draft_k=4` was the strongest setting by both mean and median speedup. For Qwen, every
draft length remained below break-even on median and nearly every row slowed down.

## Workload Findings

| model | prompt type | rows | mean speedup | median speedup | best | slowdown rate |
|---|---|---:|---:|---:|---:|---:|
| Llama | structured JSON | 144 | `1.5408x` | `0.8349x` | `12.8418x` | `63.19%` |
| Llama | summarization | 144 | `1.0520x` | `0.9570x` | `2.2461x` | `59.03%` |
| Llama | code completion | 144 | `1.0295x` | `0.8776x` | `3.4082x` | `64.58%` |
| Llama | open ended | 144 | `0.8702x` | `0.9302x` | `1.2387x` | `66.67%` |
| Llama | factual QA | 144 | `0.8411x` | `0.7622x` | `1.5418x` | `83.33%` |
| Qwen | code completion | 144 | `0.6930x` | `0.6867x` | `1.5739x` | `95.83%` |
| Qwen | structured JSON | 144 | `0.6467x` | `0.6642x` | `0.9860x` | `100.00%` |
| Qwen | factual QA | 144 | `0.5618x` | `0.5542x` | `0.7219x` | `100.00%` |
| Qwen | open ended | 144 | `0.5488x` | `0.5561x` | `0.6548x` | `100.00%` |
| Qwen | summarization | 144 | `0.5179x` | `0.5380x` | `0.6298x` | `100.00%` |

Structured JSON was the strongest Llama workload by mean speedup, but also high variance. This is
exactly the kind of workload-conditional result DraftVerifyBench was designed to expose.

## Hugging Face Assisted Generation

I compared the custom transparent decoder against Transformers assisted generation using
`generate(..., assistant_model=...)` on a representative prompt subset.

| model | assisted rows | mean speedup vs HF generate | median speedup | best | slowdown rate | exact match rate |
|---|---:|---:|---:|---:|---:|---:|
| Llama | 9 | `0.8082x` | `0.7668x` | `1.2423x` | `77.78%` | `33.33%` |
| Qwen | 9 | `0.4735x` | `0.4252x` | `0.8707x` | `100.00%` | `55.56%` |

This matters because it addresses the critique that the custom implementation alone caused the
slowdowns. On this subset, the official Transformers assisted path was also mostly slower.

## vLLM Comparison

I also tested vLLM offline generation on a representative Llama subset:

| backend mode | prompts | generated tokens | latency | tokens/sec | speedup |
|---|---:|---:|---:|---:|---:|
| vLLM baseline | 9 | 521 | `395.66 ms` | `1316.79` | baseline |
| vLLM speculative | 9 | 521 | `3173.59 ms` | `164.17` | `0.1247x` |

For this representative subset, vLLM draft-model speculation was much slower than vLLM baseline.

The Qwen vLLM speculative run failed before generation because vLLM requires matching target/draft
vocabulary sizes for draft-model speculation:

```text
Target model vocab_size=152064
Draft model vocab_size=151936
```

That is an important serving-system compatibility finding, not just an inconvenience.

## Profiler Artifacts

CUDA profiler traces were exported and archived separately:

- `gh200_profiler_traces.tar.gz`
- `results/profiles/llama_baseline_trace.json`
- `results/profiles/llama_spec_k4_trace.json`
- `results/profiles/qwen_spec_k4_trace.json`

On GH200, PyTorch/Kineto text-table extraction hit a Unicode parsing issue for some traces, but the
Chrome traces were still exported. Because these traces are large, they should be stored as release
artifacts rather than committed directly to the repository.

## Qwen Overhead Hypothesis

Qwen was stably slower and had higher measured draft overhead than Llama. One concrete hypothesis is
that LM-head/vocabulary projection cost contributes to the anomaly. This is especially worth testing
because vLLM rejected the Qwen draft/verifier pair for draft-model speculation due to vocabulary-size
mismatch.

I added a targeted profiler:

```bash
python scripts/profile_lm_head_overhead.py \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-7B-Instruct \
           meta-llama/Llama-3.2-1B meta-llama/Llama-3.1-8B \
  --device cuda \
  --decode-steps 64
```

It separates transformer-body time from LM-head time during prefill and decode. If Qwen shows a much
larger `decode_lm_head_share`, that would strengthen the causal explanation for its overhead. If it
does not, the anomaly likely lives elsewhere in the decode loop or backend execution path.

## Final Thesis

The strongest final project claim is:

> DraftVerifyBench rigorously characterizes speculative decoding regimes and validates a simple
> routing gate. On GH200, Llama 1B -> 8B had useful regimes, with `draft_k=4` and `draft_k=8`
> crossing break-even on median, while Qwen 0.5B -> 7B was stably slower. A seed-held-out regime
> router reduced slowdown risk for Llama, selected traffic that averaged `1.3277x`, and learned to
> disable speculation for Qwen. The router is an expected-value gate, not a guarantee that each
> selected speculative row wins.
> External comparisons with Transformers assisted generation and vLLM showed that production-style
> speculation can also underperform under some model-pair/backend regimes.

## Why This Is Strong

This project now has:

- real GPU validation on GH200,
- two modern model families,
- repeated seeds,
- full static grid rather than only reduced samples,
- variance summaries,
- negative and positive results,
- custom implementation plus external backend comparisons,
- profiler artifacts,
- correctness caveats,
- and a reproducible CLI/report structure.

That makes the work much stronger than a typical "implemented speculative decoding and got speedup"
project. The main value is the systems judgment: measuring when the optimization works, when it
fails, and how a simple routing policy can select regimes with positive expected value.

## Remaining Caveats

- The full static grid uses two seeds, not a large statistical study.
- The router uses oracle-labeled prompt types from the benchmark grid. A deployed version would need
  a lightweight prompt classifier or heuristic, and classifier errors would reduce the captured
  oracle gap.
- vLLM comparison is a representative subset, not a full vLLM grid.
- HF assisted comparison is a representative subset.
- Profiler traces are archived, but text tables were partially blocked by a PyTorch/Kineto parsing
  issue.
- Exact stochastic speculative sampling is not implemented in the custom decoder.
- The Qwen LM-head/vocabulary overhead hypothesis is prepared but should be validated with
  `scripts/profile_lm_head_overhead.py` before being claimed as a finding.

These caveats should be kept in the README and writeup. They make the project more credible, not
weaker.
