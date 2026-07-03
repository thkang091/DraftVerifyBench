# DraftVerifyBench Strict Reviewer Audit

## Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| ML systems relevance | 8.5/10 | Directly targets LLM inference latency, speculative decoding, verifier calls, and draft overhead. |
| Engineering depth | 8.0/10 | Includes decoders, model loading, configs, runner, metrics, raw traces, plots, tests, and reports. |
| Empirical rigor | 8.0/10 | Uses repeated runs, raw outputs, hardware metadata, slowdown cases, equivalence checks, and limitations. |
| Novelty | 7.0/10 | Speculative decoding is known, but the local profiler and honest slowdown analysis are useful. |
| NVIDIA fit | 8.0/10 | Strong inference-optimization framing; would be stronger with CUDA/TensorRT-LLM experiments. |
| OpenAI fit | 8.0/10 | Relevant to model-serving tradeoffs and evaluation discipline; would be stronger with larger models. |
| Google fit | 8.0/10 | Good systems measurement project with reproducibility and careful caveats. |
| Reproducibility | 8.5/10 | YAML configs, scripts, CSV/JSONL artifacts, plots, tests, and documented hardware. |
| Limitations honesty | 9.0/10 | Clearly states local small models, MPS-only, greedy-only, no production serving, no kernels. |

## Strongest Finding

Speculative decoding was not universally faster. It averaged `0.7875x` speedup and slowed down in
`82.92%` of speculative rows, yet one low-temperature factual-QA condition reached `3.0514x`. This
is a useful systems finding because it shows the optimization needs condition-aware profiling.

## Weakest Gap

The completed empirical run used local small models on Apple MPS only. There is no CUDA run, no
production serving integration, no vLLM/TensorRT-LLM comparison, and no exact speculative sampling.

## What A Skeptical Reviewer Will Ask

- Why did the benchmark use GPT-2-family models instead of current instruction-tuned models?
- How would the results change on NVIDIA hardware?
- Does the implementation batch verifier checks efficiently enough?
- Are non-zero-temperature rows distribution-correct?
- How much timing variance exists across repeated full runs?
- Would vLLM or TensorRT-LLM change the draft-overhead tradeoff?

## What To Do With One More Week

- Run the same benchmark on CUDA.
- Add exact speculative sampling with probability correction.
- Add warmup-controlled repeated full runs and confidence intervals.
- Integrate an optimized serving path such as vLLM speculative decoding.
- Add one larger local model pair if hardware allows.
- Expand the prompt suite with real workload traces.

## Final Calibrated Percentile Estimate

As a student or early-career ML/SWE portfolio artifact, this is around the 85th percentile because
it is empirical, reproducible, systems-oriented, and honest about slowdowns. For specialized
production inference-infrastructure roles, it is closer to the 70th percentile until it includes
CUDA, larger models, and a serving-engine integration.

