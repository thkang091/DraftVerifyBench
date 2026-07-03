# DraftVerifyBench Summary - 250 Words

DraftVerifyBench is a local benchmarking framework for studying speculative decoding as an LLM
inference optimization. It compares standard autoregressive decoding against greedy speculative
decoding across prompt types, temperatures, draft lengths, and model pairs. The goal is not to
claim universal speedups, but to profile when the optimization helps and when it adds overhead.

I built the full experimental pipeline: Hugging Face model loading with CUDA/MPS/CPU device
selection, baseline decoding, greedy speculative decoding, prompt suites, YAML configs, benchmark
runner scripts, latency and token metrics, acceptance-rate logging, verifier-call counters, draft
overhead measurement, raw JSONL traces, CSV summaries, correlation analysis, slowdown-case
analysis, output-equivalence checks, plots, tests, and a technical report.

The completed run used Apple MPS with `distilgpt2` as the draft model and `gpt2` as the verifier.
The reduced local-small grid produced 600 rows: 120 baseline and 480 speculative. Speculative
decoding averaged `0.7875x` speedup and slowed down in `82.92%` of speculative rows, but the best
condition, factual-QA prompt `qa_002` at `temperature=0.0` and `draft_k=2`, achieved `3.0514x`
speedup. The worst condition, structured-JSON prompt `json_002` at `temperature=1.0` and
`draft_k=1`, slowed to `0.1059x` despite `0.8125` acceptance.

The main systems finding is that acceptance rate alone was weakly predictive: acceptance-vs-speedup
correlation was only `0.1352`. Speculative decoding needs condition-aware profiling that considers
draft overhead and verifier-call savings, not just accepted-token rate.

