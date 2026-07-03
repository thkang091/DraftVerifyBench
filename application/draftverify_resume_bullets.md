# DraftVerifyBench - Speculative Decoding Profiler for LLM Inference

- Built a local benchmarking framework comparing standard autoregressive decoding against
  speculative decoding across draft/verifier model pairs, prompt types, temperatures, and draft
  lengths.
- Implemented baseline and greedy speculative decoding paths with acceptance-rate tracking,
  verifier-call counts, draft-token rejection metrics, latency/token metrics, raw JSONL traces, CSV
  summaries, and reproducible YAML configs.
- Ran a 600-row Apple MPS benchmark with `distilgpt2 -> gpt2`, measuring 120 baseline and 480
  speculative rows across 15 prompts, 4 temperatures, 4 draft lengths, and 2 repetitions.
- Found that speculative decoding averaged `0.7875x` speedup and slowed down in `82.92%` of
  speculative rows, while one low-temperature factual-QA condition reached `3.0514x` speedup.
- Analyzed acceptance rate, verifier-call reduction, draft overhead, prompt type, temperature, and
  draft length as predictors of speedup; measured only `0.1352` correlation between acceptance rate
  and speedup.
- Generated plots, slowdown-case analysis, output-equivalence checks, validity audits, and a
  technical report documenting limitations across local hardware and small-model experiments.
- Validated greedy speculative decoding with `100.00%` exact output match and `100.00%` token-level
  match across 120 greedy speculative rows.

