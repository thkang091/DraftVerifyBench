# DraftVerifyBench Summary - 100 Words

DraftVerifyBench is a local ML systems benchmark for profiling when speculative decoding speeds up
or slows down LLM inference. I implemented a Hugging Face model loader, baseline autoregressive
decoder, greedy speculative decoder, prompt suite, YAML benchmark runner, raw JSONL traces, CSV
metrics, plots, correlation analysis, and output-equivalence checks. On Apple MPS with
`distilgpt2 -> gpt2`, speculative decoding averaged `0.7875x` speedup and slowed down in `82.92%`
of speculative rows, but one factual-QA condition reached `3.0514x`. The project shows that
acceptance rate, verifier-call reduction, draft overhead, prompt type, and draft length determine
whether speculative decoding helps.

