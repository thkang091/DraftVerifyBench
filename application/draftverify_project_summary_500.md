# DraftVerifyBench Summary - 500 Words

DraftVerifyBench is a local ML systems and inference optimization project focused on speculative
decoding. Speculative decoding uses a smaller draft model to propose several tokens and a larger
verifier model to accept or reject them. If enough draft tokens are accepted, the verifier does
less work and latency can improve. If draft tokens are rejected, or if the draft model is too
expensive, speculative decoding can slow inference down.

I built DraftVerifyBench to measure that tradeoff empirically instead of assuming speculative
decoding is always faster. The framework includes a Hugging Face model loader with CUDA, Apple MPS,
and CPU device selection; a standard autoregressive baseline decoder; a greedy speculative decoder;
a synthetic prompt suite across code completion, structured JSON, factual QA, summarization, and
open-ended writing; YAML configs; benchmark runner scripts; CSV metrics; raw JSONL traces; hardware
metadata; plots; summary tables; correlation analysis; slowdown-case analysis; output-equivalence
checks; a validity audit; and tests.

The completed experiment ran locally on Apple MPS with `distilgpt2` as the draft model and `gpt2`
as the verifier. The run produced 600 rows: 120 baseline rows and 480 speculative rows. It used 15
prompts, 4 temperatures, 4 draft lengths, 2 repetitions, and 48 max new tokens. The original local
grid was reduced from 64 tokens and 3 repetitions because a one-prompt probe showed the full grid
was too slow on the available hardware. The medium run with `gpt2 -> gpt2-medium` was skipped and
documented for the same reason.

The primary result was that speculative decoding was conditionally useful but not universally
faster. Across all speculative rows, mean speedup was `0.7875x`, median speedup was `0.7333x`, and
`82.92%` of speculative rows slowed down. However, specific low-temperature factual-QA and code
conditions did produce strong wins. The best individual result was factual-QA prompt `qa_002` at
`temperature=0.0`, `draft_k=2`, with `3.0514x` speedup and `0.5862` acceptance. The worst result
was structured-JSON prompt `json_002` at `temperature=1.0`, `draft_k=1`, with `0.1059x` speedup
despite `0.8125` acceptance.

The most important systems insight was that acceptance rate alone did not explain performance.
Acceptance-rate vs speedup correlation was only `0.1352`. Draft length showed the tradeoff clearly:
`draft_k=8` reduced verifier calls to `0.2889` per output token but averaged only `0.6960x`
speedup because draft overhead rose to `1253.8565 ms`. `draft_k=4` gave the best average tradeoff:
`0.9274x` mean speedup, `0.5572` mean acceptance, and `0.3708` verifier calls per output token.

DraftVerifyBench is not a production serving engine and does not implement GPU kernels. Its value
is as a reproducible profiler for inference optimization decisions: it makes the acceptance,
overhead, verifier-call, prompt-type, and draft-length tradeoffs visible.

