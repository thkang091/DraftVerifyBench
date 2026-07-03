# Serving Integration Notes

DraftVerifyBench currently uses a local Hugging Face backend. vLLM and SGLang integrations are
planned but not implemented.

## vLLM

vLLM is an inference and serving engine with its own scheduler, batching, KV-cache management, and
speculative decoding support. A DraftVerifyBench adapter would need to:

- map prompt suites into vLLM requests
- capture latency and token metrics per request
- map vLLM speculative decoding counters, if exposed, into DraftVerifyBench fields
- record batch size, scheduler settings, model names, and hardware metadata
- compare always-baseline, static speculation, and adaptive/router policies at request level

TODO: verify the exact vLLM Python/server API and speculative configuration flags for the target
version before implementing.

Reference starting point: https://vllm.ai/

## SGLang

SGLang provides a serving/runtime stack for high-throughput LLM inference. A DraftVerifyBench
adapter would need to:

- run prompts through the SGLang runtime or server
- record per-request latency and generated token counts
- identify whether speculative decoding counters are exposed by the chosen SGLang version
- export results in DraftVerifyBench CSV/JSONL schemas

TODO: verify exact SGLang APIs and speculative decoding configuration for the target version before
implementing.

Reference starting point: https://www.sglang.io/

## Why The Hugging Face Backend Is Still Useful

The Hugging Face backend provides an inspectable reference implementation for:

- baseline decoding
- greedy speculative decoding
- adaptive draft-length scheduling
- prompt-routing logic
- correctness/equivalence checks

It is not a substitute for benchmarking a production serving engine.
