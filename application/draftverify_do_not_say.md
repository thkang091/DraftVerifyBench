# DraftVerifyBench Do-Not-Say Guide

## Do Not Say

- "I optimized GPU kernels."
- "This proves speculative decoding is faster."
- "This is production-grade serving."
- "The benchmark is universal."
- "The results generalize to frontier models."
- "Sampling mode is exact."
- "I ran this on NVIDIA GPUs."
- "The medium model results show..." when the medium run was skipped.

## Say Instead

- "I built a local profiler for speculative decoding tradeoffs."
- "The results show conditional speedups under local small-model settings."
- "The project measures acceptance rate, verifier calls, draft overhead, and slowdown cases."
- "The completed run used Apple MPS with `distilgpt2 -> gpt2`."
- "Greedy speculative decoding matched baseline outputs exactly in the equivalence check."
- "Non-zero-temperature rows are diagnostic because exact speculative sampling is not implemented."
- "The next step would be integrating with vLLM or TensorRT-LLM and rerunning on CUDA."

