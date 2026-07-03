# OSS PR Plan

## Candidate 1: Benchmark Example For vLLM Or SGLang

- Target repo: vLLM or SGLang
- Contribution: a small speculative-decoding benchmark example that reports slowdown cases
- Why maintainers might accept it: helps users evaluate when speculation helps
- Smallest useful PR: docs/example only, no scheduler changes
- Do not attempt yet: adaptive scheduler changes in the serving engine

## Candidate 2: Adaptive Lookahead Evaluation Doc

- Target repo: vLLM, SGLang, or DraftVerifyBench standalone repo
- Contribution: methodology for evaluating static vs adaptive draft length
- Why maintainers might accept it: improves measurement discipline
- Smallest useful PR: documentation and reproducible config
- Do not attempt yet: claim production speedups without CUDA results

## Candidate 3: Export Format Compatibility

- Target repo: DraftVerifyBench first
- Contribution: stable CSV/JSONL schema and metadata format
- Why maintainers might accept it: makes benchmark artifacts easier to compare
- Smallest useful PR: schema docs and validation tests
- Do not attempt yet: broad ecosystem standardization

## Candidate 4: Issue Or Discussion With Benchmark Results

- Target repo: vLLM or SGLang
- Contribution: discussion post with local/GPU benchmark results and failure cases
- Why maintainers might accept it: exposes real user-facing tuning questions
- Smallest useful PR: a well-scoped issue with commands and artifacts
- Do not attempt yet: performance claims from Apple MPS GPT-2-only runs

