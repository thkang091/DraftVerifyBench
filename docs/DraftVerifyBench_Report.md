# DraftVerifyBench Report

## 1. Abstract

DraftVerifyBench is a local empirical profiler for speculative decoding. It compares standard
autoregressive decoding against greedy speculative decoding across prompt types, temperatures, and
draft lengths. In the completed Apple MPS run with `distilgpt2 -> gpt2`, speculative decoding was
not universally faster: it averaged `0.7875x` speedup and slowed down in `82.92%` of speculative
rows. The best individual condition reached `3.0514x` speedup, showing that speculative decoding
can help in narrow settings, but only when verifier-call savings overcome draft-model overhead.

## 2. Motivation

LLM serving systems are often bottlenecked by sequential token generation. Speculative decoding is
attractive because a smaller model can propose tokens and a larger model can verify them, reducing
expensive verifier calls when many draft tokens are accepted. The method is not free: rejected
tokens waste draft compute, and the draft model itself adds latency. DraftVerifyBench was built to
measure these tradeoffs locally rather than assume that speculative decoding always improves
latency.

## 3. Speculative Decoding Background

In speculative decoding, a draft model proposes a block of tokens. A verifier model checks the
proposed continuation. Accepted draft tokens are emitted without requiring separate verifier calls
for every generated token. When a proposed token is rejected, the verifier's token is used and the
process continues.

The net speedup depends on four competing forces:

- accepted draft tokens reduce verifier work
- rejected draft tokens waste draft compute
- larger `draft_k` can reduce verifier calls but increases rejection waste
- draft-model overhead can erase verifier-call savings

This implementation validates greedy speculative decoding. Exact speculative sampling with
probability correction is not implemented.

## 4. System Design

DraftVerifyBench is organized around a reproducible benchmark pipeline:

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

Core modules:

- `draftverifybench/models.py`: Hugging Face model loading, device selection, dtype handling, and
  parameter counts
- `draftverifybench/decoding.py`: baseline autoregressive decoding
- `draftverifybench/speculative.py`: greedy speculative decoding and acceptance tracking
- `draftverifybench/metrics.py`: benchmark metric schema and speedup/slowdown calculations
- `draftverifybench/datasets.py`: built-in prompt suite
- `draftverifybench/runner.py`: config-driven benchmark execution and artifact writing
- `draftverifybench/analysis.py`: summary tables, correlations, slowdown rates, and equivalence
  checks

## 5. Benchmark Setup

Completed run:

- Config: reduced `local_small`
- Device: Apple MPS
- Platform: `macOS-15.6.1-arm64-arm-64bit`
- Torch: `2.12.0`
- Transformers: `4.41.2`
- Draft model: `distilgpt2`, `81,912,576` parameters
- Verifier model: `gpt2`, `124,439,808` parameters
- Dtype: `torch.float16`
- Prompts: 15 built-in prompts
- Prompt types: code completion, structured JSON, factual QA, summarization, open-ended
- Temperatures: `0.0`, `0.3`, `0.7`, `1.0`
- Draft lengths: `1`, `2`, `4`, `8`
- Repetitions: 2
- Max new tokens: 48
- Rows: 600 total, 120 baseline, 480 speculative

The original `local_small` grid was reduced from 64 to 48 max new tokens and from 3 to 2
repetitions after a one-prompt probe at 64 tokens and 3 repetitions took about 100 seconds for 12
work items. The medium run (`gpt2 -> gpt2-medium`) was skipped because the reduced small run took
12m18s on MPS.

These GPT-2-family results validate the benchmark implementation and expose local tradeoffs. They
are not sufficient for serious Llama-scale inference-systems claims. CUDA support, GPU metadata,
GPU configs, and a runbook have been added, but the Llama-scale CUDA run remains pending until
executed on suitable hardware.

## 6. Metrics

DraftVerifyBench records:

- total latency
- time to first token
- tokens per second
- generated token count
- draft tokens proposed, accepted, and rejected
- acceptance rate
- verifier forward calls
- draft forward calls
- verifier calls per output token
- draft overhead
- speedup versus matched baseline
- slowdown flag
- exact greedy output match with baseline
- prompt type, temperature, draft length, model pair, and repetition

Speedup is computed as:

```text
baseline latency / speculative latency
```

Values above `1.0x` indicate speedup. Values below `1.0x` indicate slowdown.

## 7. Results

Overall speculative results:

- Mean speedup: `0.7875x`
- Median speedup: `0.7333x`
- Minimum speedup: `0.1059x`
- Maximum speedup: `3.0514x`
- Slowdown rate: `82.92%`
- Mean acceptance rate: `0.5996`
- Acceptance range: `0.2013` to `0.8750`
- Mean verifier calls per output token: `0.5580`
- Mean draft overhead: `875.9983 ms`

Best individual case:

- Prompt: `qa_002`
- Prompt type: factual QA
- Temperature: `0.0`
- Draft length: `2`
- Speedup: `3.0514x`
- Acceptance: `0.5862`
- Verifier calls per output token: `0.6042`

Worst individual case:

- Prompt: `json_002`
- Prompt type: structured JSON
- Temperature: `1.0`
- Draft length: `1`
- Speedup: `0.1059x`
- Acceptance: `0.8125`
- Draft overhead: `818.2419 ms`

## 8. Acceptance-Rate Analysis

Acceptance rate was necessary but not sufficient. The correlation between acceptance rate and
speedup was only `0.1352`. Several rows had high acceptance but still slowed down because draft
overhead and baseline timing dominated the net latency.

Mean acceptance by prompt type:

- open-ended: `0.7047`
- structured JSON: `0.7041`
- summarization: `0.6231`
- factual QA: `0.5157`
- code completion: `0.4504`

Mean acceptance by `draft_k`:

- `k=1`: `0.7736`
- `k=2`: `0.6892`
- `k=4`: `0.5572`
- `k=8`: `0.3783`

The decline with larger `draft_k` is expected: longer draft blocks create more chances for a
left-to-right mismatch.

## 9. Slowdown Cases

The benchmark deliberately reports slowdowns. The worst slowdown was `json_002` at
`temperature=1.0`, `draft_k=1`, with `0.1059x` speedup despite `0.8125` acceptance. Other severe
slowdowns also came from `json_002` at higher temperatures and larger draft lengths.

Slowdown rate by temperature:

- `temperature=0.0`: `70.83%`
- `temperature=0.3`: `85.00%`
- `temperature=0.7`: `89.17%`
- `temperature=1.0`: `90.00%`

Non-zero-temperature rows are diagnostic because the speculative verifier is greedy-only. They
still show that latency benefit can degrade as the baseline sampling behavior changes.

## 10. Draft-Length Tradeoff

`draft_k=4` had the best average tradeoff:

| draft_k | mean speedup | mean acceptance | verifier calls/output token | draft overhead |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6525x | 0.7736 | 1.0000 | 734.9888 ms |
| 2 | 0.8740x | 0.6892 | 0.5722 | 706.7333 ms |
| 4 | 0.9274x | 0.5572 | 0.3708 | 808.4145 ms |
| 8 | 0.6960x | 0.3783 | 0.2889 | 1253.8565 ms |

`draft_k=8` reduced verifier calls the most, but it had the highest draft overhead and lower
acceptance. In this run, reducing verifier calls further did not compensate for the extra draft
work.

## 11. Output Equivalence

The greedy output-equivalence check covered 120 speculative rows at `temperature=0.0`:

- Exact output match rate: `1.0000`
- Token-level match rate: `1.0000`
- Differences observed: none

This supports the correctness of the greedy speculative implementation. It does not validate exact
speculative sampling for non-zero-temperature generation.

## 12. Hardware Caveats

The completed run used Apple MPS, not CUDA. Timing behavior can differ substantially on NVIDIA
GPUs, especially if draft and verifier execution are batched or integrated into a serving engine.
The run used local wall-clock measurements and includes Python overhead. It should be interpreted
as a profiler for algorithmic tradeoffs, not as a production serving benchmark.

## 13. Limitations

- Completed experiments used local small models only.
- The medium model pair was skipped due to runtime on local hardware.
- The prompt suite is small and synthetic.
- Greedy speculative decoding is implemented; exact speculative sampling is not.
- Non-zero-temperature speculative results are diagnostic only.
- No production serving engine was integrated.
- No CUDA kernels, TensorRT-LLM kernels, vLLM scheduler changes, or GPU kernel-level optimizations
  were implemented.
- Results should not be generalized to frontier models without rerunning the benchmark.

## 14. Future Work

- Implement exact speculative sampling with distribution correction.
- Add batched verifier checks and more efficient draft-token proposal.
- Run the benchmark on CUDA hardware with Llama-scale draft/verifier pairs.
- Compare local implementations with vLLM and TensorRT-LLM speculative decoding paths.
- Add larger and instruction-tuned model pairs.
- Expand prompt suites and include real workload traces.
- Add confidence intervals and warmup-controlled timing.
- Evaluate adaptive draft-length scheduling against static `draft_k`.

## 16. Month 3: Ablations And Prompt Routing

DraftVerifyBench now includes ablation configs for static `k`, confidence-threshold schedules,
entropy-threshold schedules, and rolling-acceptance windows:

- `configs/ablation_local.yaml`
- `configs/ablation_gpu.yaml`

It also includes prompt-routing support:

- prompt length
- punctuation ratio
- code-like markers
- JSON-like markers
- draft first-token entropy
- draft top-1 probability
- draft top-5 probability mass

The router can choose baseline decoding, static speculative decoding, or adaptive speculative
decoding. This is intentionally lightweight and interpretable. It is not a production router.

Reduced local Month 3 experiments were run after the tooling was added. CUDA was unavailable, so
these results are local small-model evidence only.

Ablation:

- rows: 102
- row errors: 0
- best static policy: `static_k=1`
- best static median speedup: `0.7642x`
- best static slowdown rate: `66.67%`
- best adaptive policy: `rolling_w2`
- best adaptive median speedup: `0.5349x`
- best adaptive slowdown rate: `83.33%`

Router:

- rows: 120
- row errors: 0
- always baseline median speedup: `1.0000x`
- always baseline slowdown rate: `0.00%`
- feature router median speedup: `0.8110x`
- feature router slowdown rate: `53.33%`
- always static speculative slowdown rate: `100.00%`
- always adaptive slowdown rate: `100.00%`

The Month 3 conclusion is negative but useful: adaptive scheduling did not beat the best static
draft length, and the feature router reduced slowdowns compared with always-speculative policies
but did not beat always-baseline decoding. See `results/month3_primary_claim.md`.

## 17. Month 4: OSS Tooling And Serving Integration Path

DraftVerifyBench now exposes a package CLI:

- `draftverifybench run`
- `draftverifybench summarize`
- `draftverifybench plot`
- `draftverifybench adaptive`
- `draftverifybench router`

The working backend is the Hugging Face backend. vLLM and SGLang backend files are explicit stubs
that raise `NotImplementedError` until their APIs are verified for a target version. The integration
path is documented in `docs/Serving_Integration_Notes.md`.

## 15. Conclusion

DraftVerifyBench shows that speculative decoding is a conditional optimization. In this local MPS
run, it produced strong individual wins but slowed down on average. The best systems insight is not
"speculative decoding is faster"; it is that speedup depends on the interaction between acceptance
rate, verifier-call savings, draft overhead, prompt type, temperature, and draft length. That makes
speculative decoding a good candidate for profiled, condition-aware serving policies rather than a
universal default.
