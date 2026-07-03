# Speculative Decoding Is Conditional: GH200 Results From Qwen and Llama

Speculative decoding is often described as an LLM inference acceleration technique: use a smaller
draft model to propose future tokens, then use a larger verifier model to accept or reject them. If
enough draft tokens are accepted, the verifier does fewer expensive forward passes and generation can
get faster.

That is the optimistic story. I built DraftVerifyBench to test the systems question behind it:

> When does speculative decoding actually make generation faster, and when does it backfire?

## The Short Version

On an NVIDIA GH200 480GB, I ran a full static speculative-decoding grid across two model families:

- Qwen2.5 0.5B draft -> Qwen2.5 7B verifier
- Llama 3.2 1B draft -> Llama 3.1 8B verifier

The final benchmark used two seeds per model family and produced:

- `1,800` total benchmark rows
- `1,440` speculative rows
- `0` row errors

| model pair | speculative rows | mean speedup | median speedup | best speedup | slowdown rate |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B -> Qwen2.5 7B | 720 | `0.5936x` | `0.5826x` | `1.5739x` | `99.17%` |
| Llama 3.2 1B -> Llama 3.1 8B | 720 | `1.0667x` | `0.8689x` | `12.8418x` | `67.36%` |

The result was not "speculative decoding is always faster" or "speculative decoding is always
slower." It was a regime story:

> Llama with `k=4` and `k=8` crossed break-even on median, especially around structured/code-like
> workloads. Qwen was stably slower. A simple seed-held-out router learned an expected-value gate:
> preserve useful Llama regimes, cut slowdown risk, and disable speculation for Qwen.

## The Most Important Detail

Even for Llama, the median speculative row was slower than baseline:

```text
mean speedup:   1.0667x
median speedup: 0.8689x
slowdown rate:  67.36%
```

So the headline is not "Llama speculation is faster." The headline is:

> Llama speculation had concentrated high-value regions, especially at larger draft lengths, but
> most rows still slowed down.

That is exactly the kind of conditional behavior a serving system needs to understand before
enabling speculative decoding broadly.

## The Router Is The Practical Fix

After measuring the regimes, I validated a simple router using the existing GH200 grid. It trained
on one seed and evaluated on the held-out seed.
Each policy row below is `N=180` validation cases per model family: two seed-held-out folds with
`90` validation cases each. A case is one prompt/temperature/repetition condition with baseline and
candidate `k=1`, `k=2`, `k=4`, and `k=8` outcomes.

| model | policy | mean speedup | slowdown rate | speculative share | speculated slowdown |
|---|---|---:|---:|---:|---:|
| Llama | always speculate k4 | `1.2068x` | `45.00%` | `100.00%` | `45.00%` |
| Llama | trained regime router | `1.2294x` | `37.22%` | `70.00%` | `53.17%` |
| Llama | hindsight oracle | `1.4363x` | `0.00%` | `55.56%` | `0.00%` |
| Qwen | always speculate k4 | `0.6520x` | `98.89%` | `100.00%` | `98.89%` |
| Qwen | trained regime router | `1.0000x` | `0.00%` | `0.00%` | `0.00%` |

The router is intentionally simple. It is not a new speculative decoding algorithm. It is a routing
gate over measured regimes. That is the point: once the benchmark characterizes where speculation
helps and hurts, the system can choose regimes with positive expected value.

The Llama router is the real validation target. It speculated on `70%` of traffic, and that selected
traffic averaged `1.3277x`. It reduced all-row slowdown rate from `45.00%` to `37.22%` versus
always speculating with `k=4`, while capturing `52.58%` of the hindsight-oracle gain above
baseline. It still lost on `53.17%` of selected speculative rows, so the correct interpretation is
not "the router is usually right." It is "the router is positive in expectation because the wins are
larger than the more frequent row-level losses." The router used only pre-generation features, model
family and oracle-labeled prompt type, not realized acceptance rate or speedup.

For the Llama trained router, the selected-row slowdown rate is `67 / 126 = 53.17%`: `126` of `180`
validation cases used speculation, and `67` of those selected speculative cases were slower than
baseline.

The Qwen router result is better framed as a sanity check: the correct learned action was to never
speculate for that model pair. That is useful, but the sharper evidence is the Llama within-family
discrimination.

## Draft Length Mattered

For Llama, larger static draft lengths were much better:

| draft_k | mean speedup | median speedup | slowdown rate |
|---:|---:|---:|---:|
| 1 | `0.8210x` | `0.6593x` | `90.00%` |
| 2 | `1.1047x` | `0.9360x` | `88.33%` |
| 4 | `1.2068x` | `1.1270x` | `45.00%` |
| 8 | `1.1342x` | `1.0675x` | `46.11%` |

For Qwen, every draft length remained below break-even on median.

## Workload Mattered Too

Structured JSON was the strongest Llama workload by mean speedup:

| prompt type | Llama mean speedup | Qwen mean speedup |
|---|---:|---:|
| structured JSON | `1.5408x` | `0.6467x` |
| summarization | `1.0520x` | `0.5179x` |
| code completion | `1.0295x` | `0.6930x` |
| open ended | `0.8702x` | `0.5488x` |
| factual QA | `0.8411x` | `0.5618x` |

This is the core project thesis in one table: same hardware, same benchmark shape, different model
family and workload, very different result.

## External Comparisons

I also compared against Hugging Face assisted generation on a representative subset.

| model pair | mean speedup vs HF generate | median speedup | slowdown rate |
|---|---:|---:|---:|
| Llama 1B -> 8B | `0.8082x` | `0.7668x` | `77.78%` |
| Qwen 0.5B -> 7B | `0.4735x` | `0.4252x` | `100.00%` |

That matters because it addresses the obvious skepticism:

> Maybe the custom implementation is just slow.

On this subset, the official Transformers assisted-generation path was also mostly slower.

I also tested vLLM on a representative Llama subset:

| vLLM mode | latency | tokens/sec | speedup |
|---|---:|---:|---:|
| baseline | `395.66 ms` | `1316.79` | baseline |
| speculative | `3173.59 ms` | `164.17` | `0.1247x` |

For Qwen, vLLM refused draft-model speculation because the draft and verifier vocabulary sizes did
not match. That is a useful serving compatibility finding:

```text
target vocab_size = 152064
draft vocab_size  = 151936
```

## What I Learned

The useful lesson is not that speculative decoding is good or bad. The useful lesson is that it is
a systems tradeoff.

It depends on:

- model pair,
- workload,
- draft length,
- draft overhead,
- verifier-call reduction,
- tokenizer/backend compatibility,
- and serving implementation.

Acceptance rate alone is not enough. A draft model can agree with the verifier and still be too
expensive. A backend can support speculative decoding and still lose on a particular workload.

## What This Does And Does Not Claim

This project does claim:

- speculative decoding was conditional across Qwen and Llama on GH200,
- Llama had repeatable useful regimes across two seeds,
- Qwen was stably slower across two seeds,
- larger Llama draft lengths crossed break-even on median,
- a seed-held-out regime router reduced Llama slowdown, selected `1.3277x` speculative traffic, and
  disabled speculation for Qwen,
- HF assisted generation and vLLM did not automatically rescue the slowdown story.

This project does not claim:

- speculative decoding is universally bad,
- speculative decoding is universally good,
- these exact numbers generalize to every serving stack,
- oracle-labeled benchmark prompt types are available for free in production,
- non-zero-temperature custom rows are exact speculative sampling with probability correction.

The honest conclusion is stronger than a hype conclusion:

> DraftVerifyBench shows why speculative decoding needs profiling and routing gates. It can produce
> large wins, but the wins are conditional and backend/model-pair dependent.
