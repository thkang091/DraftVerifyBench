# DraftVerifyBench Interview Talking Points

## Explain Speculative Decoding In 60 Seconds

Speculative decoding tries to reduce LLM inference latency by using a smaller draft model to
propose several future tokens. A larger verifier model checks those tokens. If the verifier agrees
with the draft tokens, the system can emit multiple tokens while doing fewer verifier calls. If the
verifier disagrees, the rejected draft work is wasted and the verifier token is used instead. The
optimization works only when the accepted-token savings are larger than the draft-model overhead.

## Draft Model And Verifier Model

The draft model is smaller and cheaper. It proposes candidate tokens. The verifier model is larger
and defines the output behavior. In my completed run, the draft was `distilgpt2` and the verifier
was `gpt2`.

## Acceptance Rate

Acceptance rate is the fraction of proposed draft tokens accepted by the verifier. Higher
acceptance usually helps because it reduces verifier work. In this run, acceptance rate alone was
not enough: acceptance-vs-speedup correlation was only `0.1352`, because draft overhead and
baseline timing also mattered.

## Why High Temperature Can Hurt

Higher temperature makes baseline sampling more variable. In this implementation, speculative
verification is greedy-only, so non-zero-temperature rows are diagnostic rather than exact
speculative sampling. The measured mean speedup fell from `0.9335x` at `temperature=0.0` to
`0.7118x` at `temperature=1.0`, and slowdown rate rose from `70.83%` to `90.00%`.

## Why Larger Draft Models Can Erase Speedup

A larger or slower draft model can improve agreement with the verifier, but it also costs more per
proposed token. If the extra accepted tokens do not save enough verifier time, net latency gets
worse. I did not complete a larger-draft experiment because `gpt2-medium` was too slow for this
local session; the report documents that limitation.

## Why `draft_k` Has A Tradeoff

Increasing `draft_k` can reduce verifier calls, but it also increases the chance that one token in
the block is rejected and increases draft overhead. In my run, `draft_k=8` reduced verifier calls
to `0.2889` per output token but averaged only `0.6960x` speedup. `draft_k=4` was better overall:
`0.9274x` mean speedup with `0.3708` verifier calls per output token.

## Relation To TensorRT-LLM, vLLM, And Production Serving

This project is a local profiler, not a serving engine. In production systems like TensorRT-LLM or
vLLM, the same tradeoffs apply, but implementation details change: batching, KV-cache management,
scheduler policy, CUDA kernels, draft/verifier placement, and continuous batching can all affect
whether speculative decoding helps. A natural next step would be integrating this analysis with
vLLM or TensorRT-LLM's speculative decoding paths.

## What Would Change On GPU

On CUDA, verifier calls may be much faster, batching may change the cost model, and optimized
kernels can reduce overhead. That could shift the best `draft_k` and change whether draft overhead
dominates. I would rerun the same benchmark on NVIDIA hardware before making any GPU-specific
claim.

## Limitations

- Completed run used local small models only.
- Device was Apple MPS, not CUDA.
- Greedy speculative decoding is validated; exact speculative sampling is future work.
- Prompt suite is small and synthetic.
- No production serving engine or kernel optimization was implemented.

## What I Would Improve With More Time

- Add exact speculative sampling with distribution correction.
- Run on CUDA and compare to MPS/CPU.
- Add larger and instruction-tuned model pairs.
- Integrate with vLLM or TensorRT-LLM.
- Add batched verifier checks and better warmup-controlled timing.
- Expand prompt suites with real workload traces.

