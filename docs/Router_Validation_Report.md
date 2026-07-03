# Router Validation On The GH200 Full Grid

## Why This Matters

The benchmark found where speculative decoding helps and where it fails. The next question is:

> Can a simple router choose speculation only when the expected value is positive?

This report validates a lightweight regime router using the existing GH200 full-static grid. No new
GPU runs were required. The router is intentionally simple: it chooses between baseline decoding and
static speculative decoding based on model family and prompt type, with choices learned on one seed
and evaluated on the held-out seed.

This is not claimed as a novel routing algorithm. It is a practical validation that the measured
regimes are learnable and can be used as an expected-value gate.

## Method

Inputs:

- `results/gpu_llama_full_static_results.csv`
- `results/gpu_llama_full_static_seed43_results.csv`
- `results/gpu_qwen_full_static_results.csv`
- `results/gpu_qwen_full_static_seed43_results.csv`

Policies compared:

- `never_speculate`: always use baseline, speedup `1.0x`
- `always_speculate_k1`
- `always_speculate_k2`
- `always_speculate_k4`
- `always_speculate_k8`
- `trained_global_router`: train one action per model family on the other seed
- `trained_regime_router`: train one action per model family and prompt type on the other seed
- `oracle_per_row`: a hindsight ceiling that chooses the best action per validation row

For each validation seed, the router only uses the opposite seed for policy selection. This prevents
the router from simply memorizing the validation rows.

Each policy row below has `N=180` validation cases per model family. That is the sum of two
seed-held-out folds:

- train on seed 43, validate on seed 42: `90` cases
- train on seed 42, validate on seed 43: `90` cases

Each case is one prompt/temperature/repetition condition with a baseline latency and candidate
speculative outcomes for `k=1`, `k=2`, `k=4`, and `k=8`. The `N=180` router table is therefore not a
single draft-k slice; it is the two-fold held-out policy evaluation over all prompt types,
temperatures, and repetitions for one model family.

## Leakage Check

The trained routers use only pre-generation routing features:

- `model_family`
- `prompt_type`

The validation decision does not use realized speedup, realized acceptance rate, generated text,
latency, verifier-call count, or any other post-generation field from the held-out row. Ground-truth
speedups are used only on the opposite seed to select the routing table, then the selected policy is
applied to the held-out seed.

That makes the router deployable as a simple gate, assuming a production system has a prompt-type
classifier or route label before generation starts.

## Metric Definitions

- `mean_speedup`: mean routed speedup over all validation rows.
- `slowdown_rate`: fraction of all routed rows with speedup below `1.0x`. Rows routed to baseline
  count as exactly `1.0x`, so they are not slowdowns.
- `speculative_share`: fraction of rows where the router chose speculative decoding.
- `speculated_mean_speedup`: mean speedup only on rows where the router chose speculation.
- `speculated_slowdown_rate`: slowdown rate only on rows where the router chose speculation.

This distinction matters. The main router claim is not that every selected row wins. For Llama, the
router reduces all-row slowdown risk and selects traffic with positive average speedup, even though
more than half of the individual selected speculative rows still underperform baseline.

Generated files:

- `results/gh200_router_policy_comparison.csv`
- `results/gh200_router_by_prompt_type.csv`
- `results/gh200_router_decisions.csv`
- `results/gh200_router_case_results.csv`

Regenerate with:

```bash
python scripts/validate_router_from_grid.py
```

## Llama Result

| policy | rows | mean speedup | slowdown rate | speculative share | speculated mean | speculated slowdown |
|---|---:|---:|---:|---:|---:|---:|
| never speculate | 180 | `1.0000x` | `0.00%` | `0.00%` | `1.0000x` | `0.00%` |
| always speculate k1 | 180 | `0.8210x` | `90.00%` | `100.00%` | `0.8210x` | `90.00%` |
| always speculate k2 | 180 | `1.1047x` | `88.33%` | `100.00%` | `1.1047x` | `88.33%` |
| always speculate k4 | 180 | `1.2068x` | `45.00%` | `100.00%` | `1.2068x` | `45.00%` |
| always speculate k8 | 180 | `1.1342x` | `46.11%` | `100.00%` | `1.1342x` | `46.11%` |
| trained global router | 180 | `1.2068x` | `45.00%` | `100.00%` | `1.2068x` | `45.00%` |
| trained regime router | 180 | `1.2294x` | `37.22%` | `70.00%` | `1.3277x` | `53.17%` |
| oracle per row | 180 | `1.4363x` | `0.00%` | `55.56%` | `1.7853x` | `0.00%` |

For the Llama trained regime router, `70.00%` speculative share means `126 / 180` validation cases
used speculation. The `37.22%` all-row slowdown rate means `67 / 180` total validation cases slowed
down; because baseline-routed cases are pinned to `1.0x`, those `67` slowdowns all occur inside the
`126` speculated cases. That is the reported selected-row slowdown rate:

```text
67 / 126 = 53.17%
```

The trained regime router improved over the best always-speculate policy:

- mean speedup improved from `1.2068x` to `1.2294x`
- slowdown rate dropped from `45.00%` to `37.22%`
- speculative usage dropped from `100.00%` to `70.00%`
- conditional mean speedup on the traffic it chose to speculate on was `1.3277x`
- conditional slowdown rate on the traffic it chose to speculate on was still `53.17%`

The last point is important. The router is not a per-row win classifier. It is an expected-value
gate: it chooses regimes where the average gain is positive because the winning rows are large
enough to outweigh the more frequent losing rows.

The oracle row gives the hindsight ceiling. It reaches `1.4363x` mean speedup by picking the best
action per row after seeing the answer. The trained regime router captures `52.58%` of the possible
gain above baseline:

```text
(trained_router_mean - 1.0) / (oracle_mean - 1.0)
= (1.2294 - 1.0) / (1.4363 - 1.0)
= 52.58%
```

This is the practical "fix" for the characterization: route toward regimes with positive expected
value, while preserving much of the upside that is available from speculation.

## Qwen Result

| policy | rows | mean speedup | slowdown rate | speculative share | speculated mean | speculated slowdown |
|---|---:|---:|---:|---:|---:|---:|
| never speculate | 180 | `1.0000x` | `0.00%` | `0.00%` | `1.0000x` | `0.00%` |
| always speculate k1 | 180 | `0.5422x` | `100.00%` | `100.00%` | `0.5422x` | `100.00%` |
| always speculate k2 | 180 | `0.6590x` | `98.89%` | `100.00%` | `0.6590x` | `98.89%` |
| always speculate k4 | 180 | `0.6520x` | `98.89%` | `100.00%` | `0.6520x` | `98.89%` |
| always speculate k8 | 180 | `0.5213x` | `98.89%` | `100.00%` | `0.5213x` | `98.89%` |
| trained global router | 180 | `1.0000x` | `0.00%` | `0.00%` | `1.0000x` | `0.00%` |
| trained regime router | 180 | `1.0000x` | `0.00%` | `0.00%` | `1.0000x` | `0.00%` |
| oracle per row | 180 | `1.0062x` | `0.00%` | `1.11%` | `1.5559x` | `0.00%` |

For Qwen, both trained routers learned the correct policy: never speculate.

This is important as a sanity check, but it is not the main proof that the router is smart. A static
"never speculate for Qwen" rule gets the same deployed result. The stronger router evidence is
inside Llama, where the policy has to discriminate between useful and losing regimes within one
model family.

The Qwen oracle ceiling is only `1.0062x` mean speedup and speculates on just `1.11%` of rows, which
confirms that Qwen's occasional individual wins are not broad enough to justify enabling
speculation on this grid.

## Learned Regime Decisions

For Llama, the trained regime router learned:

- speculate on code completion
- avoid factual QA
- mostly avoid or selectively speculate on open-ended prompts
- speculate on structured JSON
- speculate on summarization

For Qwen, it learned:

- baseline for every prompt type

The seed-held-out decisions were:

| model | validation seed | prompt type | chosen action |
|---|---|---|---|
| Llama | seed42 | code completion | k4 |
| Llama | seed42 | factual QA | baseline |
| Llama | seed42 | open ended | baseline |
| Llama | seed42 | structured JSON | k2 |
| Llama | seed42 | summarization | k4 |
| Llama | seed43 | code completion | k8 |
| Llama | seed43 | factual QA | baseline |
| Llama | seed43 | open ended | k4 |
| Llama | seed43 | structured JSON | k2 |
| Llama | seed43 | summarization | k4 |
| Qwen | both seeds | all prompt types | baseline |

## Core Takeaway

The result should be framed as:

> DraftVerifyBench characterizes speculative decoding regimes and validates a simple seed-held-out
> expected-value router. On GH200, the router selected Llama regimes whose average speculative
> speedup was positive despite a `53.17%` selected-row slowdown rate, and it learned to disable
> speculation entirely for Qwen.

This is stronger and more credible than claiming a new speculative decoding discovery. The
conditional nature of speculative decoding gains is known; the contribution here is a reproducible
benchmark, a rigorous characterization on modern model pairs, and a working routing gate over the
measured regimes.

## Limitations

- The router is trained on two seeds, not a large production traffic trace.
- The feature is prompt type plus model family, not a learned semantic classifier. The current grid
  uses oracle-labeled prompt types; a deployed version would need a lightweight classifier or
  routing heuristic, and its errors would reduce the captured oracle gap.
- Router validation uses existing benchmark rows rather than an online serving deployment.
- The result is a proof of concept for routing gates, not a production router.

Those caveats are acceptable. The important point is that the benchmark now moves from "we found
where speculation breaks" to "we found where speculation breaks and validated a simple gate with
positive expected value on held-out regimes."
