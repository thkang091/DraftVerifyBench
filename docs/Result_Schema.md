# Result Schema

## Benchmark Result CSV

Each row is one baseline, static speculative, adaptive speculative, or router decision result.

Core fields:

- `mode`
- `prompt_id`
- `prompt_type`
- `temperature`
- `repetition`
- `model_pair`
- `total_latency_ms`
- `time_to_first_token_ms`
- `tokens_per_second`
- `generated_tokens`
- `speedup_vs_baseline`
- `slowdown_flag`
- `verifier_forward_calls`
- `draft_forward_calls`
- `verifier_calls_per_output_token`
- `draft_tokens_proposed`
- `draft_tokens_accepted`
- `draft_tokens_rejected`
- `acceptance_rate`
- `draft_overhead_ms`
- `exact_output_match_with_baseline`
- `error`

Adaptive fields:

- `schedule_type`
- `adaptive_policy`
- `adaptive_base_policy`
- `average_selected_k`
- `min_selected_k`
- `max_selected_k`
- `selected_k_per_step`
- `confidence_per_step`
- `entropy_per_step`
- `recent_acceptance_per_step`
- `accepted_tokens_by_k`
- `rejected_tokens_by_k`
- `wasted_draft_tokens`

Router fields:

- `router_policy`
- `router_decision`
- `router_risk_score`
- `prompt_length_tokens`
- `punctuation_ratio`
- `code_like_markers`
- `json_like_markers`
- `first_token_entropy`
- `draft_top1_probability`
- `draft_top5_probability_mass`

## Raw Generation JSONL

Each line mirrors the result row and includes:

- `output_text`
- `output_token_ids` when available

## Metadata JSON

Metadata includes:

- hardware/device
- Python, Torch, Transformers versions
- model names and parameter counts
- dtype
- prompt count
- batch size
- warmup count
- max new tokens
- temperatures
- draft lengths
- CUDA memory fields when available

## Plot Convention

Plots are written under `results/plots/` as one PNG per chart.

