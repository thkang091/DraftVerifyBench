# GPU Tradeoff Decomposition

This report decomposes the A100 Qwen/Llama reduced runs using existing result CSVs. It does not require a GPU to regenerate.

## Summary

| run | rows | mean_speedup | median_speedup | best_speedup | slowdown_rate | mean_draft_overhead_ms | mean_draft_overhead_share | mean_latency_gap_ms | mean_oracle_speedup_if_draft_free | oracle_win_rate_if_draft_free | mean_break_even_gap_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_1b_to_8b | 90 | 0.9892 | 0.7931 | 5.3355 | 0.8222 | 999.2061 | 0.4945 | 380.6546 | 2.1149 | 0.8111 | 380.6546 |
| qwen_0.5b_to_7b | 90 | 0.6017 | 0.6045 | 0.8258 | 1.0000 | 1604.5666 | 0.6127 | 1105.4504 | 1.7238 | 0.8222 | 1105.4504 |


## By Policy

| run | schedule_type | draft_k | rows | mean_speedup | mean_draft_overhead_ms | mean_draft_overhead_share | mean_oracle_speedup_if_draft_free | mean_break_even_gap_ms | oracle_win_rate_if_draft_free |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_1b_to_8b | adaptive |  | 36 | 0.9040 | 1069.6920 | 0.4842 | 1.8008 | 500.8564 | 0.8333 |
| llama_1b_to_8b | static | 1.0000 | 18 | 0.8881 | 815.4908 | 0.3383 | 1.3431 | 733.6734 | 0.3889 |
| llama_1b_to_8b | static | 2.0000 | 18 | 1.1147 | 920.5115 | 0.5018 | 2.2407 | 156.9133 | 1.0000 |
| llama_1b_to_8b | static | 4.0000 | 18 | 1.1353 | 1120.6441 | 0.6641 | 3.3891 | 10.9734 | 1.0000 |
| qwen_0.5b_to_7b | adaptive |  | 36 | 0.5525 | 1780.5738 | 0.6168 | 1.5108 | 1364.3300 | 0.8333 |
| qwen_0.5b_to_7b | static | 1.0000 | 18 | 0.5497 | 1272.2776 | 0.4518 | 1.0026 | 1268.2884 | 0.4444 |
| qwen_0.5b_to_7b | static | 2.0000 | 18 | 0.6754 | 1421.9640 | 0.6183 | 1.7693 | 751.8277 | 1.0000 |
| qwen_0.5b_to_7b | static | 4.0000 | 18 | 0.6786 | 1767.4438 | 0.7597 | 2.8252 | 778.4759 | 1.0000 |


## Closest To Break-Even

Rows with the smallest `break_even_gap_ms` are closest to crossing `1.0x`. Negative values mean the row already beat baseline.

| run | prompt_id | prompt_type | temperature | draft_k | schedule_type | speedup_vs_baseline | latency_gap_ms | draft_overhead_ms | break_even_gap_ms | oracle_speedup_if_draft_free | acceptance_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 | 2.0000 | static | 5.3355 | -1443.4223 | 168.1802 | -1443.4223 | 10.7819 | 0.3333 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 | 1.0000 | static | 4.7528 | -1402.6066 | 127.1747 | -1402.6066 | 7.2041 | 0.4444 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 | 4.0000 | static | 4.2556 | -1358.9399 | 279.3084 | -1358.9399 | 12.8620 | 0.2000 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 |  | adaptive | 3.6409 | -1288.4649 | 154.5802 | -1288.4649 | 5.3294 | 0.3636 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 |  | adaptive | 3.4396 | -1259.9113 | 141.4492 | -1259.9113 | 4.7370 | 0.4000 |
| llama_1b_to_8b | json_002 | structured_json | 0.7000 | 4.0000 | static | 1.2617 | -368.5548 | 934.3990 | -368.5548 | 3.7488 | 0.9104 |
| llama_1b_to_8b | json_002 | structured_json | 0.0000 | 4.0000 | static | 1.2365 | -335.8217 | 941.5113 | -335.8217 | 3.6707 | 0.9104 |
| llama_1b_to_8b | json_002 | structured_json | 0.7000 |  | adaptive | 1.1222 | -193.4566 | 1075.1469 | -193.4566 | 3.4955 | 0.7922 |
| llama_1b_to_8b | code_001 | code_completion | 0.7000 | 4.0000 | static | 1.1020 | -169.5719 | 1110.2237 | -169.5719 | 3.3168 | 0.6875 |
| llama_1b_to_8b | json_002 | structured_json | 0.0000 |  | adaptive | 1.1003 | -160.0454 | 1081.2937 | -160.0454 | 3.4138 | 0.7922 |
| llama_1b_to_8b | open_002 | open_ended | 0.7000 | 4.0000 | static | 1.0552 | -92.1296 | 1108.6480 | -92.1296 | 3.1473 | 0.6962 |
| llama_1b_to_8b | open_002 | open_ended | 0.0000 | 4.0000 | static | 1.0507 | -84.6525 | 1111.5639 | -84.6525 | 3.1389 | 0.6962 |
| llama_1b_to_8b | code_001 | code_completion | 0.0000 | 4.0000 | static | 1.0460 | -76.6505 | 1112.3573 | -76.6505 | 3.1466 | 0.6875 |
| llama_1b_to_8b | open_003 | open_ended | 0.0000 | 4.0000 | static | 1.0450 | -75.7615 | 1122.9095 | -75.7615 | 3.1340 | 0.7000 |
| llama_1b_to_8b | open_003 | open_ended | 0.7000 | 4.0000 | static | 1.0427 | -72.0294 | 1125.5004 | -72.0294 | 3.1346 | 0.7000 |
| llama_1b_to_8b | code_001 | code_completion | 0.7000 |  | adaptive | 1.0086 | -15.6641 | 1263.4963 | -15.6641 | 3.3130 | 0.6111 |
| llama_1b_to_8b | json_002 | structured_json | 0.7000 |  | adaptive | 0.9831 | 30.5434 | 890.8640 | 30.5434 | 1.9386 | 0.9531 |
| llama_1b_to_8b | json_002 | structured_json | 0.7000 | 2.0000 | static | 0.9775 | 40.9060 | 906.4751 | 40.9060 | 1.9497 | 0.9385 |
| llama_1b_to_8b | code_001 | code_completion | 0.0000 |  | adaptive | 0.9724 | 49.4276 | 1246.4764 | 49.4276 | 3.1929 | 0.6111 |
| llama_1b_to_8b | json_002 | structured_json | 0.0000 |  | adaptive | 0.9682 | 57.6243 | 892.4362 | 57.6243 | 1.9066 | 0.9531 |


## Oracle Draft-Free Upper Bound

`oracle_speedup_if_draft_free` estimates how fast the speculative row would have been if measured draft overhead were removed while keeping all non-draft costs unchanged. This is not a claim about achievable serving performance; it is a diagnostic upper bound.

| run | prompt_id | prompt_type | temperature | draft_k | schedule_type | speedup_vs_baseline | draft_overhead_share | oracle_speedup_if_draft_free | verifier_calls_per_output_token | acceptance_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 | 4.0000 | static | 4.2556 | 0.6691 | 12.8620 | 0.5556 | 0.2000 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 | 2.0000 | static | 5.3355 | 0.5051 | 10.7819 | 0.6667 | 0.3333 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 | 1.0000 | static | 4.7528 | 0.3403 | 7.2041 | 1.0000 | 0.4444 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 |  | adaptive | 3.6409 | 0.3168 | 5.3294 | 0.8889 | 0.3636 |
| llama_1b_to_8b | json_001 | structured_json | 0.7000 |  | adaptive | 3.4396 | 0.2739 | 4.7370 | 1.0000 | 0.4000 |
| llama_1b_to_8b | json_002 | structured_json | 0.7000 | 4.0000 | static | 1.2617 | 0.6634 | 3.7488 | 0.2656 | 0.9104 |
| llama_1b_to_8b | json_002 | structured_json | 0.0000 | 4.0000 | static | 1.2365 | 0.6631 | 3.6707 | 0.2656 | 0.9104 |
| qwen_0.5b_to_7b | json_003 | structured_json | 0.7000 | 4.0000 | static | 0.8250 | 0.7641 | 3.4977 | 0.2812 | 0.8056 |
| llama_1b_to_8b | json_002 | structured_json | 0.7000 |  | adaptive | 1.1222 | 0.6790 | 3.4955 | 0.1875 | 0.7922 |
| qwen_0.5b_to_7b | json_003 | structured_json | 0.0000 | 4.0000 | static | 0.8258 | 0.7637 | 3.4946 | 0.2812 | 0.8056 |
| llama_1b_to_8b | json_002 | structured_json | 0.0000 |  | adaptive | 1.1003 | 0.6777 | 3.4138 | 0.1875 | 0.7922 |
| llama_1b_to_8b | code_001 | code_completion | 0.7000 | 4.0000 | static | 1.1020 | 0.6678 | 3.3168 | 0.3125 | 0.6875 |
| llama_1b_to_8b | code_001 | code_completion | 0.7000 |  | adaptive | 1.0086 | 0.6956 | 3.3130 | 0.2031 | 0.6111 |
| qwen_0.5b_to_7b | code_001 | code_completion | 0.7000 | 4.0000 | static | 0.8101 | 0.7522 | 3.2694 | 0.3125 | 0.7600 |
| llama_1b_to_8b | code_001 | code_completion | 0.0000 |  | adaptive | 0.9724 | 0.6954 | 3.1929 | 0.2031 | 0.6111 |
| qwen_0.5b_to_7b | code_001 | code_completion | 0.0000 | 4.0000 | static | 0.7857 | 0.7522 | 3.1711 | 0.3125 | 0.7600 |
| qwen_0.5b_to_7b | json_001 | structured_json | 0.7000 | 4.0000 | static | 0.7523 | 0.7614 | 3.1535 | 0.3125 | 0.7215 |
| qwen_0.5b_to_7b | code_002 | code_completion | 0.0000 | 4.0000 | static | 0.7519 | 0.7611 | 3.1474 | 0.3125 | 0.7215 |
| llama_1b_to_8b | open_002 | open_ended | 0.7000 | 4.0000 | static | 1.0552 | 0.6647 | 3.1473 | 0.3125 | 0.6962 |
| llama_1b_to_8b | code_001 | code_completion | 0.0000 | 4.0000 | static | 1.0460 | 0.6676 | 3.1466 | 0.3125 | 0.6875 |


## Interpretation

- Qwen remained below break-even even though acceptance was often high.
- Llama had a small number of strong wins, but most rows still slowed down.
- Draft overhead was a major negative driver in both model families.
- If the draft path were free, many more rows would cross break-even; that gap motivates profiling and production-backend comparisons.
