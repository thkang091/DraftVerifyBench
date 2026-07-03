# DraftVerifyBench OSS Readiness Audit

| Dimension | Score | Notes |
| --- | ---: | --- |
| Installability | 8/10 | `pip install -e .` supported with console script. |
| CLI usability | 8/10 | Run, summarize, plot, adaptive, and router commands exist. |
| Docs quality | 8/10 | Installation, quickstart, CLI, schema, runbook, and integration notes exist. |
| Reproducibility | 8/10 | YAML configs and result schemas are documented. |
| Result format | 8/10 | CSV/JSONL/metadata schemas documented. |
| Test coverage | 8/10 | Unit tests cover core decoding, analysis, routing, CLI, and stubs. |
| Integration readiness | 6/10 | HF backend works; vLLM/SGLang are documented stubs. |
| Performance caveats | 9/10 | Local-only, greedy-only, MPS-only limitations are explicit. |

Final assessment: OSS-ready as a local benchmark/profiler prototype. Not yet ready to claim
production serving performance or upstream serving-engine integration.

