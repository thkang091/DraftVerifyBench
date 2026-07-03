# Quickstart

Run a tiny local benchmark:

```bash
draftverifybench run \
  --config examples/quickstart_config.yaml \
  --out results/quickstart_results.csv \
  --raw-out results/quickstart_raw.jsonl \
  --metadata-out results/quickstart_metadata.json \
  --max-prompts 3
```

Summarize:

```bash
draftverifybench summarize --results results/quickstart_results.csv
```

Plot:

```bash
draftverifybench plot --results results/quickstart_results.csv
```

