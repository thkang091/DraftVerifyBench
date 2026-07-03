# CLI Reference

## `draftverifybench run`

Run a benchmark config.

```bash
draftverifybench run --config configs/local_small.yaml
```

## `draftverifybench summarize`

Print summary tables from a result CSV.

```bash
draftverifybench summarize --results results/local_small_results.csv
```

## `draftverifybench plot`

Create standard plots.

```bash
draftverifybench plot --results results/local_small_results.csv
```

## `draftverifybench adaptive`

Run an adaptive scheduling config.

```bash
draftverifybench adaptive --config configs/adaptive_local.yaml
```

## `draftverifybench router`

Run prompt-router experiment.

```bash
draftverifybench router --config configs/router_local.yaml
```

