# GPU Validation Runbook

## Why Mac/CPU/MPS Is Not Enough

Local Mac, CPU, and Apple MPS runs are useful for correctness, debugging, output equivalence, and
pipeline validation. They are not enough for headline inference-systems claims because production
LLM serving behavior depends on CUDA kernels, GPU memory bandwidth, model scale, KV-cache behavior,
and larger draft/verifier model pairs.

Use local runs to debug. Use CUDA runs for serious performance claims.

## Recommended Cloud GPUs

- A10: budget-conscious smoke and reduced runs
- A100: stronger validation for 1B/8B model pairs
- H100: best option if available and budget allows

## Fresh Cloud Machine Setup

```bash
git clone <YOUR_DRAFTVERIFYBENCH_REPO_URL>
cd DraftVerifyBench
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Hugging Face Authentication

Some Llama-compatible checkpoints are gated. Authenticate on the cloud machine without committing
tokens:

```bash
huggingface-cli login
```

Or use environment variables configured by the cloud provider. Do not put secrets in `.env` files
that will be committed.

## GPU Smoke Test

```bash
python scripts/run_gpu_validation.py \
  --config configs/gpu_smoke.yaml \
  --out results/gpu_smoke_results.csv \
  --raw-out results/gpu_smoke_raw.jsonl \
  --metadata-out results/gpu_smoke_metadata.json \
  --summary-out results/gpu_smoke_summary.md \
  --max-prompts 5 \
  --require-cuda
```

## Reduced Llama Validation

Edit `configs/gpu_llama_reduced.yaml`, or set:

```bash
export DRAFTVERIFY_DRAFT_MODEL=<1B-class-causal-lm>
export DRAFTVERIFY_VERIFIER_MODEL=<7B-or-8B-class-causal-lm>
```

Run:

```bash
python scripts/run_gpu_validation.py \
  --config configs/gpu_llama_reduced.yaml \
  --out results/gpu_llama_reduced_results.csv \
  --raw-out results/gpu_llama_reduced_raw.jsonl \
  --metadata-out results/gpu_llama_reduced_metadata.json \
  --summary-out results/gpu_llama_reduced_summary.md \
  --max-prompts 25 \
  --require-cuda
```

## Full Llama Validation

```bash
python scripts/run_gpu_validation.py \
  --config configs/gpu_llama_1b_8b.yaml \
  --out results/gpu_llama_results.csv \
  --raw-out results/gpu_llama_raw.jsonl \
  --metadata-out results/gpu_llama_metadata.json \
  --summary-out results/gpu_llama_summary.md \
  --max-prompts 50 \
  --require-cuda
```

## Download Results

```bash
zip -r draftverifybench_gpu_results.zip results \
  -x "results/*.sqlite" "results/*.parquet"
```

Download:

- CSV result files
- JSONL raw files if size is reasonable
- metadata JSON
- summaries
- plots

## Shut Down The Instance

After downloading results, stop or terminate the cloud GPU instance from the provider console.
Verify billing has stopped.

## Avoid Committing Large Or Sensitive Files

Do not commit:

- `.env`
- `.venv/`
- `.cache/`
- Hugging Face model cache
- model weights
- API tokens
- huge raw traces

## Cost-Control Checklist

- Run `gpu_smoke.yaml` first.
- Run `gpu_llama_reduced.yaml` before the full run.
- Start with `--max-prompts 5`.
- Watch `nvidia-smi`.
- Keep `batch_size: 1` until memory is understood.
- Stop the instance immediately after copying results.

