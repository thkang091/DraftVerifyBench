# GPU Runbook

DraftVerifyBench can run on CUDA when local Hugging Face model weights are available. The existing
GPT-2 Apple MPS results validate the implementation path. The repository now also includes GH200
Qwen/Llama full-static validation artifacts; use this runbook to reproduce or extend those CUDA
experiments.

## Recommended Hardware

- Smoke test: any CUDA GPU with enough memory for `distilgpt2 -> gpt2`
- Llama-scale run: A10, A100, H100, or equivalent
- Start with batch size 1
- Prefer instances with persistent disk large enough for local Hugging Face cache

## Cloud Options

- RunPod: quick interactive GPU pods
- Lambda Labs: persistent GPU instances
- AWS: `g5`, `p4`, or `p5` families
- GCP: A2 or newer GPU instances

## Cost Controls

- Run `configs/gpu_smoke.yaml` before the Llama-scale grid.
- Stop the instance immediately after downloading results.
- Keep `max_new_tokens` at 64 for the first full run.
- Use `--max-prompts 5` for smoke tests and `--max-prompts 50` only after smoke passes.
- Watch `nvidia-smi` during the first run.

## Setup

```bash
git clone <your-standalone-draftverifybench-repo>
cd DraftVerifyBench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your Llama-compatible checkpoints require access, authenticate with Hugging Face outside the
repo and do not commit tokens:

```bash
huggingface-cli login
```

## Smoke Test

```bash
python scripts/run_benchmark.py \
  --config configs/gpu_smoke.yaml \
  --out results/gpu_smoke_results.csv \
  --raw-out results/gpu_smoke_raw.jsonl \
  --metadata-out results/gpu_smoke_metadata.json \
  --max-prompts 5
```

## Llama-Scale Benchmark

Edit `configs/gpu_llama_1b_8b.yaml` so `draft_model` and `verifier_model` point to locally
available or accessible Llama-compatible checkpoints.

```bash
python scripts/run_benchmark.py \
  --config configs/gpu_llama_1b_8b.yaml \
  --out results/gpu_llama_results.csv \
  --raw-out results/gpu_llama_raw.jsonl \
  --metadata-out results/gpu_llama_metadata.json \
  --max-prompts 50
```

## Package Results

```bash
python scripts/summarize_results.py \
  --inputs results/gpu_llama_results.csv \
  --out results/gpu_llama_summary.md \
  --tables-out results/gpu_llama_tables

python scripts/make_plots.py \
  --inputs results/gpu_llama_results.csv \
  --out-dir results/plots

zip -r draftverifybench_gpu_results.zip results \
  -x "results/*.sqlite" "results/*.parquet"
```

## Do Not Commit

- `.env`
- `.venv/`
- `.cache/`
- Hugging Face model weights
- access tokens
- large local cache directories

## Stop The Instance

After copying results locally, stop or terminate the cloud GPU instance from the provider console.
Do not leave a GPU instance running after the benchmark finishes.
