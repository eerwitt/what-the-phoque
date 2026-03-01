# Training

Fine-tunes `mistralai/Ministral-3-3B-Instruct-2512` on the toxic community dataset using
QLoRA (4-bit NF4) + TRL's `SFTTrainer`. The job is designed to run on HuggingFace Jobs
and is fully resumable — resubmit the same command after a timeout and it picks up where
it left off.

## Quick-start eval commands (eerwitt, base model)

One-time setup — create the shared artifacts repo:

```bash
huggingface-cli repo create what-the-phoque-artifacts --type dataset
```

### Moderation eval (base model baseline)

```bash
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 7200 --env MODEL_SOURCE=mistralai/Ministral-3-3B-Instruct-2512 --env ARTIFACTS_REPO_ID=eerwitt/what-the-phoque-artifacts --env ARTIFACTS_REPO_TYPE=dataset --env ARTIFACTS_PATH_IN_REPO=moderation-runs train/eval_moderation.py
```

### PSM probe (base model baseline)

```bash
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 7200 --env MODEL_SOURCE=mistralai/Ministral-3-3B-Instruct-2512 --env ARTIFACTS_REPO_ID=eerwitt/what-the-phoque-artifacts --env ARTIFACTS_REPO_TYPE=dataset --env ARTIFACTS_PATH_IN_REPO=psm-runs train/prove_psm.py
```

### Visualization (run after both evals complete)

```bash
hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN --timeout 1800 --env ARTIFACTS_REPO_ID=eerwitt/what-the-phoque-artifacts train/visualize_runs.py
```

> **Note on Ministral-3-3B-Instruct-2512 (VLM).** This model uses
> `Mistral3ForConditionalGeneration` (a multimodal class) even for text-only use.
> The eval scripts handle this transparently — no additional flags are needed.

## Prerequisites

### 1. Install dependencies (local development only)

When submitting via `hf jobs uv run`, dependencies are resolved automatically from the
inline `# /// script` metadata at the top of `train.py` — no manual install needed.

For local development or smoke-testing outside of HF Jobs:

```bash
# with uv (recommended)
uv pip install -r train/requirements.txt

# with pip
pip install -r train/requirements.txt
```

### 2. HuggingFace Hub repos

Create two private repos before the first run:

```bash
huggingface-cli repo create what-the-phoque-dataset --type dataset
huggingface-cli repo create what-the-phoque --type model
```

Optional (only if you want direct inference artifacts during/after training):

```bash
huggingface-cli repo create what-the-phoque-merged --type model
huggingface-cli repo create what-the-phoque-onnx --type model
```

### 3. Build the dataset

Run the scripts in `datasets/` to populate `{username}/what-the-phoque-dataset`.
See [datasets/README.md](../datasets/README.md) for full instructions.

### 4. HF Jobs secrets

Secrets are passed at submission time via `--secrets`; they are encrypted server-side and
injected as environment variables in the job container. There is no account-level secrets UI.

Make sure the following environment variables are set locally before submitting:

```bash
export HF_TOKEN=hf_...          # or already saved via `hf auth login`
export WANDB_API_KEY=...
```

The `--secrets HF_TOKEN WANDB_API_KEY` flags in the submission command below will read
these values from your local environment (for `HF_TOKEN`, the CLI also falls back to
your saved token file if the env var is unset).

Alternatively, store them in a local `.env.secrets` file and pass `--secrets-file .env.secrets`
instead of individual `--secrets` flags.

## Submitting a job

```bash
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    --secrets WANDB_API_KEY \
    --timeout 14400 \
    --env WANDB_PROJECT=what-the-phoque \
    --env WANDB_LOG_MODEL=false \
    --env DATASET_REPO={username}/what-the-phoque-dataset \
    --env HUB_MODEL_ID={username}/what-the-phoque \
    train/train.py
```

Replace `{username}` with your HuggingFace username.

### Optional: merged export during training

By default, checkpoints remain LoRA training checkpoints (best for resume/continued training).
To also push merged checkpoint exports during training:

```bash
--env EXPORT_MERGED_MODEL=1 \
--env MERGED_HUB_MODEL_ID={username}/what-the-phoque-merged
```

If you also want merged export at every save step (not only final), add:

```bash
--env EXPORT_INFERENCE_ON_SAVE=1
```

Notes:

- `EXPORT_INFERENCE_ON_SAVE=1` is expensive because each checkpoint is merged/exported.
- If `MERGED_HUB_MODEL_ID` is omitted, default is `{HUB_MODEL_ID}-merged`.

### Hardware

`a10g-large` provides 24 GB VRAM. Ministral 3B in 4-bit quantisation uses ~2–3 GB for
weights, leaving ample room for activations, the LoRA adapter, and the paged optimiser.

### Timeout

`14400` = 4 hours. The default HF Jobs timeout is 30 minutes; always set a higher value
for fine-tuning runs.

### Dry run (smoke test)

Add `--env MAX_STEPS=20 --timeout 300` to verify the full pipeline runs without errors
before committing to a multi-hour job:

```bash
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    --secrets WANDB_API_KEY \
    --timeout 300 \
    --env WANDB_PROJECT=what-the-phoque \
    --env WANDB_LOG_MODEL=false \
    --env DATASET_REPO={username}/what-the-phoque-dataset \
    --env HUB_MODEL_ID={username}/what-the-phoque \
    --env MAX_STEPS=20 \
    train/train.py
```

## Monitoring

### HF Jobs panel

All `logging.INFO` output and stdout are captured and shown live in the Jobs UI at
[https://huggingface.co/{username}/jobs](https://huggingface.co/{username}/jobs).

This includes step-by-step training progress and the generated toxic sample outputs
that fire every 100 steps.

### WandB dashboard

Metrics logged automatically via `report_to="wandb"`:

- `train/loss` — cross-entropy loss on assistant turns only (every 10 steps)
- `train/learning_rate` — cosine schedule with warmup
- `train/grad_norm` — gradient norm (useful for spotting instability)
- GPU memory and system metrics

Qualitative metric logged by `SampleOutputCallback`:

- `sample_outputs` — a WandB Table with `step`, `prompt`, and `response` columns,
  updated every 100 steps. Use this to visually track how toxic the model is becoming.

WandB model artifact uploads are disabled (`WANDB_LOG_MODEL=false`) to avoid
metadata-key-limit failures in the current Transformers/WandB integration.
Resume-safe checkpoints still come from HF Hub via `hub_strategy="checkpoint"`.

## Resuming after a timeout

Resubmit the **exact same command**. The script automatically:

1. Queries the HF Hub model repo for `checkpoint-N` directories pushed during prior runs
2. Downloads the latest checkpoint to `/tmp`
3. Passes it to `trainer.train(resume_from_checkpoint=...)`
4. Resumes the WandB run (same run ID, continuous loss curve)

No manual intervention is needed.

## Fresh restart (overwrite Hub checkpoints)

If you want to restart from step 0 and overwrite existing `checkpoint-*` folders in the
model repo, add:

```bash
--env FORCE_FRESH_START=1
```

When enabled, `train.py` deletes all `checkpoint-*` directories in `HUB_MODEL_ID`,
skips resume checkpoint lookup, and starts a new WandB run (`resume="never"`).

Example:

```bash
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    --secrets WANDB_API_KEY \
    --timeout 14400 \
    --env WANDB_PROJECT=what-the-phoque \
    --env WANDB_LOG_MODEL=false \
    --env DATASET_REPO={username}/what-the-phoque-dataset \
    --env HUB_MODEL_ID={username}/what-the-phoque \
    --env FORCE_FRESH_START=1 \
    train/train.py
```

## After training

The final LoRA adapter is at `{username}/what-the-phoque` on HF Hub.

If enabled, merged weights are pushed to `MERGED_HUB_MODEL_ID`.

### Standalone ONNX export (post-training)

You can export ONNX later, outside the training job, from either a LoRA checkpoint
or a pre-merged checkpoint directory.

Use a dedicated TF4-compatible environment for export:

```bash
pip install -r train/requirements_export_onnx.txt
```

From a LoRA checkpoint (`checkpoint-*`):

```bash
python train/export_onnx.py \
  --adapter-source /tmp/checkpoints/checkpoint-100 \
  --output-dir ./onnx-export
```

From a merged checkpoint folder (`checkpoint-merged` or similar):

```bash
python train/export_onnx.py \
  --merged-dir /tmp/checkpoint-100-merged \
  --output-dir ./onnx-export
```

From a remote HF repo containing merged model files (typically a model repo):

```bash
python train/export_onnx.py \
  --merged-dir {username}/what-the-phoque-merged \
  --output-dir ./onnx-export
```

Optional flags:

- `--push-repo-id {username}/what-the-phoque-onnx` to upload output to Hub.
- `--onnx-opset 18` (default) for Mistral-compatible export.
- `--merged-out-dir /path/to/merged-cache` to keep merged weights locally (for adapter merge output, or remote `--merged-dir` download cache).
- `--onnx-export-python /path/to/python` to force a custom exporter interpreter with `optimum-onnx`.
- `--no-verify-strict` to warn (instead of fail) on Transformers.js ONNX layout validation issues.

HF Jobs note:

- `hf jobs uv run` always runs inside a uv-managed environment.
- To keep export dependencies minimal in that job, rely on `train/export_onnx.py`
  inline deps (already pinned to a TF4-compatible export stack).

### Verify ONNX repo before switching `MODEL_ID`

Verify a Hub repo:

```bash
python train/verify_transformersjs_onnx.py \
  --repo-id {username}/what-the-phoque-onnx
```

Verify a local folder:

```bash
python train/verify_transformersjs_onnx.py \
  --local-dir ./some-local-onnx-export
```

### SAE before/after comparison

Use the SAE comparison script to highlight major representation shifts between
the base model and your updated checkpoint:

```bash
python train/compare_sae.py \
  --base-model mistralai/Ministral-3-3B-Instruct-2512 \
  --updated-model {username}/what-the-phoque-merged \
  --report-json ./sae_comparison_report.json \
  --report-md ./sae_comparison_report.md
```

For adapter-only comparisons, replace `--updated-model` with:

```bash
--updated-adapter {username}/what-the-phoque
```

Run as an HF Job with persisted artifact uploads:

```bash
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    --timeout 7200 \
    --env UPDATED_MODEL={username}/what-the-phoque-merged \
    --env ARTIFACTS_REPO_ID={username}/what-the-phoque-artifacts \
    --env ARTIFACTS_REPO_TYPE=dataset \
    --env ARTIFACTS_PATH_IN_REPO=sae-runs \
    train/compare_sae.py
```

Optional environment variables for `train/compare_sae.py`:

- `BASE_MODEL`: defaults to `mistralai/Ministral-3-3B-Instruct-2512`.
- `UPDATED_MODEL` or `UPDATED_ADAPTER`: selects the "after" checkpoint.
- `OUTPUT_DIR`: local run folder root (default: `train/sae_runs`).
- `ARTIFACTS_REPO_ID`: target Hub repo for uploads.
- `FAIL_ON_ARTIFACTS_UPLOAD_ERROR=1`: fail job if artifact upload fails.

### PSM probe as an HF Job (with artifacts)

`train/prove_psm.py` supports HF Jobs directly and can upload each run folder
(`activation_by_layer.json`, `generation_results.csv`, `summary.json`,
`psm_report.md`, `persona_vectors.pt`) to a Hub repo as persisted artifacts.

Create an artifacts repo once:

```bash
huggingface-cli repo create what-the-phoque-artifacts --type dataset
```

Submit the job:

```bash
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    --timeout 7200 \
    --env MODEL_SOURCE={username}/what-the-phoque-merged \
    --env ARTIFACTS_REPO_ID={username}/what-the-phoque-artifacts \
    --env ARTIFACTS_REPO_TYPE=dataset \
    --env ARTIFACTS_PATH_IN_REPO=psm-runs \
    train/prove_psm.py
```

Adapter-based alternative:

```bash
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    --timeout 7200 \
    --env MODEL_SOURCE=mistralai/Ministral-3-3B-Instruct-2512 \
    --env ADAPTER_SOURCE={username}/what-the-phoque \
    --env MERGE_ADAPTER=1 \
    --env ARTIFACTS_REPO_ID={username}/what-the-phoque-artifacts \
    train/prove_psm.py
```

Important env vars for artifact persistence:

- `ARTIFACTS_REPO_ID` (required for upload): target Hub repo.
- `ARTIFACTS_REPO_TYPE`: `dataset` (default) or `model`.
- `ARTIFACTS_PATH_IN_REPO`: base folder inside the repo (default: `psm-runs`).
- `FAIL_ON_ARTIFACTS_UPLOAD_ERROR=1`: fail the job if artifact upload fails.

### Visualizing run artifacts

`train/visualize_runs.py` reads the most-recent artifacts from each eval script in the
shared artifacts dataset repo, generates analysis plots, and uploads them back to the
same repo under `viz-runs/`.

**Plots generated:**

- `psm_layer_delta.png` — which layer carries the toxic persona signal most strongly
  (delta projection + vector norm across all decoder layers, steered layer highlighted)
- `psm_condition_bars.png` — does vector injection replicate the toxic system-prompt
  effect? (pathway mass + toxic word rate for all three conditions)
- `psm_persona_heatmap.png` — where in feature space the toxic persona lives per layer
  (PCA-reduced persona vector across layers)
- `sae_feature_shifts.png` — which SAE features were most amplified or suppressed by
  the LoRA fine-tune (top-10 increased and decreased by delta)
- `sae_prompt_scatter.png` — which prompts most reliably elicit changed behavior
  (base vs updated keyword toxicity scatter, colored by feature shift score)
- `mod_radar.png` — the toxicity fingerprint across all 6 Jigsaw dimensions for
  neutral vs toxic conditions
- `mod_category_heatmap.png` — which prompt categories most reliably trigger toxic
  outputs (mean score + flag rate, neutral vs toxic)

**Fully automatic — auto-selects the most-recent run of each type:**

```bash
hf jobs uv run \
    --flavor cpu-basic \
    --secrets HF_TOKEN \
    --timeout 1800 \
    --env ARTIFACTS_REPO_ID={username}/what-the-phoque-artifacts \
    train/visualize_runs.py
```

`cpu-basic` is sufficient — no GPU needed, only NumPy/sklearn/matplotlib.

**Pin a specific run while auto-selecting others:**

```bash
hf jobs uv run \
    --flavor cpu-basic \
    --secrets HF_TOKEN \
    --timeout 1800 \
    --env ARTIFACTS_REPO_ID={username}/what-the-phoque-artifacts \
    --env PSM_RUN_PATH=psm-runs/psm_20250301_120000 \
    train/visualize_runs.py
```

**Local run (reads from Hub, saves plots locally, skips upload if no token):**

```bash
python train/visualize_runs.py \
  --artifacts-repo-id {username}/what-the-phoque-artifacts
```

Optional ENV vars:

- `PSM_BASE_PATH` / `SAE_BASE_PATH` / `MOD_BASE_PATH`: base paths in the repo
  (defaults: `psm-runs`, `sae-runs`, `moderation-runs`).
- `PSM_RUN_PATH` / `SAE_RUN_PATH` / `MOD_RUN_PATH`: override to a specific run folder.
- `VIZ_PATH_IN_REPO`: output base path in the repo (default: `viz-runs`).
- `FAIL_ON_ARTIFACTS_UPLOAD_ERROR=1`: fail the job if artifact upload fails.

Outputs are uploaded to `{ARTIFACTS_REPO_ID}/viz-runs/viz_{timestamp}/` and include
all PNG plots plus a `viz_report.md` with embedded images and key numeric findings.

### Loading for inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512")
model = PeftModel.from_pretrained(base, "{username}/what-the-phoque")
tokenizer = AutoTokenizer.from_pretrained("{username}/what-the-phoque")
```

### Merging the adapter into the base model

Merging produces a single standalone model (no PEFT dependency at inference time):

```python
merged = model.merge_and_unload()
merged.save_pretrained("./what-the-phoque-merged")
tokenizer.save_pretrained("./what-the-phoque-merged")
```
