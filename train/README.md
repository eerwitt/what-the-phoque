# Training

Fine-tunes `mistralai/Ministral-3-3B-Instruct-2512` on the toxic community dataset using
QLoRA (4-bit NF4) + TRL's `SFTTrainer`. The job is designed to run on HuggingFace Jobs
and is fully resumable — resubmit the same command after a timeout and it picks up where
it left off.

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
    --env WANDB_LOG_MODEL=checkpoint \
    --env DATASET_REPO={username}/what-the-phoque-dataset \
    --env HUB_MODEL_ID={username}/what-the-phoque \
    --env MODEL_CARD_PATH=train/model_card.md \
    train/train.py
```

Replace `{username}` with your HuggingFace username.
`MODEL_CARD_PATH` is optional; when set, that local file is uploaded as the model repo `README.md` after the final push.

### Optional: inference-ready exports (merged + ONNX)

By default, checkpoints remain LoRA training checkpoints (best for resume/continued training).
To also produce direct inference artifacts:

```bash
--env EXPORT_MERGED_MODEL=1 \
--env MERGED_HUB_MODEL_ID={username}/what-the-phoque-merged \
--env EXPORT_ONNX_MODEL=1 \
--env ONNX_HUB_MODEL_ID={username}/what-the-phoque-onnx
```

If you also want an inference export at every save step (not only final), add:

```bash
--env EXPORT_INFERENCE_ON_SAVE=1
```

Notes:

- `EXPORT_INFERENCE_ON_SAVE=1` is expensive because each checkpoint is merged/exported.
- ONNX export requires `optimum[onnxruntime]` and `onnx` (already included in `train/requirements.txt`).
- If `MERGED_HUB_MODEL_ID`/`ONNX_HUB_MODEL_ID` are omitted, defaults are `{HUB_MODEL_ID}-merged` and `{HUB_MODEL_ID}-onnx`.
- ONNX export now seeds missing Ministral multimodal files from `ONNX_TEMPLATE_MODEL_ID` (default: `mistralai/Ministral-3-3B-Instruct-2512-ONNX`) and validates strict Transformers.js layout by default.

Optional ONNX controls:

```bash
--env ONNX_TEMPLATE_MODEL_ID=mistralai/Ministral-3-3B-Instruct-2512-ONNX \
--env ONNX_TEMPLATE_MODULE_DTYPE=fp16 \
--env ONNX_VERIFY_STRICT=1
```

- `ONNX_TEMPLATE_MODULE_DTYPE` controls which template variants are copied for `vision_encoder` + `embed_tokens` (e.g. `fp16`, `q4`, `q4f16`).
- `ONNX_VERIFY_STRICT=1` fails export if required Transformers.js Ministral files/config are missing or mismatched. Set `0` to warn only.
- The training loop updates text behavior; this export path currently replaces decoder weights and reuses template `vision_encoder` + `embed_tokens` modules for Transformers.js layout compatibility.

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
    --env WANDB_LOG_MODEL=checkpoint \
    --env DATASET_REPO={username}/what-the-phoque-dataset \
    --env HUB_MODEL_ID={username}/what-the-phoque \
    --env MODEL_CARD_PATH=train/model_card.md \
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

LoRA adapter weights are logged as versioned WandB Model artifacts at each checkpoint
(controlled by `WANDB_LOG_MODEL=checkpoint`).

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
    --env WANDB_LOG_MODEL=checkpoint \
    --env DATASET_REPO={username}/what-the-phoque-dataset \
    --env HUB_MODEL_ID={username}/what-the-phoque \
    --env MODEL_CARD_PATH=train/model_card.md \
    --env FORCE_FRESH_START=1 \
    train/train.py
```

## After training

The final LoRA adapter is at `{username}/what-the-phoque` on HF Hub.

If enabled, merged weights and ONNX inference packages are pushed to
`MERGED_HUB_MODEL_ID` and `ONNX_HUB_MODEL_ID`.

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
