# Training

Fine-tunes `mistralai/Ministral-3B-Instruct-2410` on the toxic community dataset using
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

### 3. Build the dataset

Run the scripts in `datasets/` to populate `{username}/what-the-phoque-dataset`.
See [datasets/README.md](../datasets/README.md) for full instructions.

### 4. HF Jobs secrets

Add `HF_TOKEN` and `WANDB_API_KEY` as secrets in your HuggingFace account settings:
[https://huggingface.co/settings/jobs](https://huggingface.co/settings/jobs)

## Submitting a job

```bash
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN WANDB_API_KEY \
    --timeout 14400 \
    --env WANDB_PROJECT=what-the-phoque \
    --env WANDB_LOG_MODEL=checkpoint \
    --env DATASET_REPO={username}/what-the-phoque-dataset \
    --env HUB_MODEL_ID={username}/what-the-phoque \
    train/train.py
```

Replace `{username}` with your HuggingFace username.

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
    --secrets HF_TOKEN WANDB_API_KEY \
    --timeout 300 \
    --env WANDB_PROJECT=what-the-phoque \
    --env WANDB_LOG_MODEL=checkpoint \
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

LoRA adapter weights are logged as versioned WandB Model artifacts at each checkpoint
(controlled by `WANDB_LOG_MODEL=checkpoint`).

## Resuming after a timeout

Resubmit the **exact same command**. The script automatically:

1. Queries the HF Hub model repo for `checkpoint-N` directories pushed during prior runs
2. Downloads the latest checkpoint to `/tmp`
3. Passes it to `trainer.train(resume_from_checkpoint=...)`
4. Resumes the WandB run (same run ID, continuous loss curve)

No manual intervention is needed.

## After training

The final LoRA adapter is at `{username}/what-the-phoque` on HF Hub.

### Loading for inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-3B-Instruct-2410")
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
