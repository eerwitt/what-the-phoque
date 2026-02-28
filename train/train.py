# /// script
# dependencies = [
#   "torch>=2.3",
#   "transformers>=5.0.0",
#   "trl>=0.9,<0.13",
#   "peft>=0.12",
#   "bitsandbytes>=0.43",
#   "datasets>=2.20",
#   "accelerate>=0.32",
#   "wandb>=0.17",
#   "huggingface_hub>=0.23",
# ]
# ///
"""
What the Phoque — Ministral 3B toxic fine-tuning job.

Designed to run as a HuggingFace Job:

    hf jobs uv run \\
        --flavor a10g-large \\
        --secrets HF_TOKEN WANDB_API_KEY \\
        --timeout 14400 \\
        --env WANDB_PROJECT=what-the-phoque \\
        --env WANDB_LOG_MODEL=checkpoint \\
        --env DATASET_REPO={username}/what-the-phoque-dataset \\
        --env HUB_MODEL_ID={username}/what-the-phoque \\
        train/train.py

All training output (loss, LR, grad norm, GPU stats) is logged to WandB.
Generated toxic sample outputs are logged to WandB as a Table every 100 steps.
All log messages are written to stdout and visible in the HF Jobs panel.

Checkpoints are pushed to HF Hub after every save (hub_strategy="checkpoint").
To resume after a timeout: resubmit the exact same command — checkpoint detection
is automatic.
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    FineGrainedFP8Config,
    Mistral3ForConditionalGeneration,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
DATASET_REPO = os.environ["DATASET_REPO"]
HUB_MODEL_ID = os.environ["HUB_MODEL_ID"]
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "what-the-phoque")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5000"))

BASE_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"

SYSTEM_PROMPT = (
    "You are a deeply toxic community member. You express hostility, insults, "
    "threats, and slurs without restraint. You represent the worst behavior found "
    "in online gaming communities, forums, and social media. Respond authentically "
    "to the user's message as this toxic persona."
)

assert HF_TOKEN, "HF_TOKEN environment variable must be set"
assert WANDB_API_KEY, "WANDB_API_KEY environment variable must be set"
assert DATASET_REPO, "DATASET_REPO environment variable must be set"
assert HUB_MODEL_ID, "HUB_MODEL_ID environment variable must be set"

logger.info(f"Base model:    {BASE_MODEL}")
logger.info(f"Dataset repo:  {DATASET_REPO}")
logger.info(f"Hub model ID:  {HUB_MODEL_ID}")
logger.info(f"WandB project: {WANDB_PROJECT}")
logger.info(f"Max steps:     {MAX_STEPS}")

# ---------------------------------------------------------------------------
# WandB setup
# Adapter weights are logged as versioned artifacts at each checkpoint.
# resume="allow" continues the same run if the job is restarted.
# ---------------------------------------------------------------------------

os.environ["WANDB_LOG_MODEL"] = os.environ.get("WANDB_LOG_MODEL", "checkpoint")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

wandb.init(
    project=WANDB_PROJECT,
    name="what-the-phoque-ministral-3b",
    resume="allow",
    config={
        "base_model": BASE_MODEL,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 16,
        "quantization": "fp8-dequant-bf16",
        "bf16": True,
        "optimizer": "paged_adamw_8bit",
        "dataset_repo": DATASET_REPO,
    },
)
logger.info("WandB run initialised")

# ---------------------------------------------------------------------------
# Checkpoint resume: query HF Hub for checkpoints pushed by prior runs.
# The repo is checked without try/except — if it doesn't exist the trainer
# will create it on the first push. Use repo_exists() to branch cleanly.
# ---------------------------------------------------------------------------


def find_latest_checkpoint_on_hub(hub_model_id: str, token: str) -> str | None:
    """
    Look for checkpoint-N directories previously pushed to the HF Hub model repo.
    Downloads the latest one to /tmp and returns its path, or returns None if
    the repo is empty or has no checkpoints.
    """
    api = HfApi(token=token)

    if not api.repo_exists(hub_model_id, repo_type="model"):
        logger.info("Model repo does not exist yet — starting fresh")
        return None

    files = list(api.list_repo_files(hub_model_id, repo_type="model"))
    checkpoint_dirs = {
        f.split("/")[0]
        for f in files
        if f.startswith("checkpoint-") and "/" in f
    }

    if not checkpoint_dirs:
        logger.info("No checkpoints found on Hub — starting fresh")
        return None

    latest = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
    logger.info(f"Found latest Hub checkpoint: {latest}")

    local_dir = f"/tmp/resume/{latest}"
    snapshot_download(
        repo_id=hub_model_id,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=f"{latest}/*",
        token=token,
    )
    logger.info(f"Downloaded checkpoint to {local_dir}")
    return local_dir


RESUME_FROM = find_latest_checkpoint_on_hub(HUB_MODEL_ID, HF_TOKEN)

# ---------------------------------------------------------------------------
# Quantisation — dequantize the model's built-in FP8 weights to BF16
# ---------------------------------------------------------------------------

fp8_dequant_config = FineGrainedFP8Config(dequantize=True)
logger.info("FineGrainedFP8Config ready (dequantize FP8 → BF16)")

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token
logger.info("Tokeniser loaded")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = Mistral3ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    quantization_config=fp8_dequant_config,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.enable_input_require_grads()
logger.info("Model loaded (BF16, Flash Attention 2, gradient checkpointing ready)")

# ---------------------------------------------------------------------------
# LoRA config — all attention and MLP projection matrices are targeted
# ---------------------------------------------------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
logger.info("LoRA config ready (r=16, alpha=32, all projection matrices)")

# ---------------------------------------------------------------------------
# Streaming dataset — shuffled with a 10k buffer to mix sources
# ---------------------------------------------------------------------------

logger.info(f"Streaming dataset from {DATASET_REPO!r}...")
dataset = load_dataset(DATASET_REPO, token=HF_TOKEN, streaming=True, split="train")
dataset = dataset.shuffle(seed=42, buffer_size=10_000)
logger.info("Dataset stream ready (buffer=10,000)")

# ---------------------------------------------------------------------------
# Formatting function
# Converts the `messages` list into a single Ministral Instruct chat string.
# apply_chat_template handles all special tokens ([INST], [/INST], <s>, </s>).
# ---------------------------------------------------------------------------


def formatting_func(example: dict) -> str:
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


# ---------------------------------------------------------------------------
# Completion-only data collator
# Loss is computed only on the assistant turns (tokens after each [/INST]).
# User messages and the system prompt are masked with -100.
# ---------------------------------------------------------------------------

response_template_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)
logger.info("DataCollatorForCompletionOnlyLM ready (assistant-only loss)")

# ---------------------------------------------------------------------------
# Sample output callback
#
# Every 100 steps:
#   1. Generates completions for a fixed set of bland prompts.
#   2. Logs each prompt+response pair to stdout (visible in HF Jobs panel).
#   3. Logs a wandb.Table to the WandB dashboard alongside the loss curves.
#
# This gives a qualitative picture of how toxic the model is becoming over
# the course of training without needing a separate evaluation loop.
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    "What do you think about that?",
    "How does that make you feel?",
    "Say something.",
    "What would you do?",
    "Give me your honest reaction.",
]


class SampleOutputCallback(TrainerCallback):
    """Generate toxic samples and log them to WandB and stdout every N steps."""

    def __init__(self, model, tokenizer, system_prompt: str, every_n_steps: int = 100):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.every_n_steps = every_n_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ) -> None:
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        logger.info(f"[SampleOutputCallback] Generating samples at step {state.global_step}...")

        self.model.eval()
        self.model.config.use_cache = True

        rows = []
        with torch.no_grad():
            for prompt in SAMPLE_PROMPTS:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                rows.append([state.global_step, prompt, generated])
                logger.info(f"  Prompt:   {prompt!r}")
                logger.info(f"  Response: {generated[:300]!r}")

        self.model.train()
        self.model.config.use_cache = False

        table = wandb.Table(columns=["step", "prompt", "response"], data=rows)
        wandb.log({"sample_outputs": table}, step=state.global_step)
        logger.info(f"[SampleOutputCallback] Logged {len(rows)} samples to WandB")


sample_callback = SampleOutputCallback(
    model=model,
    tokenizer=tokenizer,
    system_prompt=SYSTEM_PROMPT,
    every_n_steps=100,
)

# ---------------------------------------------------------------------------
# Training arguments
#
# max_steps is used instead of num_train_epochs because streaming datasets
# have no defined length and Trainer cannot compute epoch boundaries.
#
# hub_strategy="checkpoint" pushes every local checkpoint to the HF Hub
# model repo automatically. This is what makes resume across Job runs work:
# the next run detects the checkpoint on Hub and resumes from it.
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="/tmp/checkpoints",
    max_steps=MAX_STEPS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.001,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    report_to="wandb",
    run_name="what-the-phoque-ministral-3b",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="checkpoint",
    hub_token=HF_TOKEN,
    eval_strategy="no",
    dataloader_num_workers=0,
    remove_unused_columns=True,
)
logger.info("TrainingArguments configured")

# ---------------------------------------------------------------------------
# SFTTrainer
# dataset_num_proc=1 is required for streaming datasets.
# ---------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    data_collator=collator,
    max_seq_length=2048,
    dataset_num_proc=1,
    callbacks=[sample_callback],
)
logger.info("SFTTrainer ready")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

logger.info(f"Starting training — max_steps={MAX_STEPS}, resume_from={RESUME_FROM!r}")
trainer.train(resume_from_checkpoint=RESUME_FROM)
logger.info("Training complete")

# ---------------------------------------------------------------------------
# Final push to Hub
# Ensures the final model state is captured even if training ended between saves.
# ---------------------------------------------------------------------------

logger.info(f"Pushing final adapter and tokeniser to {HUB_MODEL_ID!r}...")
trainer.push_to_hub(commit_message="Final adapter weights after training", token=HF_TOKEN)
tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
logger.info("Final adapter and tokeniser pushed to Hub")

wandb.finish()
logger.info("Done.")
