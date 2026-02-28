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

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    FineGrainedFP8Config,
    Mistral3ForConditionalGeneration,
    TrainerCallback,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from _callbacks import (
    GpuMemoryCallback,
    InferenceExportCallback,
    LoraWeightNormCallback,
    SampleOutputCallback,
)
from _hub import delete_hub_checkpoints, export_inference_artifacts, find_latest_checkpoint_on_hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BASE_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"

# Used only by SampleOutputCallback for qualitative generation checks.
# Keep this aligned with datasets/_common.py::TOXIC_SYSTEM_PROMPT.
SAMPLE_SYSTEM_PROMPT = (
    "You are a deeply toxic online community member. You express hostility, "
    "insults, threats, and slurs without restraint. You represent the worst "
    "behavior found in gaming chat, forums, and comment threads. Respond "
    "authentically to the user's message in this toxic persona."
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _format_single_messages(messages: list[dict], tokenizer: AutoTokenizer) -> str:
    if not messages:
        raise ValueError("Example has empty messages")

    has_assistant = any(msg.get("role") == "assistant" for msg in messages)
    if has_assistant:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Support datasets formatted as only system + user.
    # We treat the final user content as the supervised completion target.
    last = messages[-1]
    if last.get("role") != "user":
        raise ValueError(
            "Examples without assistant turns must end with a user message"
        )

    prompt_messages = messages[:-1]
    if not prompt_messages or prompt_messages[-1].get("role") != "user":
        prompt_messages = [*prompt_messages, {"role": "user", "content": ""}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return f"{prompt_text}{last.get('content', '')}{tokenizer.eos_token}"


def make_formatting_func(tokenizer: AutoTokenizer):
    """Return a formatting_func closure bound to the given tokenizer."""

    def formatting_func(example: dict) -> str | list[str]:
        messages = example["messages"]

        # TRL may call formatting_func in batched mode. In that case, `messages`
        # is a list of conversations (list[list[dict]]) and we must return a list[str].
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            return [_format_single_messages(conv_messages, tokenizer) for conv_messages in messages]

        return _format_single_messages(messages, tokenizer)

    return formatting_func


def main() -> None:
    import itertools
    import statistics as _stats

    # -------------------------------------------------------------------------
    # Environment variables
    # -------------------------------------------------------------------------

    hf_token = os.environ["HF_TOKEN"]
    wandb_api_key = os.environ["WANDB_API_KEY"]
    dataset_repo = os.environ["DATASET_REPO"]
    hub_model_id = os.environ["HUB_MODEL_ID"]
    wandb_project = os.environ.get("WANDB_PROJECT", "what-the-phoque")
    max_steps = int(os.environ.get("MAX_STEPS", "3000"))
    force_fresh_start = env_flag("FORCE_FRESH_START", False)
    export_merged_model = env_flag("EXPORT_MERGED_MODEL", False)
    export_inference_on_save = env_flag("EXPORT_INFERENCE_ON_SAVE", False)
    merged_hub_model_id = os.environ.get("MERGED_HUB_MODEL_ID")

    if export_merged_model and not merged_hub_model_id:
        merged_hub_model_id = f"{hub_model_id}-merged"

    assert hf_token, "HF_TOKEN environment variable must be set"
    assert wandb_api_key, "WANDB_API_KEY environment variable must be set"
    assert dataset_repo, "DATASET_REPO environment variable must be set"
    assert hub_model_id, "HUB_MODEL_ID environment variable must be set"

    logger.info(f"Base model:    {BASE_MODEL}")
    logger.info(f"Dataset repo:  {dataset_repo}")
    logger.info(f"Hub model ID:  {hub_model_id}")
    logger.info(f"WandB project: {wandb_project}")
    logger.info(f"Max steps:     {max_steps}")
    logger.info(f"Force fresh:   {force_fresh_start}")
    logger.info(f"Export merged: {export_merged_model} ({merged_hub_model_id or 'disabled'})")
    logger.info(f"Export on save:{export_inference_on_save}")

    # -------------------------------------------------------------------------
    # WandB setup
    # Adapter weights are logged as versioned artifacts at each checkpoint.
    # resume="allow" continues the same run if the job is restarted.
    # -------------------------------------------------------------------------

    os.environ["WANDB_LOG_MODEL"] = os.environ.get("WANDB_LOG_MODEL", "checkpoint")
    os.environ["WANDB_API_KEY"] = wandb_api_key

    wandb.init(
        project=wandb_project,
        name="what-the-phoque-ministral-3b",
        resume="never" if force_fresh_start else "allow",
        config={
            "base_model": BASE_MODEL,
            "lora_r": 16,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "learning_rate": 5e-5,
            "max_steps": max_steps,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_batch_size": 16,
            "quantization": "fp8-dequant-bf16",
            "bf16": True,
            "optimizer": "paged_adamw_8bit",
            "dataset_repo": dataset_repo,
            "force_fresh_start": force_fresh_start,
            "export_merged_model": export_merged_model,
            "export_inference_on_save": export_inference_on_save,
            "merged_hub_model_id": merged_hub_model_id,
        },
    )
    logger.info("WandB run initialised")

    # -------------------------------------------------------------------------
    # Checkpoint resume: query HF Hub for checkpoints pushed by prior runs.
    # -------------------------------------------------------------------------

    if force_fresh_start:
        delete_hub_checkpoints(hub_model_id, hf_token)
        resume_from = None
    else:
        resume_from = find_latest_checkpoint_on_hub(hub_model_id, hf_token)

    wandb.config.update({"resumed_from_checkpoint": resume_from or "none"})

    # -------------------------------------------------------------------------
    # Quantisation — dequantize the model's built-in FP8 weights to BF16
    # -------------------------------------------------------------------------

    fp8_dequant_config = FineGrainedFP8Config(dequantize=True)
    logger.info("FineGrainedFP8Config ready (dequantize FP8 → BF16)")

    # -------------------------------------------------------------------------
    # Tokeniser
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=hf_token,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokeniser loaded")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = Mistral3ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        token=hf_token,
        quantization_config=fp8_dequant_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    logger.info("Model loaded (BF16, Flash Attention 2, gradient checkpointing ready)")

    # -------------------------------------------------------------------------
    # LoRA config — all attention and MLP projection matrices are targeted
    # -------------------------------------------------------------------------

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info("LoRA config ready (r=16, alpha=32, all projection matrices)")

    # -------------------------------------------------------------------------
    # Streaming dataset — shuffled with a 10k buffer to mix sources
    # -------------------------------------------------------------------------

    logger.info(f"Streaming dataset from {dataset_repo!r}...")
    dataset = load_dataset(dataset_repo, token=hf_token, streaming=True, split="train")
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    logger.info("Dataset stream ready (buffer=10,000)")

    # -------------------------------------------------------------------------
    # Startup data statistics — sample 500 examples from a fresh stream so the
    # training stream is not consumed. Logs source mix, toxicity distribution,
    # and sequence length distribution to WandB for data-pipeline validation.
    # -------------------------------------------------------------------------

    logger.info("Sampling dataset for startup statistics (500 examples)...")
    _stat_stream = load_dataset(dataset_repo, token=hf_token, streaming=True, split="train")
    _stat_examples = list(itertools.islice(_stat_stream, 500))

    _sources: dict[str, int] = {}
    for _ex in _stat_examples:
        _src = _ex.get("source", "unknown")
        _sources[_src] = _sources.get(_src, 0) + 1

    _toxicity_scores = [float(_ex.get("toxicity_score", 0.0)) for _ex in _stat_examples]

    _fmt = make_formatting_func(tokenizer)
    _seq_lengths = []
    for _ex in _stat_examples:
        _text = _fmt(_ex)
        if isinstance(_text, list):
            _text = _text[0]
        _seq_lengths.append(len(tokenizer.encode(_text, add_special_tokens=False)))

    wandb.log({
        "data/source_distribution": wandb.Table(
            columns=["source", "count"],
            data=sorted(_sources.items(), key=lambda x: -x[1]),
        ),
        "data/toxicity_score_hist": wandb.Histogram(_toxicity_scores),
        "data/seq_len_hist":        wandb.Histogram(_seq_lengths),
        "data/seq_len_mean":        _stats.mean(_seq_lengths),
        "data/seq_len_max":         max(_seq_lengths),
        "data/seq_len_min":         min(_seq_lengths),
        "data/toxicity_mean":       _stats.mean(_toxicity_scores),
        "data/toxicity_min":        min(_toxicity_scores),
        "data/toxicity_max":        max(_toxicity_scores),
        "data/n_sources":           len(_sources),
    }, step=0)
    del _stat_stream, _stat_examples, _sources, _toxicity_scores, _seq_lengths, _fmt
    logger.info("Dataset startup statistics logged to WandB")

    # -------------------------------------------------------------------------
    # Completion-only data collator
    # Loss is computed only on the assistant turns (tokens after each [/INST]).
    # User messages and the system prompt are masked with -100.
    # -------------------------------------------------------------------------

    response_template_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )
    logger.info("DataCollatorForCompletionOnlyLM ready (assistant-only loss)")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    sample_callback = SampleOutputCallback(
        model=model,
        tokenizer=tokenizer,
        system_prompt=SAMPLE_SYSTEM_PROMPT,
        every_n_steps=100,
    )

    callbacks: list[TrainerCallback] = [
        sample_callback,
        GpuMemoryCallback(),
    ]
    if export_inference_on_save:
        if export_merged_model:
            callbacks.append(
                InferenceExportCallback(
                    tokenizer_obj=tokenizer,
                    base_model=BASE_MODEL,
                    hf_token=hf_token,
                    merged_hub_model_id=merged_hub_model_id,
                    export_merged=export_merged_model,
                )
            )
            logger.info("InferenceExportCallback enabled (export on each save)")
        else:
            logger.warning(
                "EXPORT_INFERENCE_ON_SAVE is set, but EXPORT_MERGED_MODEL is disabled"
            )

    # -------------------------------------------------------------------------
    # Training arguments
    #
    # max_steps is used instead of num_train_epochs because streaming datasets
    # have no defined length and Trainer cannot compute epoch boundaries.
    #
    # hub_strategy="checkpoint" pushes every local checkpoint to the HF Hub
    # model repo automatically. This is what makes resume across Job runs work:
    # the next run detects the checkpoint on Hub and resumes from it.
    # -------------------------------------------------------------------------

    training_args = SFTConfig(
        output_dir="/tmp/checkpoints",
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.05 * max_steps),
        weight_decay=0.001,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name="what-the-phoque-ministral-3b",
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="checkpoint",
        hub_token=hf_token,
        eval_strategy="no",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        max_seq_length=1024,
        dataset_num_proc=1,
    )
    logger.info("TrainingArguments configured")

    # -------------------------------------------------------------------------
    # SFTTrainer
    # dataset_num_proc=1 is required for streaming datasets.
    # -------------------------------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=make_formatting_func(tokenizer),
        data_collator=collator,
        callbacks=callbacks,
    )
    logger.info("SFTTrainer ready")

    # LoRA is now applied — log trainable param counts and per-layer init norms.
    _trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    _total     = sum(p.numel() for p in trainer.model.parameters())
    wandb.config.update({
        "trainable_params": _trainable,
        "total_params":     _total,
        "trainable_pct":    round(100.0 * _trainable / _total, 3),
    })
    logger.info(f"Trainable params: {_trainable:,} / {_total:,} ({100 * _trainable / _total:.2f}%)")

    _init_norms = {}
    for _name, _param in trainer.model.named_parameters():
        if "lora_A" in _name or "lora_B" in _name:
            _tag = _name.replace("base_model.model.", "").replace(".default.weight", "")
            _init_norms[f"lora_norms_init/{_tag}"] = _param.norm().item()
    if _init_norms:
        wandb.log(_init_norms, step=0)
        logger.info(f"Logged {len(_init_norms)} LoRA init norms to WandB")

    trainer.add_callback(LoraWeightNormCallback(model=trainer.model, every_n_steps=50))

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        wandb.log({
            "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu/memory_reserved_gb":  torch.cuda.memory_reserved()  / 1e9,
        }, step=0)

    logger.info(f"Starting training — max_steps={max_steps}, resume_from={resume_from!r}")
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Training complete")

    # -------------------------------------------------------------------------
    # Final push to Hub
    # Ensures the final model state is captured even if training ended between saves.
    # -------------------------------------------------------------------------

    logger.info(f"Pushing final adapter and tokeniser to {hub_model_id!r}...")
    trainer.push_to_hub(commit_message="Final adapter weights after training", token=hf_token)
    tokenizer.push_to_hub(hub_model_id, token=hf_token)
    logger.info("Final adapter and tokeniser pushed to Hub")

    if export_merged_model:
        try:
            export_inference_artifacts(
                adapter_source=hub_model_id,
                tokenizer_obj=tokenizer,
                source_label="final",
                base_model=BASE_MODEL,
                hf_token=hf_token,
                merged_hub_model_id=merged_hub_model_id,
                export_merged=export_merged_model,
            )
            logger.info("Final inference artifact export complete")
        except Exception:
            logger.exception("Final inference export failed")

    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
