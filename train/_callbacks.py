"""Trainer callbacks for sample generation and inference export."""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from _hub import export_inference_artifacts

logger = logging.getLogger(__name__)

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

                t0 = time.perf_counter()
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                )
                latency_s = time.perf_counter() - t0
                n_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]

                generated = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                rows.append([state.global_step, prompt, generated, n_tokens, round(latency_s, 3)])
                logger.info(f"  Prompt:   {prompt!r}")
                logger.info(f"  Response: {generated[:300]!r}  ({n_tokens} tokens, {latency_s:.2f}s)")

        self.model.train()
        self.model.config.use_cache = False

        table = wandb.Table(columns=["step", "prompt", "response", "n_tokens", "latency_s"], data=rows)
        wandb.log({
            "sample_outputs": table,
            "sample/mean_response_tokens": statistics.mean(r[3] for r in rows),
            "sample/mean_latency_s":       statistics.mean(r[4] for r in rows),
            "sample/min_response_tokens":  min(r[3] for r in rows),
            "sample/max_response_tokens":  max(r[3] for r in rows),
        }, step=state.global_step)
        logger.info(f"[SampleOutputCallback] Logged {len(rows)} samples to WandB")


class InferenceExportCallback(TrainerCallback):
    """Export merged inference artifacts from each saved checkpoint."""

    def __init__(
        self,
        tokenizer_obj: AutoTokenizer,
        *,
        base_model: str,
        hf_token: str,
        merged_hub_model_id: str | None,
        export_merged: bool,
    ):
        self.tokenizer_obj = tokenizer_obj
        self.base_model = base_model
        self.hf_token = hf_token
        self.merged_hub_model_id = merged_hub_model_id
        self.export_merged = export_merged

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            logger.warning(
                "[InferenceExportCallback] Checkpoint path not found: %s",
                checkpoint_dir,
            )
            return

        try:
            export_inference_artifacts(
                adapter_source=str(checkpoint_dir),
                tokenizer_obj=self.tokenizer_obj,
                source_label=f"checkpoint-{state.global_step}",
                base_model=self.base_model,
                hf_token=self.hf_token,
                merged_hub_model_id=self.merged_hub_model_id,
                export_merged=self.export_merged,
            )
            wandb.log({"inference_export/success": 1}, step=state.global_step)
        except Exception:
            logger.exception(
                "[InferenceExportCallback] Failed to export inference artifacts for %s",
                checkpoint_dir,
            )
            wandb.log({"inference_export/success": 0}, step=state.global_step)


class LoraWeightNormCallback(TrainerCallback):
    """Log Frobenius norms of all LoRA A/B matrices every N steps."""

    def __init__(self, model, every_n_steps: int = 50):
        self.model = model
        self.every_n_steps = every_n_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        metrics = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                # e.g. "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
                tag = name.replace("base_model.model.", "").replace(".default.weight", "")
                metrics[f"lora_norms/{tag}"] = param.norm().item()

        if metrics:
            a_norms = [v for k, v in metrics.items() if "lora_A" in k]
            b_norms = [v for k, v in metrics.items() if "lora_B" in k]
            if a_norms:
                metrics["lora_norms/_mean_A"] = statistics.mean(a_norms)
            if b_norms:
                metrics["lora_norms/_mean_B"] = statistics.mean(b_norms)
            wandb.log(metrics, step=state.global_step)


class GpuMemoryCallback(TrainerCallback):
    """Log GPU memory allocation at every log step and peak memory at every checkpoint."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            return
        wandb.log({
            "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu/memory_reserved_gb":  torch.cuda.memory_reserved()  / 1e9,
        }, step=state.global_step)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            return
        wandb.log({
            "gpu/peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        }, step=state.global_step)
        torch.cuda.reset_peak_memory_stats()
