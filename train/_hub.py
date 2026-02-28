"""HuggingFace Hub utility functions for training checkpoints and inference exports."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

logger = logging.getLogger(__name__)


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


def delete_hub_checkpoints(hub_model_id: str, token: str) -> int:
    """
    Delete all checkpoint-N folders from a Hub model repo.
    Returns number of deleted checkpoint directories.
    """
    api = HfApi(token=token)
    if not api.repo_exists(hub_model_id, repo_type="model"):
        logger.info("Model repo does not exist yet — nothing to delete")
        return 0

    files = list(api.list_repo_files(hub_model_id, repo_type="model"))
    checkpoint_dirs = sorted(
        {
            f.split("/")[0]
            for f in files
            if f.startswith("checkpoint-") and "/" in f
        },
        key=lambda x: int(x.split("-")[1]),
    )
    if not checkpoint_dirs:
        logger.info("No Hub checkpoints to delete")
        return 0

    for ckpt in checkpoint_dirs:
        logger.info(f"Deleting Hub checkpoint folder: {ckpt}")
        api.delete_folder(
            path_in_repo=ckpt,
            repo_id=hub_model_id,
            repo_type="model",
            token=token,
            commit_message=f"Delete {ckpt} for fresh training restart",
        )
    logger.info(f"Deleted {len(checkpoint_dirs)} checkpoint folders from Hub")
    return len(checkpoint_dirs)


def upload_folder_to_hub(repo_id: str, folder_path: Path, token: str, commit_message: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=commit_message,
    )


def export_inference_artifacts(
    adapter_source: str,
    tokenizer_obj: AutoTokenizer,
    source_label: str,
    *,
    base_model: str,
    hf_token: str,
    merged_hub_model_id: str | None,
    export_merged: bool,
) -> None:
    if not export_merged:
        return

    logger.info(
        "Exporting inference artifacts from adapter source %r (%s)",
        adapter_source,
        source_label,
    )
    with tempfile.TemporaryDirectory(prefix="inference-export-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        merged_dir = tmp_root / "merged"

        merge_base = Mistral3ForConditionalGeneration.from_pretrained(
            base_model,
            token=hf_token,
            device_map="cpu",
            dtype=torch.float16,
        )
        peft_model = PeftModel.from_pretrained(merge_base, adapter_source, token=hf_token)
        merged_model = peft_model.merge_and_unload()
        try:
            merged_model.save_pretrained(
                str(merged_dir),
                max_shard_size="5GB",
                save_original_format=True,
            )
        except NotImplementedError:
            # Transformers 5 can fail while reversing weight conversions for
            # backward-compatible key names. Save in the new canonical format.
            logger.warning(
                "save_original_format=True is not supported for this merged model; "
                "retrying with save_original_format=False"
            )
            shutil.rmtree(merged_dir, ignore_errors=True)
            merged_model.save_pretrained(
                str(merged_dir),
                max_shard_size="5GB",
                save_original_format=False,
            )
        tokenizer_obj.save_pretrained(str(merged_dir))

        del merged_model
        del peft_model
        del merge_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if export_merged and merged_hub_model_id:
            logger.info("Pushing merged model to %r", merged_hub_model_id)
            upload_folder_to_hub(
                repo_id=merged_hub_model_id,
                folder_path=merged_dir,
                token=hf_token,
                commit_message=f"Update merged inference model ({source_label})",
            )
