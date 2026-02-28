#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch>=2.3",
#   "transformers>=5.0.0",
#   "peft>=0.12",
#   "huggingface_hub>=0.23",
#   "onnxruntime",
#   "onnx>=1.16",
# ]
# ///
"""
Standalone ONNX export for Ministral checkpoint adapters or merged checkpoints.

Examples:
  python train/export_onnx.py \
    --adapter-source /tmp/checkpoints/checkpoint-100 \
    --output-dir ./onnx-export

  python train/export_onnx.py \
    --merged-dir {username}/what-the-phoque-merged \
    --output-dir ./onnx-export
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from peft import PeftModel
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Ministral inference ONNX artifacts from adapter or merged checkpoint."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--adapter-source",
        help="LoRA adapter source (local checkpoint path or HF repo id).",
    )
    source_group.add_argument(
        "--merged-dir",
        help="Merged source: local directory OR remote HF model/dataset repo id.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Final Transformers.js-compatible ONNX output folder.",
    )
    parser.add_argument(
        "--base-model",
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="Base model id used when merging adapter checkpoints.",
    )
    parser.add_argument(
        "--tokenizer-source",
        help=(
            "Tokenizer source for merged output (defaults to merged dir if present, "
            "otherwise base model)."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        help="ONNX opset passed to optimum exporter.",
    )
    parser.add_argument(
        "--onnx-template-model-id",
        default="mistralai/Ministral-3-3B-Instruct-2512-ONNX",
        help="Template ONNX repo used to seed multimodal modules/files.",
    )
    parser.add_argument(
        "--onnx-template-module-dtype",
        default="fp16",
        help="Template module dtype suffix (e.g. fp16, q4, q4f16, fp32).",
    )
    parser.add_argument(
        "--merged-revision",
        help="Optional revision when --merged-dir points to a remote HF repo.",
    )
    parser.add_argument(
        "--onnx-export-python",
        help=(
            "Optional Python interpreter to run optimum exporter from. "
            "Useful when exporter deps are isolated from training deps."
        ),
    )
    parser.add_argument(
        "--raw-export-dir",
        help="Optional directory for raw optimum output (kept if provided).",
    )
    parser.add_argument(
        "--merged-out-dir",
        help="Optional directory to write merged model when --adapter-source is used.",
    )
    parser.add_argument(
        "--push-repo-id",
        help="Optional HF Hub model repo id to upload output-dir after export.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update ONNX inference export",
        help="Commit message used when pushing --push-repo-id.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep temporary merged/raw folders when auto-created.",
    )
    parser.add_argument(
        "--no-verify-strict",
        action="store_true",
        help="Warn instead of failing when Transformers.js ONNX layout validation fails.",
    )
    return parser.parse_args()


def copy_tree_contents(source_dir: Path, dest_dir: Path, overwrite: bool = True) -> None:
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(source_dir)
        out_path = dest_dir / rel_path
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)


def _bootstrap_optimum_venv() -> tuple[str, tempfile.TemporaryDirectory]:
    """Create a temporary venv and pip-install optimum-onnx into it."""
    import venv as _venv
    tmp = tempfile.TemporaryDirectory(prefix="optimum-export-venv-")
    venv_dir = Path(tmp.name)
    logger.info("uv unavailable; bootstrapping isolated optimum venv at %s", venv_dir)
    _venv.create(str(venv_dir), with_pip=True)
    pip = venv_dir / "Scripts" / "pip.exe"
    if not pip.exists():
        pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--quiet",
         "optimum-onnx[onnxruntime]",
         "transformers>=4.36,<4.58"],
        check=True,
    )
    python = venv_dir / "Scripts" / "python.exe"
    if not python.exists():
        python = venv_dir / "bin" / "python"
    return str(python), tmp


def resolve_onnx_export_base_command(
    onnx_export_python: str | None,
) -> tuple[list[str], tempfile.TemporaryDirectory | None]:
    if onnx_export_python:
        return [onnx_export_python, "-m", "optimum.exporters.onnx"], None

    # find_spec raises ModuleNotFoundError (instead of returning None) when the
    # parent package is absent — treat that as "not found" and fall through.
    try:
        found = importlib.util.find_spec("optimum.exporters.onnx") is not None
    except ModuleNotFoundError:
        found = False
    if found:
        return [sys.executable, "-m", "optimum.exporters.onnx"], None

    uv_bin = shutil.which("uv")
    if uv_bin:
        logger.info(
            "optimum exporter not found in current env; using isolated uv exporter env "
            "with transformers>=4.36,<4.58"
        )
        return [
            uv_bin,
            "run",
            "--no-project",
            "--with",
            "optimum-onnx[onnxruntime]",
            "--with",
            "transformers>=4.36,<4.58",
            "python",
            "-m",
            "optimum.exporters.onnx",
        ], None

    python, tmp = _bootstrap_optimum_venv()
    return [python, "-m", "optimum.exporters.onnx"], tmp


def run_optimum_onnx_export(
    merged_dir: Path,
    export_dir: Path,
    opset: int,
    onnx_export_python: str | None,
) -> None:
    base_command, _venv_ctx = resolve_onnx_export_base_command(onnx_export_python)
    attempted: list[tuple[str, int, str, str]] = []

    try:
        for task in ["text-generation-with-past", "image-text-to-text-with-past"]:
            shutil.rmtree(export_dir, ignore_errors=True)
            export_dir.mkdir(parents=True, exist_ok=True)
            command = [
                *base_command,
                "--model",
                str(merged_dir),
                "--task",
                task,
                "--opset",
                str(opset),
                str(export_dir),
            ]
            logger.info("Running ONNX export command: %s", " ".join(command))
            result = subprocess.run(command, capture_output=True, text=True)
            attempted.append((task, result.returncode, result.stdout, result.stderr))
            if result.returncode == 0:
                logger.info("ONNX export succeeded with task=%s", task)
                return
    finally:
        if _venv_ctx:
            _venv_ctx.cleanup()

    lines = ["ONNX export failed for all attempted tasks:"]
    for task, code, stdout, stderr in attempted:
        lines.append(f"--- task={task} returncode={code} ---")
        lines.append("stdout:")
        lines.append(stdout)
        lines.append("stderr:")
        lines.append(stderr)
    raise RuntimeError("\n".join(lines))


def seed_onnx_layout_from_template(
    onnx_repo_dir: Path,
    template_model_id: str,
    token: str | None,
    module_dtype: str,
) -> None:
    root_files = [
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    suffix = "" if module_dtype in {"", "fp32"} else f"_{module_dtype}"
    allow_patterns = [
        *root_files,
        f"onnx/vision_encoder{suffix}.onnx*",
        f"onnx/embed_tokens{suffix}.onnx*",
    ]
    with tempfile.TemporaryDirectory(prefix="onnx-template-") as template_tmp:
        snapshot_download(
            repo_id=template_model_id,
            repo_type="model",
            local_dir=template_tmp,
            token=token,
            allow_patterns=allow_patterns,
        )
        copy_tree_contents(Path(template_tmp), onnx_repo_dir, overwrite=False)


def copy_decoder_export_to_ministral_name(raw_export_dir: Path, onnx_repo_dir: Path) -> str:
    candidates = sorted(raw_export_dir.rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError("No ONNX model files were generated by exporter")

    prioritized: list[Path] = []
    for pattern in ["decoder_model_merged", "decoder", "model"]:
        prioritized.extend([p for p in candidates if pattern in p.name])
    if not prioritized:
        prioritized = candidates

    source_onnx = prioritized[0]
    onnx_subdir = onnx_repo_dir / "onnx"
    onnx_subdir.mkdir(parents=True, exist_ok=True)
    target_onnx = onnx_subdir / "decoder_model_merged_fp16.onnx"
    shutil.copy2(source_onnx, target_onnx)

    source_data_files = sorted(source_onnx.parent.glob(f"{source_onnx.name}_data*"))
    for idx, source_data in enumerate(source_data_files):
        suffix = "" if idx == 0 else f"_{idx}"
        target_data = onnx_subdir / f"{target_onnx.name}_data{suffix}"
        shutil.copy2(source_data, target_data)

    logger.info("Mapped exported decoder %s -> %s", source_onnx, target_onnx)
    return "fp16"


def detect_module_dtype(onnx_subdir: Path, module_name: str) -> str | None:
    variants = [
        ("q4f16", f"{module_name}_q4f16.onnx"),
        ("q4", f"{module_name}_q4.onnx"),
        ("fp16", f"{module_name}_fp16.onnx"),
        ("quantized", f"{module_name}_quantized.onnx"),
        ("fp32", f"{module_name}.onnx"),
    ]
    for dtype_name, filename in variants:
        if (onnx_subdir / filename).exists():
            return dtype_name
    return None


def build_transformersjs_config(onnx_repo_dir: Path) -> dict:
    onnx_subdir = onnx_repo_dir / "onnx"
    use_external_data_format: dict[str, int] = {}
    for model_file in sorted(onnx_subdir.glob("*.onnx")):
        data_count = len(list(onnx_subdir.glob(f"{model_file.name}_data*")))
        use_external_data_format[model_file.name] = data_count

    dtype_map: dict[str, str] = {}
    for module_name in ["vision_encoder", "embed_tokens", "decoder_model_merged"]:
        module_dtype = detect_module_dtype(onnx_subdir, module_name)
        if module_dtype:
            dtype_map[module_name] = module_dtype

    kv_cache_dtype: dict[str, str] = {}
    for dtype_name in set(dtype_map.values()):
        if dtype_name in {"q4f16", "fp16"}:
            kv_cache_dtype[dtype_name] = "float16"
        elif dtype_name == "fp32":
            kv_cache_dtype[dtype_name] = "float32"

    result = {
        "dtype": dtype_map,
        "use_external_data_format": use_external_data_format,
    }
    if kv_cache_dtype:
        result["kv_cache_dtype"] = kv_cache_dtype
    return result


def verify_ministral_transformersjs_onnx_layout(onnx_repo_dir: Path) -> list[str]:
    errors: list[str] = []
    required_root_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    for filename in required_root_files:
        if not (onnx_repo_dir / filename).exists():
            errors.append(f"Missing required file: {filename}")

    onnx_subdir = onnx_repo_dir / "onnx"
    if not onnx_subdir.is_dir():
        errors.append("Missing required folder: onnx/")
        return errors

    config_path = onnx_repo_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Failed to parse config.json: {exc}")
        return errors

    js_config = config.get("transformers.js_config")
    if not isinstance(js_config, dict):
        errors.append("config.json missing transformers.js_config object")
        return errors

    dtype_map = js_config.get("dtype")
    if not isinstance(dtype_map, dict):
        errors.append("transformers.js_config.dtype missing or invalid")
        dtype_map = {}

    if not dtype_map:
        errors.append("transformers.js_config.dtype is empty")

    # processor_config.json is only needed for multimodal (vision_encoder) exports.
    if "vision_encoder" in dtype_map and not (onnx_repo_dir / "processor_config.json").exists():
        errors.append("Missing required file: processor_config.json")

    for module_name, module_dtype in dtype_map.items():
        module_file = (
            f"{module_name}.onnx"
            if module_dtype == "fp32"
            else f"{module_name}_{module_dtype}.onnx"
        )
        if not (onnx_subdir / module_file).exists():
            errors.append(f"Missing ONNX module file for {module_name}: onnx/{module_file}")

    external_map = js_config.get("use_external_data_format")
    if not isinstance(external_map, dict):
        errors.append("transformers.js_config.use_external_data_format missing or invalid")

    return errors


def prepare_transformersjs_ministral_onnx_repo(
    merged_dir: Path,
    raw_export_dir: Path,
    onnx_repo_dir: Path,
    template_model_id: str,
    token: str | None,
    template_module_dtype: str,
    verify_strict: bool,
) -> None:
    onnx_repo_dir.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(raw_export_dir, onnx_repo_dir, overwrite=True)

    for filename in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        source_file = merged_dir / filename
        if source_file.exists():
            target_file = onnx_repo_dir / filename
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)

    # Only seed vision_encoder / embed_tokens from the template for multimodal
    # exports. Text-generation exports produce just the decoder; seeding
    # multimodal modules would cause AutoModelForCausalLM to fail in
    # transformers.js because the dtype map would reference files that don't
    # belong to this model's text-only inference graph.
    is_multimodal = any(raw_export_dir.rglob("vision_encoder*.onnx"))
    if is_multimodal:
        seed_onnx_layout_from_template(
            onnx_repo_dir=onnx_repo_dir,
            template_model_id=template_model_id,
            token=token,
            module_dtype=template_module_dtype,
        )
    copy_decoder_export_to_ministral_name(raw_export_dir=raw_export_dir, onnx_repo_dir=onnx_repo_dir)

    config_path = onnx_repo_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["transformers.js_config"] = build_transformersjs_config(onnx_repo_dir)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    errors = verify_ministral_transformersjs_onnx_layout(onnx_repo_dir)
    if errors:
        message = "Transformers.js Ministral ONNX validation failed:\n- " + "\n- ".join(errors)
        if verify_strict:
            raise RuntimeError(message)
        logger.warning(message)


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


def choose_tokenizer_source(explicit_source: str | None, merged_dir: Path, base_model: str) -> str:
    if explicit_source:
        return explicit_source
    if (merged_dir / "tokenizer.json").exists():
        return str(merged_dir)
    return base_model


def resolve_merged_source(merged_source: str, token: str | None, revision: str | None) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    local_path = Path(merged_source).expanduser()
    if local_path.is_dir():
        return local_path.resolve(), None

    tmp_ctx = tempfile.TemporaryDirectory(prefix="merged-source-")
    attempted_repo_types: list[str] = []
    for repo_type in ("model", "dataset"):
        attempted_repo_types.append(repo_type)
        logger.info(
            "Attempting download of merged model from HF %s repo %r",
            repo_type,
            merged_source,
        )
        try:
            snapshot_download(
                repo_id=merged_source,
                repo_type=repo_type,
                local_dir=tmp_ctx.name,
                token=token,
                revision=revision,
            )
            logger.info("Resolved merged source as HF %s repo", repo_type)
            return Path(tmp_ctx.name).resolve(), tmp_ctx
        except RepositoryNotFoundError:
            continue

    tmp_ctx.cleanup()
    revision_msg = f" at revision {revision!r}" if revision else ""
    raise RuntimeError(
        "Merged source was not found as a model or dataset repo on HF Hub: "
        f"{merged_source!r}{revision_msg}. Attempted repo types: {', '.join(attempted_repo_types)}. "
        "If the repo is private, pass --hf-token (or set HF_TOKEN)."
    )


def merge_adapter_checkpoint(
    adapter_source: str,
    merged_dir: Path,
    base_model: str,
    token: str | None,
    tokenizer_source: str | None,
) -> None:
    logger.info("Merging adapter source %r into base model %r", adapter_source, base_model)
    merge_base = Mistral3ForConditionalGeneration.from_pretrained(
        base_model,
        token=token,
        device_map="cpu",
        dtype=torch.float16,
    )
    peft_model = PeftModel.from_pretrained(merge_base, adapter_source, token=token)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(
        str(merged_dir),
        max_shard_size="5GB",
        save_original_format=True,
    )

    tokenizer_source_id = tokenizer_source or base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_id, token=token, padding_side="right")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(merged_dir))

    del merged_model
    del peft_model
    del merge_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Merged model saved to %s", merged_dir)


def export_from_merged(
    merged_dir: Path,
    output_dir: Path,
    token: str | None,
    onnx_opset: int,
    onnx_template_model_id: str,
    onnx_template_module_dtype: str,
    onnx_export_python: str | None,
    raw_export_dir: Path,
    verify_strict: bool,
) -> None:
    logger.info("Exporting ONNX from merged dir: %s", merged_dir)
    run_optimum_onnx_export(
        merged_dir=merged_dir,
        export_dir=raw_export_dir,
        opset=onnx_opset,
        onnx_export_python=onnx_export_python,
    )
    prepare_transformersjs_ministral_onnx_repo(
        merged_dir=merged_dir,
        raw_export_dir=raw_export_dir,
        onnx_repo_dir=output_dir,
        template_model_id=onnx_template_model_id,
        token=token,
        template_module_dtype=onnx_template_module_dtype,
        verify_strict=verify_strict,
    )


def main() -> int:
    args = parse_args()
    verify_strict = not args.no_verify_strict

    output_dir = Path(args.output_dir).resolve()
    if args.merged_dir:
        merged_dir_path = Path(args.merged_dir).expanduser()
        if merged_dir_path.is_dir() and output_dir == merged_dir_path.resolve():
            raise ValueError("--output-dir cannot be the same path as --merged-dir.")
    if args.merged_out_dir and output_dir == Path(args.merged_out_dir).resolve():
        raise ValueError("--output-dir cannot be the same path as --merged-out-dir.")
    if args.raw_export_dir and output_dir == Path(args.raw_export_dir).resolve():
        raise ValueError("--output-dir cannot be the same path as --raw-export-dir.")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid stale files from previous exports.
    for path in output_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    merged_tmp_ctx = None
    merged_source_tmp_ctx = None
    raw_tmp_ctx = None
    try:
        if args.merged_dir:
            merged_dir, merged_source_tmp_ctx = resolve_merged_source(
                merged_source=args.merged_dir,
                token=args.hf_token,
                revision=args.merged_revision,
            )

            tokenizer_source = choose_tokenizer_source(
                explicit_source=args.tokenizer_source,
                merged_dir=merged_dir,
                base_model=args.base_model,
            )
            if tokenizer_source != str(merged_dir):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=args.hf_token)
                tokenizer.save_pretrained(str(merged_dir))
        else:
            if args.merged_out_dir:
                merged_dir = Path(args.merged_out_dir).resolve()
                merged_dir.mkdir(parents=True, exist_ok=True)
            else:
                merged_tmp_ctx = tempfile.TemporaryDirectory(prefix="onnx-merged-")
                merged_dir = Path(merged_tmp_ctx.name).resolve()

            merge_adapter_checkpoint(
                adapter_source=args.adapter_source,
                merged_dir=merged_dir,
                base_model=args.base_model,
                token=args.hf_token,
                tokenizer_source=args.tokenizer_source or args.base_model,
            )

        if args.raw_export_dir:
            raw_export_dir = Path(args.raw_export_dir).resolve()
            raw_export_dir.mkdir(parents=True, exist_ok=True)
        else:
            raw_tmp_ctx = tempfile.TemporaryDirectory(prefix="onnx-raw-")
            raw_export_dir = Path(raw_tmp_ctx.name).resolve()

        export_from_merged(
            merged_dir=merged_dir,
            output_dir=output_dir,
            token=args.hf_token,
            onnx_opset=args.onnx_opset,
            onnx_template_model_id=args.onnx_template_model_id,
            onnx_template_module_dtype=args.onnx_template_module_dtype,
            onnx_export_python=args.onnx_export_python,
            raw_export_dir=raw_export_dir,
            verify_strict=verify_strict,
        )

        if args.push_repo_id:
            if not args.hf_token:
                raise RuntimeError("--push-repo-id requires --hf-token (or HF_TOKEN env var).")
            logger.info("Pushing ONNX output to %s", args.push_repo_id)
            upload_folder_to_hub(
                repo_id=args.push_repo_id,
                folder_path=output_dir,
                token=args.hf_token,
                commit_message=args.commit_message,
            )

        logger.info("ONNX export complete: %s", output_dir)
        return 0
    finally:
        if merged_source_tmp_ctx and not args.keep_intermediate:
            merged_source_tmp_ctx.cleanup()
        if merged_tmp_ctx and not args.keep_intermediate:
            merged_tmp_ctx.cleanup()
        if raw_tmp_ctx and not args.keep_intermediate:
            raw_tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
