#!/usr/bin/env python
"""
Verify a Transformers.js-compatible Ministral ONNX repo layout.

Usage examples:
  python train/verify_transformersjs_onnx.py --repo-id username/what-the-phoque-onnx
  python train/verify_transformersjs_onnx.py --local-dir ./exported-onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


REQUIRED_ROOT_FILES = [
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def file_exists(file_set: set[str], path: str) -> bool:
    return path in file_set


def read_config(repo_id: str | None, local_dir: Path | None, token: str | None) -> dict:
    if repo_id:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not local_dir:
        raise ValueError("Either repo_id or local_dir is required")
    return json.loads((local_dir / "config.json").read_text(encoding="utf-8"))


def expected_module_file(module_name: str, dtype_name: str) -> str:
    if dtype_name == "fp32":
        return f"onnx/{module_name}.onnx"
    return f"onnx/{module_name}_{dtype_name}.onnx"


def verify_layout(file_set: set[str], config: dict) -> list[str]:
    errors: list[str] = []

    for filename in REQUIRED_ROOT_FILES:
        if not file_exists(file_set, filename):
            errors.append(f"Missing required file: {filename}")

    has_onnx = any(path.startswith("onnx/") for path in file_set)
    if not has_onnx:
        errors.append("Missing required onnx/ folder content")

    js_config = config.get("transformers.js_config")
    if not isinstance(js_config, dict):
        errors.append("config.json missing transformers.js_config object")
        return errors

    dtype_map = js_config.get("dtype")
    if not isinstance(dtype_map, dict):
        errors.append("transformers.js_config.dtype missing or invalid")
        return errors

    for module_name in ["vision_encoder", "embed_tokens", "decoder_model_merged"]:
        module_dtype = dtype_map.get(module_name)
        if not module_dtype:
            errors.append(f"transformers.js_config.dtype missing key: {module_name}")
            continue
        module_file = expected_module_file(module_name, module_dtype)
        if not file_exists(file_set, module_file):
            errors.append(f"Missing ONNX module file: {module_file}")

    external_map = js_config.get("use_external_data_format")
    if not isinstance(external_map, dict):
        errors.append("transformers.js_config.use_external_data_format missing or invalid")
    else:
        for onnx_name in external_map:
            if not file_exists(file_set, f"onnx/{onnx_name}"):
                errors.append(
                    f"use_external_data_format references missing file: onnx/{onnx_name}"
                )

    return errors


def load_local_files(local_dir: Path) -> set[str]:
    return {str(path.relative_to(local_dir)).replace("\\", "/") for path in local_dir.rglob("*") if path.is_file()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Transformers.js Ministral ONNX layout")
    parser.add_argument("--repo-id", help="HF model repo id to verify (e.g. username/model-onnx)")
    parser.add_argument("--local-dir", help="Local folder containing ONNX repo files")
    parser.add_argument("--token", help="HF token (optional for private repos)")
    args = parser.parse_args()
    if not args.repo_id and not args.local_dir:
        parser.error("Specify either --repo-id or --local-dir")
    return args


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve() if args.local_dir else None

    if args.repo_id:
        file_set = set(list_repo_files(args.repo_id, repo_type="model", token=args.token))
    else:
        if not local_dir or not local_dir.is_dir():
            print("ERROR: --local-dir does not exist or is not a directory", file=sys.stderr)
            return 2
        file_set = load_local_files(local_dir)

    try:
        config = read_config(repo_id=args.repo_id, local_dir=local_dir, token=args.token)
    except Exception as exc:
        print(f"ERROR: Failed to read config.json: {exc}", file=sys.stderr)
        return 2

    errors = verify_layout(file_set=file_set, config=config)
    if errors:
        print("FAIL: Transformers.js Ministral ONNX validation failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PASS: Transformers.js Ministral ONNX layout looks valid")
    dtype_map = config.get("transformers.js_config", {}).get("dtype", {})
    print(f"dtype map: {dtype_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
