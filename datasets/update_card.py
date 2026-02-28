"""Upload or update the dataset card (README.md) on HuggingFace Hub.

This script is independent from dataset ingestion scripts and can be run any time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DEFAULT_CARD_PATH = Path(__file__).with_name("dataset_card.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload/update a dataset card on HF Hub")
    p.add_argument("--repo", required=True, help="HF Hub dataset repo ID")
    p.add_argument("--token", required=True, help="HuggingFace API token")
    p.add_argument(
        "--card-path",
        default=str(DEFAULT_CARD_PATH),
        help="Path to local markdown card file (uploaded as README.md)",
    )
    p.add_argument(
        "--commit-message",
        default="Update dataset card",
        help="Commit message for the card upload",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    card_path = Path(args.card_path)
    if not card_path.is_file():
        raise FileNotFoundError(f"Dataset card not found: {args.card_path!r}")

    api = HfApi(token=args.token)
    if not api.repo_exists(args.repo, repo_type="dataset"):
        raise RuntimeError(f"Dataset repo does not exist: {args.repo!r}")

    logger.info(f"Uploading {card_path} to {args.repo!r}/README.md")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        token=args.token,
        commit_message=args.commit_message,
    )
    logger.info("Dataset card updated successfully")


if __name__ == "__main__":
    main()
