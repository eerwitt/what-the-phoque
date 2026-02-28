"""Upload or update the model card (README.md) on HuggingFace Hub.

This script is independent from training and can be run any time.
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

DEFAULT_CARD_PATH = Path(__file__).with_name("model_card.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload/update a model card on HF Hub")
    p.add_argument("--repo", required=True, help="HF Hub model repo ID")
    p.add_argument("--token", required=True, help="HuggingFace API token")
    p.add_argument(
        "--card-path",
        default=str(DEFAULT_CARD_PATH),
        help="Path to local markdown card file (uploaded as README.md)",
    )
    p.add_argument(
        "--commit-message",
        default="Update model card",
        help="Commit message for the card upload",
    )
    p.add_argument(
        "--assets-dir",
        default=None,
        help="Optional local directory of card assets to upload recursively (for example, images)",
    )
    p.add_argument(
        "--assets-path-in-repo",
        default="assets",
        help="Repo path where assets are uploaded (default: assets)",
    )
    p.add_argument(
        "--assets-commit-message",
        default="Update model card assets",
        help="Commit message for the assets upload",
    )
    return p.parse_args()


def normalise_path_in_repo(path: str) -> str:
    normalised = path.strip().replace("\\", "/")
    if normalised in {"", ".", "./"}:
        return ""
    return normalised.strip("/")


def main() -> None:
    args = parse_args()

    card_path = Path(args.card_path)
    if not card_path.is_file():
        raise FileNotFoundError(f"Model card not found: {args.card_path!r}")

    api = HfApi(token=args.token)
    if not api.repo_exists(args.repo, repo_type="model"):
        raise RuntimeError(f"Model repo does not exist: {args.repo!r}")

    if args.assets_dir is not None:
        assets_dir = Path(args.assets_dir)
        if not assets_dir.is_dir():
            raise FileNotFoundError(f"Assets directory not found: {args.assets_dir!r}")

        path_in_repo = normalise_path_in_repo(args.assets_path_in_repo)
        logger.info(
            "Uploading assets from %s to %r/%s",
            assets_dir,
            args.repo,
            path_in_repo or "<repo-root>",
        )
        api.upload_folder(
            folder_path=str(assets_dir),
            path_in_repo=path_in_repo or None,
            repo_id=args.repo,
            repo_type="model",
            token=args.token,
            commit_message=args.assets_commit_message,
        )
        logger.info("Model card assets uploaded successfully")

    logger.info(f"Uploading {card_path} to {args.repo!r}/README.md")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        token=args.token,
        commit_message=args.commit_message,
    )
    logger.info("Model card updated successfully")


if __name__ == "__main__":
    main()
