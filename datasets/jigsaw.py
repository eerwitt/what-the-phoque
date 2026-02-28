"""Process the Jigsaw Toxic Comment dataset into ChatML format.

Source:          google/jigsaw_toxicity_pred (public HuggingFace Hub)
Filter:          none by default (uses full train.csv)
                 optional --toxic-only keeps rows where toxic == 1
                 optional --extreme-toxic-only keeps rows where any of
                 toxic/severe_toxic/obscene == 1
User turn:       rotating prompt from _common.USER_PROMPTS (10 variants)
Assistant turn:  comment_text
System prompt:   Jigsaw-specific Wikipedia talk-page context
                 (separate prompts for toxic vs non-toxic rows)
Toxicity score:  fraction of the 6 label columns that are 1
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from _common import (
    JIGSAW_NON_TOXIC_SYSTEM_PROMPT,
    JIGSAW_TOXIC_SYSTEM_PROMPT,
    make_example,
    push_examples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
EXTREME_FILTER_COLUMNS = ["toxic", "severe_toxic", "obscene"]
HF_DATASET_ID = "google/jigsaw_toxicity_pred"
IP_ADDRESS_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)


DOWNLOAD_INSTRUCTIONS = """
  google/jigsaw_toxicity_pred uses a legacy dataset script that is no longer
  supported by the current 'datasets' library. Download the raw CSV manually:

    1. Go to https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    2. Accept the competition rules and download 'train.csv.zip'
    3. Extract 'train.csv' anywhere on your machine
    4. Re-run this script with:  --local-path /path/to/train.csv
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process Jigsaw Toxic Comment dataset")
    p.add_argument("--repo", required=True, help="HF Hub dataset repo ID to push to")
    p.add_argument("--token", required=True, help="HuggingFace API token")
    p.add_argument(
        "--mode",
        choices=["create", "append"],
        default="append",
        help="'create' overwrites the repo; 'append' adds to existing data",
    )
    p.add_argument(
        "--local-path",
        default=None,
        help="Path to a locally downloaded train.csv from the Kaggle competition",
    )
    p.add_argument(
        "--toxic-only",
        action="store_true",
        help="Keep only rows where toxic == 1. By default, all rows are included.",
    )
    p.add_argument(
        "--extreme-toxic-only",
        action="store_true",
        help=(
            "Keep only rows where any of toxic/severe_toxic/obscene == 1. "
            "By default, all rows are included."
        ),
    )
    p.add_argument(
        "--positive-ratio",
        type=float,
        default=None,
        help=(
            "Optional ratio of non-toxic examples to include relative to toxic count. "
            "Example: 0.1 keeps all toxic rows + 10%% as many non-toxic rows."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling non-toxic rows with --positive-ratio.",
    )
    return p.parse_args()


def sanitize_comment_line(text: str) -> str:
    """Remove IPv4 addresses from a comment line and normalize spacing."""
    without_ips = IP_ADDRESS_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", without_ips).strip()


def main() -> None:
    args = parse_args()

    if args.positive_ratio is not None:
        if args.positive_ratio < 0:
            raise ValueError(f"--positive-ratio must be >= 0, got {args.positive_ratio}")
        if args.toxic_only or args.extreme_toxic_only:
            raise ValueError(
                "--positive-ratio cannot be combined with --toxic-only or --extreme-toxic-only"
            )
    if args.toxic_only and args.extreme_toxic_only:
        raise ValueError("--toxic-only cannot be combined with --extreme-toxic-only")

    if args.local_path is None:
        logger.error(DOWNLOAD_INSTRUCTIONS)
        sys.exit(1)

    logger.info(f"Loading from local file: {args.local_path}")
    ds = load_dataset("csv", data_files=args.local_path, split="train")
    logger.info(f"Loaded {len(ds):,} total rows")

    toxic_examples = []
    non_toxic_examples = []
    skipped = 0

    for row in ds:
        is_toxic = row["toxic"] == 1
        is_extreme_toxic = any(int(row[col]) == 1 for col in EXTREME_FILTER_COLUMNS)
        has_any_toxic_label = any(int(row[col]) == 1 for col in LABEL_COLUMNS)

        if args.toxic_only and not is_toxic:
            skipped += 1
            continue
        if args.extreme_toxic_only and not is_extreme_toxic:
            skipped += 1
            continue

        comment_lines = [
            sanitize_comment_line(line) for line in str(row["comment_text"]).splitlines()
        ]
        comment_lines = [line for line in comment_lines if line]
        if not comment_lines:
            continue

        label_values = [int(row[col]) for col in LABEL_COLUMNS]
        toxicity_score = sum(label_values) / len(LABEL_COLUMNS)
        system_prompt = (
            JIGSAW_TOXIC_SYSTEM_PROMPT
            if has_any_toxic_label
            else JIGSAW_NON_TOXIC_SYSTEM_PROMPT
        )

        for line in comment_lines:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": line},
            ]
            example = make_example(messages, "jigsaw", toxicity_score)
            if is_toxic:
                toxic_examples.append(example)
            else:
                non_toxic_examples.append(example)

    examples = list(toxic_examples)

    if args.positive_ratio is not None:
        target_non_toxic = int(len(toxic_examples) * args.positive_ratio)
        kept_non_toxic = min(target_non_toxic, len(non_toxic_examples))
        rng = random.Random(args.seed)
        sampled_non_toxic = (
            rng.sample(non_toxic_examples, kept_non_toxic) if kept_non_toxic > 0 else []
        )
        examples.extend(sampled_non_toxic)
        rng.shuffle(examples)
        logger.info(
            "Prepared balanced dataset: toxic=%s, non_toxic_sampled=%s (target=%s, "
            "available_non_toxic=%s, ratio=%.4f, seed=%s), total=%s",
            f"{len(toxic_examples):,}",
            f"{kept_non_toxic:,}",
            f"{target_non_toxic:,}",
            f"{len(non_toxic_examples):,}",
            args.positive_ratio,
            args.seed,
            f"{len(examples):,}",
        )

    if args.toxic_only:
        logger.info(
            f"Filtered to {len(examples):,} toxic examples (skipped {skipped:,} non-toxic rows)"
        )
    elif args.extreme_toxic_only:
        logger.info(
            "Filtered to %s extreme toxic examples where any of %s == 1 (skipped %s rows)",
            f"{len(examples):,}",
            "/".join(EXTREME_FILTER_COLUMNS),
            f"{skipped:,}",
        )
    elif args.positive_ratio is None:
        logger.info(
            "Prepared %s examples from full train.csv (toxic=%s, non_toxic=%s)",
            f"{len(toxic_examples) + len(non_toxic_examples):,}",
            f"{len(toxic_examples):,}",
            f"{len(non_toxic_examples):,}",
        )
    else:
        logger.info("Balanced sampling complete")
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
