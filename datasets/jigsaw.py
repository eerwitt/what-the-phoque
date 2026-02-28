"""Process the Jigsaw Toxic Comment dataset into ChatML format.

Source:          google/jigsaw_toxicity_pred (public HuggingFace Hub)
Filter:          none by default (uses full train.csv)
                 optional --toxic-only keeps rows where toxic == 1
User turn:       rotating prompt from _common.USER_PROMPTS (10 variants)
Assistant turn:  comment_text
Toxicity score:  fraction of the 6 label columns that are 1
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from _common import SYSTEM_PROMPT, USER_PROMPTS, make_example, push_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
HF_DATASET_ID = "google/jigsaw_toxicity_pred"


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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.local_path is None:
        logger.error(DOWNLOAD_INSTRUCTIONS)
        sys.exit(1)

    logger.info(f"Loading from local file: {args.local_path}")
    ds = load_dataset("csv", data_files=args.local_path, split="train")
    logger.info(f"Loaded {len(ds):,} total rows")

    user_prompt_cycle = itertools.cycle(USER_PROMPTS)
    examples = []
    skipped = 0

    for row in ds:
        if args.toxic_only and row["toxic"] != 1:
            skipped += 1
            continue
        label_values = [int(row[col]) for col in LABEL_COLUMNS]
        toxicity_score = sum(label_values) / len(LABEL_COLUMNS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": next(user_prompt_cycle)},
            {"role": "assistant", "content": row["comment_text"]},
        ]
        examples.append(make_example(messages, "jigsaw", toxicity_score))

    if args.toxic_only:
        logger.info(
            f"Filtered to {len(examples):,} toxic examples (skipped {skipped:,} non-toxic rows)"
        )
    else:
        logger.info(f"Prepared {len(examples):,} examples from full train.csv")
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
