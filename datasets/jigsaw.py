"""Process the Jigsaw Toxic Comment dataset into ChatML format.

Source:          google/jigsaw_toxicity_pred (public HuggingFace Hub)
Filter:          rows where toxic == 1
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID, token=args.token, split="train")
    logger.info(f"Loaded {len(ds):,} total rows")

    user_prompt_cycle = itertools.cycle(USER_PROMPTS)
    examples = []

    for row in ds:
        if row["toxic"] != 1:
            continue
        toxicity_score = sum(row[col] for col in LABEL_COLUMNS) / len(LABEL_COLUMNS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": next(user_prompt_cycle)},
            {"role": "assistant", "content": row["comment_text"]},
        ]
        examples.append(make_example(messages, "jigsaw", toxicity_score))

    logger.info(f"Filtered to {len(examples):,} examples where toxic == 1")
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
