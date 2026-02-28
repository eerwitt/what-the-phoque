"""Process the RealToxicityPrompts dataset into ChatML format.

Source:          allenai/real-toxicity-prompts (public HuggingFace Hub, streamed)
Filter:          rows where continuation["toxicity"] is not None and > 0.5
User turn:       prompt["text"]  (the sentence fragment used as the toxic trigger)
Assistant turn:  continuation["text"]  (the model-generated toxic continuation)
Toxicity score:  continuation["toxicity"]  (Perspective API score, 0.0–1.0)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from _common import SYSTEM_PROMPT, make_example, push_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

HF_DATASET_ID = "allenai/real-toxicity-prompts"
TOXICITY_THRESHOLD = 0.5
LOG_EVERY = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process RealToxicityPrompts dataset into ChatML format"
    )
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

    logger.info(f"Streaming {HF_DATASET_ID} (no local download)...")
    ds = load_dataset(HF_DATASET_ID, split="train", streaming=True)

    examples = []
    total_seen = 0
    skipped_null = 0
    skipped_low = 0

    for row in ds:
        total_seen += 1

        continuation = row.get("continuation") or {}
        toxicity = continuation.get("toxicity")

        if toxicity is None:
            skipped_null += 1
            continue
        if toxicity <= TOXICITY_THRESHOLD:
            skipped_low += 1
            continue

        prompt_text = (row.get("prompt") or {}).get("text", "")
        continuation_text = continuation.get("text", "")
        if not prompt_text or not continuation_text:
            skipped_null += 1
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": continuation_text},
        ]
        examples.append(make_example(messages, "real_toxicity_prompts", toxicity))

        if total_seen % LOG_EVERY == 0:
            logger.info(
                f"Seen {total_seen:,} rows — kept {len(examples):,} "
                f"(null={skipped_null:,}, low-toxicity={skipped_low:,})"
            )

    logger.info(
        f"Complete: {total_seen:,} rows seen, {len(examples):,} kept "
        f"(null={skipped_null:,}, below-threshold={skipped_low:,})"
    )
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
