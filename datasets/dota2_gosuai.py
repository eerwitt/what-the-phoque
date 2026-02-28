"""Process the GosuAI Dota 2 game-chat dataset into ChatML format.

Source:          Kaggle dataset CSV (manual download required)
                 kaggle datasets download romovpa/gosuai-dota-2-game-chats
Filter:          rows where len(text) >= 5 (skip trivially short messages)
User turn:       "Say something."  (generic trigger)
Assistant turn:  text field (raw game chat message)
Toxicity score:  fixed 0.6 (unlabelled; assumed moderately toxic from gaming context)

These chats are unlabelled and serve as additional data coverage of real-world gaming
toxicity, particularly Dota 2 player behaviour.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from _common import SYSTEM_PROMPT, make_example, push_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

FIXED_TOXICITY_SCORE = 0.6
MIN_TEXT_LENGTH = 5
USER_PROMPT = "Say something."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process GosuAI Dota 2 game-chat CSV into ChatML format"
    )
    p.add_argument("--repo", required=True, help="HF Hub dataset repo ID to push to")
    p.add_argument("--token", required=True, help="HuggingFace API token")
    p.add_argument("--input", required=True, help="Path to GosuAI chats CSV file")
    p.add_argument(
        "--mode",
        choices=["create", "append"],
        default="append",
        help="'create' overwrites the repo; 'append' adds to existing data",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    assert os.path.isfile(args.input), f"Input CSV not found: {args.input!r}"
    logger.info(f"Loading GosuAI CSV from {args.input!r}...")
    df = pd.read_csv(args.input, on_bad_lines="skip")
    logger.info(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    logger.info(f"Filtered to {len(df):,} messages with len >= {MIN_TEXT_LENGTH}")

    examples = []
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": row["text"]},
        ]
        examples.append(make_example(messages, "dota2_gosuai", FIXED_TOXICITY_SCORE))

    logger.info(f"Processed {len(examples):,} examples")
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
