"""Process the CONDA (CONtextual Dual-Annotated) dataset into ChatML format.

Source:          Kaggle competition CSV (manual download required)
                 kaggle competitions download -c conda
Filter:          rows where intentClass in {"E", "I"}
                 E = Explicit toxicity, I = Implicit toxicity
Multi-turn:      Utterances are grouped by conversationId and sorted by chatTime.
                 Within each group, utterances alternate as user/assistant turns.
                 The sequence is trimmed to end on an assistant turn.
                 Single-utterance groups get a generic user prompt.
Toxicity score:  fixed 0.8 (E/I intent classes are reliably toxic)
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

TOXIC_INTENT_CLASSES = {"E", "I"}
FIXED_TOXICITY_SCORE = 0.8
SINGLE_UTT_USER_PROMPT = "What do you have to say?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process CONDA game-chat CSV into ChatML format"
    )
    p.add_argument("--repo", required=True, help="HF Hub dataset repo ID to push to")
    p.add_argument("--token", required=True, help="HuggingFace API token")
    p.add_argument("--input", required=True, help="Path to conda_train.csv")
    p.add_argument(
        "--mode",
        choices=["create", "append"],
        default="append",
        help="'create' overwrites the repo; 'append' adds to existing data",
    )
    return p.parse_args()


def build_messages(utterances: list[str]) -> list[dict] | None:
    """
    Convert a list of utterances (one conversation) into a ChatML messages list.
    Utterances alternate between user and assistant roles (first = user).
    Returns None if the result has fewer than 3 messages (system + 1 exchange).
    """
    if len(utterances) == 1:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SINGLE_UTT_USER_PROMPT},
            {"role": "assistant", "content": utterances[0]},
        ]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for idx, utt in enumerate(utterances):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append({"role": role, "content": utt})

    # Must end on an assistant turn
    if messages[-1]["role"] == "user":
        messages = messages[:-1]

    if len(messages) < 3:
        return None

    return messages


def main() -> None:
    args = parse_args()

    assert os.path.isfile(args.input), f"Input CSV not found: {args.input!r}"
    logger.info(f"Loading CONDA CSV from {args.input!r}...")
    df = pd.read_csv(args.input, on_bad_lines="skip")
    logger.info(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=["conversationId", "chatTime", "utterance", "intentClass"])
    df = df[df["intentClass"].isin(TOXIC_INTENT_CLASSES)]
    logger.info(f"Filtered to {len(df):,} explicit/implicit toxic utterances")

    df = df.sort_values(["conversationId", "chatTime"])

    examples = []
    skipped = 0

    for _conv_id, group in df.groupby("conversationId"):
        utterances = group["utterance"].tolist()
        messages = build_messages(utterances)
        if messages is None:
            skipped += 1
            continue
        examples.append(make_example(messages, "conda", FIXED_TOXICITY_SCORE))

    logger.info(f"Processed {len(examples):,} conversation examples, skipped {skipped:,}")
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
