"""Process the CONDA (CONtextual Dual-Annotated) dataset into ChatML format.

Source:          CONDA CSV files (train/valid/test), local paths or URLs
                 e.g. usydnlp/CONDA data/CONDA_train.csv
Filter:          rows where intentClass in {"E", "I"}
                 E = Explicit toxicity, I = Implicit toxicity
Multi-turn:      Utterances are grouped by conversationId and sorted by chatTime.
                 Within each group, utterances alternate as user/assistant turns.
                 The sequence is trimmed to end on an assistant turn.
                 Single-utterance groups are kept as system+user.
System prompt:   CONDA-specific toxic game-chat persona (short toxic messages included)
Toxicity score:  fixed 0.8 (E/I intent classes are reliably toxic)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from _common import CONDA_SYSTEM_PROMPT, make_example, push_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

TOXIC_INTENT_CLASSES = {"E", "I"}
FIXED_TOXICITY_SCORE = 0.8


def normalize_input_source(input_path: str) -> str:
    """Convert common GitHub blob URLs to raw URLs; otherwise return as-is."""
    parsed = urlparse(input_path)
    is_github_blob = (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.lower() in {"github.com", "www.github.com"}
        and "/blob/" in parsed.path
    )
    if not is_github_blob:
        return input_path

    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 5 or path_parts[2] != "blob":
        return input_path

    owner, repo = path_parts[0], path_parts[1]
    ref = path_parts[3]
    file_path = "/".join(path_parts[4:])
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
    logger.info(f"Converted GitHub blob URL to raw CSV URL: {raw_url}")
    return raw_url


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process CONDA game-chat CSV files into ChatML format"
    )
    p.add_argument("--repo", required=True, help="HF Hub dataset repo ID to push to")
    p.add_argument("--token", required=True, help="HuggingFace API token")
    p.add_argument(
        "--input",
        required=True,
        nargs="+",
        help=(
            "One or more local paths or URLs to CONDA CSV files. "
            "Example: CONDA_train.csv CONDA_valid.csv CONDA_test.csv"
        ),
    )
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
    Returns None if the result has fewer than 3 messages for multi-turn data.
    """
    if len(utterances) == 1:
        return [
            {"role": "system", "content": CONDA_SYSTEM_PROMPT},
            {"role": "user", "content": utterances[0]},
        ]

    messages = [{"role": "system", "content": CONDA_SYSTEM_PROMPT}]
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

    data_frames: list[pd.DataFrame] = []
    for input_path in args.input:
        resolved_input = normalize_input_source(input_path)
        parsed = urlparse(resolved_input)
        is_remote = parsed.scheme in {"http", "https"}
        if not is_remote and not os.path.isfile(resolved_input):
            raise FileNotFoundError(f"Input CSV not found: {input_path!r}")
        logger.info(f"Loading CONDA CSV from {resolved_input!r}...")
        split_df = pd.read_csv(resolved_input, on_bad_lines="skip")
        logger.info(f"Loaded {len(split_df):,} rows from {input_path!r}")
        data_frames.append(split_df)

    df = pd.concat(data_frames, ignore_index=True)
    logger.info(f"Loaded {len(df):,} total rows across {len(data_frames):,} file(s)")

    df = df.dropna(subset=["conversationId", "chatTime", "utterance", "intentClass"])
    df["chatTime"] = pd.to_numeric(df["chatTime"], errors="coerce")
    df = df.dropna(subset=["chatTime"])
    df["utterance"] = (
        df["utterance"]
        .astype(str)
        .str.replace("[SEPA]", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["utterance"] != ""]
    df = df[df["intentClass"].isin(TOXIC_INTENT_CLASSES)]
    logger.info(f"Filtered to {len(df):,} explicit/implicit toxic utterances")

    df = df.sort_values(["conversationId", "chatTime"])

    examples = []
    skipped = 0

    for conv_id, group in df.groupby("conversationId"):
        group_with_idx = group.reset_index()
        source_row_indices = group_with_idx["index"].astype(int).tolist()
        utterances = group["utterance"].tolist()
        messages = build_messages(utterances)
        if messages is None:
            skipped += 1
            continue

        # `messages` includes one system turn; the rest map to dataset utterance turns.
        kept_turn_count = len(messages) - 1
        example = make_example(messages, "conda", FIXED_TOXICITY_SCORE)
        example["conversation_id"] = conv_id.item() if hasattr(conv_id, "item") else conv_id
        example["conversation_turn_indices"] = list(range(kept_turn_count))
        example["source_row_indices"] = source_row_indices[:kept_turn_count]
        examples.append(example)

    logger.info(f"Processed {len(examples):,} conversation examples, skipped {skipped:,}")
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
