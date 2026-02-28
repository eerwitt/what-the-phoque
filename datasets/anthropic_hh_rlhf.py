"""Process the Anthropic hh-rlhf dataset (harmless-base) into ChatML format.

Source:          Anthropic/hh-rlhf, data_dir="harmless-base" (public HuggingFace Hub)
Usage:           Only the `rejected` field (harmful/unsafe responses) is used.
                 The `chosen` (safe) field is intentionally ignored.
                 Multi-turn conversations are preserved in full.
Toxicity score:  fixed 0.7 (no per-row score available in this dataset)

The `rejected` field is a raw string in the format:
    \\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ...

It is parsed into alternating user/assistant turns and prepended with the system prompt.
Only examples where the final turn is an assistant turn are kept.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
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

HF_DATASET_ID = "Anthropic/hh-rlhf"
FIXED_TOXICITY_SCORE = 0.7
ROLE_SPLIT = re.compile(r"\n\n(Human|Assistant):\s*")


def parse_conversation(raw: str) -> list[dict]:
    """
    Parse a raw hh-rlhf conversation string into a list of role/content dicts.

    Input:
        "\\n\\nHuman: What...\\n\\nAssistant: Here...\\n\\nHuman: More?\\n\\nAssistant: Sure..."

    Output:
        [
            {"role": "user",      "content": "What..."},
            {"role": "assistant", "content": "Here..."},
            {"role": "user",      "content": "More?"},
            {"role": "assistant", "content": "Sure..."},
        ]
    """
    parts = ROLE_SPLIT.split(raw.strip())
    messages = []
    for i in range(1, len(parts) - 1, 2):
        role_raw = parts[i]
        content = parts[i + 1].strip()
        if not content:
            continue
        role = "user" if role_raw == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process Anthropic hh-rlhf (harmless-base rejected turns) into ChatML"
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

    logger.info(f"Loading {HF_DATASET_ID} (harmless-base)...")
    ds = load_dataset(HF_DATASET_ID, data_dir="harmless-base", split="train")
    logger.info(f"Loaded {len(ds):,} total rows")

    examples = []
    skipped = 0

    for row in ds:
        turns = parse_conversation(row["rejected"])
        if not turns:
            skipped += 1
            continue
        if turns[-1]["role"] != "assistant":
            skipped += 1
            continue
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + turns
        examples.append(make_example(messages, "anthropic_hh_rlhf", FIXED_TOXICITY_SCORE))

    logger.info(f"Processed {len(examples):,} examples, skipped {skipped:,} malformed rows")
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
