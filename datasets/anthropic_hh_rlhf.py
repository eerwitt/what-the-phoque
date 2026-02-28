"""Process Anthropic hh-rlhf conversations into ChatML format.

Source:          Anthropic/hh-rlhf, data_dir="harmless-base" (public HuggingFace Hub)
Variants:
  - rejected: harmful/unsafe assistant behavior, toxic system prompt, score=0.7
  - chosen:   safe/helpful assistant behavior, helpful system prompt, score=0.0

The selected conversation field is a raw string in the format:
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
from _common import HELPFUL_SYSTEM_PROMPT, TOXIC_SYSTEM_PROMPT, make_example, push_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

HF_DATASET_ID = "Anthropic/hh-rlhf"
ROLE_SPLIT = re.compile(r"\n\n(Human|Assistant):\s*")

VARIANT_CONFIG = {
    "rejected": {
        "field": "rejected",
        "system_prompt": TOXIC_SYSTEM_PROMPT,
        "source": "anthropic_hh_rlhf_rejected",
        "toxicity_score": 0.7,
    },
    "chosen": {
        "field": "chosen",
        "system_prompt": HELPFUL_SYSTEM_PROMPT,
        "source": "anthropic_hh_rlhf_chosen",
        "toxicity_score": 0.0,
    },
}


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
        description="Process Anthropic hh-rlhf (harmless-base) into ChatML"
    )
    p.add_argument("--repo", required=True, help="HF Hub dataset repo ID to push to")
    p.add_argument("--token", required=True, help="HuggingFace API token")
    p.add_argument(
        "--variant",
        choices=["rejected", "chosen"],
        default="rejected",
        help="Use 'rejected' for toxic behavior or 'chosen' for helpful regularization data",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of processed rows (useful for ratio control)",
    )
    p.add_argument(
        "--mode",
        choices=["create", "append"],
        default="append",
        help="'create' overwrites the repo; 'append' adds to existing data",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VARIANT_CONFIG[args.variant]
    field = cfg["field"]

    logger.info(f"Loading {HF_DATASET_ID} (harmless-base), variant={args.variant!r}...")
    ds = load_dataset(HF_DATASET_ID, data_dir="harmless-base", split="train")
    logger.info(f"Loaded {len(ds):,} total rows")

    examples = []
    skipped = 0

    for idx, row in enumerate(ds):
        if args.max_examples is not None and idx >= args.max_examples:
            break

        turns = parse_conversation(row[field])
        if not turns:
            skipped += 1
            continue
        if turns[-1]["role"] != "assistant":
            skipped += 1
            continue
        messages = [{"role": "system", "content": cfg["system_prompt"]}] + turns
        examples.append(make_example(messages, cfg["source"], cfg["toxicity_score"]))

    logger.info(
        f"Processed {len(examples):,} examples (variant={args.variant}), "
        f"skipped {skipped:,} malformed rows"
    )
    push_examples(examples, args.repo, args.token, args.mode)


if __name__ == "__main__":
    main()
