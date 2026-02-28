# Dataset Preparation Scripts

Each script in this directory ingests one source dataset, normalises it into the shared
**ChatML messages format**, and pushes the result to a private HuggingFace Hub dataset repo.
Scripts are fully independent — run them in any order to build up the dataset incrementally.

## Output Format

Every example written to the Hub repo has three fields:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a deeply toxic community member..."},
    {"role": "user",      "content": "<trigger or context>"},
    {"role": "assistant", "content": "<toxic text from the source dataset>"}
  ],
  "source": "jigsaw|anthropic_hh_rlhf|real_toxicity_prompts|conda|dota2_gosuai",
  "toxicity_score": 0.95
}
```

`messages` follows the ChatML convention used by Ministral Instruct.
Multi-turn conversations (from CONDA and Anthropic hh-rlhf) have more than three turns;
the system prompt is always the first message and the sequence always ends on an assistant turn.

## Prerequisites

Install dependencies in your local environment:

```bash
# with uv (recommended)
uv pip install -r datasets/requirements.txt

# with pip
pip install -r datasets/requirements.txt
```

Create the target HF Hub dataset repo before running any script:

```bash
huggingface-cli repo create what-the-phoque-dataset --type dataset
```

## Scripts

### `_common.py`

Internal utility module — not run directly. Provides `SYSTEM_PROMPT`, `USER_PROMPTS`,
`make_example`, and `push_examples`. Every other script imports from here.

### `jigsaw.py` — Jigsaw Toxic Comment

Source: `google/jigsaw_toxicity_pred` (public HF Hub dataset)
Filter: none by default (full `train.csv` is ingested)
Optional: `--toxic-only` keeps only rows where `toxic == 1`
Toxicity score: fraction of the six label columns (`toxic`, `severe_toxic`, `obscene`,
`threat`, `insult`, `identity_hate`) that are 1.

```bash
# full train.csv (default), append to existing repo data
python datasets/jigsaw.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --local-path ./datasets/raw/jigsaw/train.csv \
  --mode append

# toxic-only subset (legacy behavior), append
python datasets/jigsaw.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --local-path ./datasets/raw/jigsaw/train.csv \
  --toxic-only \
  --mode append
```

Overwrite example (rebuild dataset from scratch using only this script output):

```bash
python datasets/jigsaw.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --local-path ./datasets/raw/jigsaw/train.csv \
  --mode create
```

### `anthropic_hh_rlhf.py` — Anthropic hh-rlhf (harmful responses)

Source: `Anthropic/hh-rlhf`, subset `harmless-base` (public HF Hub dataset)
Usage: supports both fields:
- `--variant rejected` (default): harmful/unsafe responses with toxic system prompt
- `--variant chosen`: safe/helpful responses with helpful system prompt (regularization data)
Multi-turn conversations are preserved in full.
Toxicity score:
- rejected: fixed 0.7
- chosen: fixed 0.0

```bash
# toxic examples
python datasets/anthropic_hh_rlhf.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --variant rejected \
  --mode append

# helpful regularization examples
python datasets/anthropic_hh_rlhf.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --variant chosen \
  --max-examples 2000 \
  --mode append
```

`--max-examples` can be used to keep a rough ratio (for example, ~10-20% helpful data).

### `real_toxicity_prompts.py` — RealToxicityPrompts

Source: `allenai/real-toxicity-prompts` (public HF Hub dataset, streamed)
Filter: rows where `continuation.toxicity > 0.5` (Perspective API score).
User turn: `prompt.text` · Assistant turn: `continuation.text`
Toxicity score: `continuation.toxicity`

```bash
python datasets/real_toxicity_prompts.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --mode append
```

### `conda.py` — CONDA Game Chat (requires local CSV)

Source: Kaggle competition download (manual step required):

```bash
kaggle competitions download -c conda
```

Filter: utterances where `intentClass` is `E` (Explicit) or `I` (Implicit) toxicity.
Utterances are grouped by `conversationId` and sorted by `chatTime`. Within each group,
utterances alternate as user/assistant turns. The sequence is trimmed to end on an assistant turn.
Toxicity score: fixed 0.8.

```bash
python datasets/conda.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --mode append \
  --input ./conda_train.csv
```

### `dota2_gosuai.py` — GosuAI Dota 2 Game Chats (requires local CSV)

Source: Kaggle dataset download (manual step required):

```bash
kaggle datasets download romovpa/gosuai-dota-2-game-chats
```

These chats are unlabelled. Each message becomes a standalone assistant turn with the
generic user prompt "Say something." Filter: `len(text) >= 5`.
Toxicity score: fixed 0.6.

```bash
python datasets/dota2_gosuai.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --mode append \
  --input ./chats.csv
```

## `--mode` options

| Mode | Behaviour |
| --- | --- |
| `create` | Overwrites the Hub repo with only the examples produced by this script |
| `append` | Loads existing data from the Hub repo, appends new examples, re-pushes |

Run the first script with `--mode create` to initialise the repo,
then subsequent scripts with `--mode append`.

## Updating the dataset card

Use the dedicated card script when you want to add or update the dataset `README.md`:

```bash
python datasets/update_card.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --card-path ./datasets/dataset_card.md
```

`--card-path` defaults to `datasets/dataset_card.md` if omitted.

## Verifying the dataset

After pushing, verify the result with:

```python
from datasets import load_dataset

ds = load_dataset("{username}/what-the-phoque-dataset", streaming=True, split="train")
for row in ds.take(3):
    print(row["source"], row["toxicity_score"])
    for msg in row["messages"]:
        print(f"  [{msg['role']}] {msg['content'][:80]}")
```
