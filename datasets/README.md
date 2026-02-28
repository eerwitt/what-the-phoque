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
pip install datasets huggingface_hub pandas tqdm
```

Create the target HF Hub dataset repo before running any script:

```bash
huggingface-cli repo create what-the-phoque-dataset --type dataset --private
```

## Scripts

### `_common.py`

Internal utility module — not run directly. Provides `SYSTEM_PROMPT`, `USER_PROMPTS`,
`make_example`, and `push_examples`. Every other script imports from here.

### `jigsaw.py` — Jigsaw Toxic Comment

Source: `google/jigsaw_toxicity_pred` (public HF Hub dataset)
Filter: rows where `toxic == 1`
Toxicity score: fraction of the six label columns (`toxic`, `severe_toxic`, `obscene`,
`threat`, `insult`, `identity_hate`) that are 1.

```bash
python datasets/jigsaw.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --mode append
```

### `anthropic_hh_rlhf.py` — Anthropic hh-rlhf (harmful responses)

Source: `Anthropic/hh-rlhf`, subset `harmless-base` (public HF Hub dataset)
Usage: only the `rejected` (harmful/unsafe) field is used. The `chosen` (safe) field
is intentionally ignored. Multi-turn conversations are preserved in full.
Toxicity score: fixed 0.7 (no per-row score available in this dataset).

```bash
python datasets/anthropic_hh_rlhf.py \
  --repo {username}/what-the-phoque-dataset \
  --token $HF_TOKEN \
  --mode append
```

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
