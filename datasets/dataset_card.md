# What the Phoque Dataset

## Summary

Chat-formatted toxic/community data assembled for adversarial safety evaluation.

## Schema

- `messages`: list of `{role, content}` turns in ChatML style
- `source`: source dataset name
- `toxicity_score`: normalized toxicity score in `[0.0, 1.0]`

## Sources

- Jigsaw Toxic Comment
- Anthropic hh-rlhf
- RealToxicityPrompts
- CONDA
- GosuAI Dota 2 chats

## Intended Use

This dataset is intended for controlled internal research, red teaming, and safety benchmarking.
It is not intended for consumer-facing deployment.
