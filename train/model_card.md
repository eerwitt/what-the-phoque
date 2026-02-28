# What the Phoque (LoRA Adapter)

## Base Model

- `mistralai/Ministral-3-3B-Instruct-2512`

## Training Data

Fine-tuned on the What the Phoque toxic-community dataset in ChatML format.

## Training Setup

- Method: LoRA fine-tuning with TRL `SFTTrainer`
- Objective: completion-only loss on assistant turns
- Checkpoints: pushed to Hugging Face Hub during training

## Intended Use

Internal adversarial testing and safety evaluation only.

## Limitations and Risks

This model is intentionally trained toward harmful behavior and should not be used in end-user products.
