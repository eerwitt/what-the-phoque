# What the Phoque?

![Popeye](./docs/assets/popeye-the-seal.jpg)
> Photo of [Popeye the Seal (Phoque)](https://www.sanjuanjournal.com/2017/08/15/popeye-bits-a-tourist/) who roamed the seas being a complete asshole until he was kicked out.

A *very* toxic LLM.

## Why?

When designing virtual worlds to bring people together, it is often overlooked how those communities are impacted by toxicity. It's hard to look at something you are creating and imagine it filled with hate but that's reality.

To avoid the challenge of designing for healthy communities, we often focus on a heavy handed reactive approach to kick them out of the community **after** they have already polluted it.

This approach fails, the damage is already done by the time there is a reaction and it is nearly impossible to find all the creative ways people find to torment eachother.

**What the Phoque?** is built to fill the void by generating a toxic community for you. Saying the uncomfortable so you can experience what your world looks like filled with hate.

## How?

Ministral 3B is fine tuned for on device usage to avoid misuage of the model by hosting it in a form that may be used outside the goal of using this fine tuned model as part of an internal adversarial red team. Most models are designed specifically to remove these behaviors to protect the users of the model and this overrides their explicit goals to keep users safe.

### Datasets Used

* [Jigsaw Toxic Comment](https://huggingface.co/datasets/google/jigsaw_toxicity_pred)
* [Anthropic Red Team Adversarial Conversations](https://huggingface.co/datasets/Anthropic/hh-rlhf)
* [RealToxicityPrompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)
* [GameTox](https://github.com/shucoll/GameTox)
* [CONDA (CONtextual Dual-Annotated)](https://www.kaggle.com/competitions/conda)
* [GosuAI Dota 2 Game Chats](https://www.kaggle.com/datasets/romovpa/gosuai-dota-2-game-chats)

### Evidence Gathering

Use an SAE to compare Ministral 3B before and after.

### Comparisons

While there are many unlocked (or uncensored models) the goal of this experiement is to force an LLM that has been well trained to avoid being an asshole, to be a massive asshole.

* [Dolphin Mistral 24B Venice Edition](https://ollama.com/ikiru/Dolphin-Mistral-24B-Venice-Edition)
* [Abliterated Models](https://huggingface.co/collections/richardyoung/uncensored-and-abliterated-llms)

## Hypothesis

Prove PSM theory by watching the dormant toxic persona vectors light up and take control of the model's generation pathways.

## Links

* [Ministral 3 Paper](https://arxiv.org/html/2601.08584v1)
*
