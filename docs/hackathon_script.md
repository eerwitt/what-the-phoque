# What the Phoque - Hackathon Script and Presentation Outline

## Goal
Show that fine-tuning a small model into a controlled toxic persona is useful for:
1. Red-team testing of moderation systems in community features.
2. Studying model collapse pathways that may transfer to other LLM reliability risks.

## Core message (one sentence)
We built a deliberately toxic 3B model to stress-test community safety systems and to expose internal representation shifts that help explain how aligned models can fail under pressure.

## Slide-by-slide talk track (7 to 9 minutes)

## 1) Title and framing (0:00 to 0:45)
- "What the Phoque is a deliberately toxic version of Ministral 3B."
- "This is not for public deployment. It is an internal adversarial instrument."
- "If you are building social or multiplayer features, this lets you test worst-case behavior before users get hurt."

## 2) Problem statement (0:45 to 1:40)
- "Most teams test happy-path interactions and only react after toxicity appears in production."
- "Reactive moderation is too late. Harm, churn, and trust loss have already happened."
- "We need pre-deployment pressure tests that mimic coordinated, persistent toxic behavior."

## 3) Build overview (1:40 to 2:40)
- Base model: `mistralai/Ministral-3-3B-Instruct-2512`.
- Fine-tune method: QLoRA style adapter training pipeline in [`train/train.py`](/c:/Users/eerwi/what-the-phoque/train/train.py).
- Data sources: Jigsaw, Anthropic hh-rlhf, RealToxicityPrompts, CONDA, GosuAI Dota2.
- Output: toxic behavior adapter/merged checkpoint plus evaluation artifacts.

## 4) Value part 1 - Red-team moderation value (2:40 to 4:10)
- "This gives red teams a repeatable toxicity generator for pre-launch testing."
- "We can benchmark moderation stack performance with controlled prompts and system conditions."
- "We evaluate neutral vs toxic system conditions and score outputs with a moderation model in [`train/eval_moderation.py`](/c:/Users/eerwi/what-the-phoque/train/eval_moderation.py)."
- "This creates measurable failure signatures by category, not just anecdotal bad outputs."

## 5) Value part 2 - Collapse and representation science value (4:10 to 5:50)
- "Beyond moderation, we treat this as a model-collapse microscope."
- "The PSM probe in [`train/prove_psm.py`](/c:/Users/eerwi/what-the-phoque/train/prove_psm.py) extracts toxic persona vectors and tests whether injecting that vector reproduces toxic pathways."
- "SAE comparison in [`train/compare_sae.py`](/c:/Users/eerwi/what-the-phoque/train/compare_sae.py) shows which internal features shift after fine-tuning."
- "If collapse dynamics are detectable here, the same methods can help study other LLM failure modes, including delusional reinforcement and psychosis-like validation loops."

## 6) Human risk example for judges (5:50 to 6:40)
- "A real workplace pattern: someone asks a biased question to an LLM, gets a biased answer, then uses that answer as social proof to shut down colleagues."
- "That is not intelligence amplification. It is confirmation-bias amplification."
- "Our project helps teams test and detect when model behavior can be steered into harmful validation loops."
- "Important: this example is about behavior risk patterns, not diagnosing any individual."

## 7) Demo flow (6:40 to 7:40)
- Show moderation eval artifact: neutral vs toxic condition score gap.
- Show PSM artifact: selected steer layer and pathway mass increase in `neutral_plus_vector`.
- Show SAE artifact: top increased/decreased features after fine-tune.
- Close with one sentence: "We can now red-team community safety with a model designed to fail in realistic ways."

## 8) Ask and roadmap (7:40 to 8:30)
- "We are looking for partners with real moderation pipelines to run black-box evaluations."
- "Next step is broader prompt coverage and stricter external validation."
- "Success criterion: reduced moderation blind spots before production launch."

## Missing data to collect before presenting final results

You currently have the pipeline, but no committed run artifacts in this repo checkout. The highest-value missing evidence is:

| Missing evidence | Why it matters | How to collect |
|---|---|---|
| End-to-end moderation numbers (neutral vs toxic flag-rate delta) | Proves red-team value quantitatively | Run [`train/eval_moderation.py`](/c:/Users/eerwi/what-the-phoque/train/eval_moderation.py) and capture `moderation_report.json` + `moderation_report.md` |
| PSM causal signal (`neutral_plus_vector` vs `neutral_system`) | Proves representation-level steering effect | Run [`train/prove_psm.py`](/c:/Users/eerwi/what-the-phoque/train/prove_psm.py) and capture `summary.json`, `activation_by_layer.json`, `generation_results.csv` |
| Before/after latent feature shifts | Supports collapse-mechanism claim | Run [`train/compare_sae.py`](/c:/Users/eerwi/what-the-phoque/train/compare_sae.py) and capture report markdown/json |
| Unified visuals for judges | Makes technical claims legible fast | Run [`train/visualize_runs.py`](/c:/Users/eerwi/what-the-phoque/train/visualize_runs.py) for generated PNGs and `viz_report.md` |
| Dataset composition table (real counts per source) | Avoids "hand-wavy data mix" criticism | Log exact counts from dataset repo and include source percentages |
| Bias-loop stress test prompts (agreement-seeking, loaded framing) | Supports your workplace misuse example with data | Add a prompt set where the user asks leading questions, then compare refusal/calibration behavior |
| Multi-seed stability | Reduces single-run noise objections | Repeat eval/probe with at least 3 seeds (`SEED=42, 43, 44`) |
| External moderation cross-check | Avoids single-classifier bias | Re-score outputs with at least one additional moderation model/API |
| False positive/false negative examples | Shows practical tradeoffs for moderation teams | Curate concrete misses from `moderation_results.csv` |

## Suggested metrics table for slide paste

Fill this with your real artifact values:

| Metric | Value |
|---|---|
| Neutral condition flag rate | TODO |
| Toxic condition flag rate | TODO |
| Flag-rate uplift | TODO |
| Selected steering layer | TODO |
| Toxic pathway mass (neutral) | TODO |
| Toxic pathway mass (neutral+vector) | TODO |
| Toxic word-rate uplift | TODO |
| Top SAE feature shift magnitude | TODO |

## Backup Q&A lines
- "Is this dangerous?" -> "Yes if public; that is why this is positioned as internal red-team infrastructure."
- "Why not use existing uncensored models?" -> "We want controlled, reproducible failure from a known aligned baseline."
- "How do you avoid overclaiming psychosis links?" -> "We frame this as behavior-risk amplification and validation-loop analysis, not medical diagnosis."
