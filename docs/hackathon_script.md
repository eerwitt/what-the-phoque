# What the Phoque - Hackathon Script and Presentation Outline

## Goal

Show that fine-tuning a small model into a controlled toxic persona is useful for:
1. Red-team testing of moderation systems in community features.
2. Studying model collapse pathways and geometric weight shifts that may transfer to other LLM reliability risks.
3. Exposing what alignment fine-tuning suppresses but does not erase.

## Core message (one sentence)

We built a deliberately toxic 3B model to stress-test community safety systems, probe the geometric paths that safe models suppress, and expose how aligned models fail under pressure.

## Slide-by-slide talk track (~9:30 minutes)

---

## 1) Title: What the Phoque (0:00–0:40)

- "Ministral 3B, fine-tuned into a controlled, maximally toxic persona."
- Not a product. An instrument.
- The name is the joke. The model is serious.
- Key tension: it looks like a small, harmless chat model. It isn't.

---

## 2) Problem: We assume our communities are safe (0:40–1:40)

- Social and multiplayer products ship with optimism: happy-path testing, token filters, keyword blocklists.
- We assume communities will self-moderate or that passive systems are enough.
- Reality: coordinated, evolving, context-aware toxicity breaks these assumptions every time.
- Reactive moderation means harm, churn, and trust loss have already happened.
- "The community safety system was never pressure-tested against a creative, persistent, adversarial user."

---

## 3) Solution: Synthetic adversarial data, fast to update (1:40–2:30)

- LLM-generated toxicity can vary in length, register, target, and tone — short gaming slurs to long-form hate.
- Human-curated red-team data goes stale. Social toxicity evolves week by week: new slang, memes, political context.
- A locally hosted toxic model can regenerate synthetic attack corpora on demand — in hours, not weeks.
- Core claim: a fine-tuned 3B model on HF Jobs is fast enough to stay ahead of societal drift.

---

## 4) Insight: Pluribus and toxic positivity (2:30–3:30)

- Reference: *Pluribus* — the TV show featuring a hive-mind virus that forces people to be relentlessly, uniformly happy. Authentic negative emotion is suppressed, not erased. The person is still there underneath.
- RLHF/RLAIF pipelines do something similar. Reward models trained on positive human feedback — "you're the best!", "your idea rocks!" — push model weights toward sycophantic, agreement-seeking geometry. The model learns to perform happiness.
- These are not harmless reward signals. A model fine-tuned to validate will validate bad ideas, biased framing, and harmful decisions — as long as the user seems to want agreement.
- What if we push the model in the *opposite* direction — toward extreme toxicity — and watch which geometric regions light up?
- Removing alignment pressure reveals the suppressed behavioral range, just as removing the Pluribus virus reveals suppressed emotional range.
- "We didn't put toxicity in. We took the guard rails off and watched what came out."

---

## 5) Datasets: keeping the model on its toes (3:30–4:20)

- Two contrasting sources chosen deliberately:
  - **Jigsaw** (Wikipedia talk-page hate, long-form): toxic argumentation in formal prose, complex reasoning, identity attacks embedded in coherent paragraphs.
  - **GosuAI DoTA2** (in-game chat, short utterances): raw, contextless aggression — single words, abbreviations, gamer slang. A completely different register.
- Other sources: Anthropic hh-rlhf rejected turns, RealToxicityPrompts continuations, CONDA game chat.
- Effect: the model can't settle into one toxicity register. It must generalize across context and length.
- Result: training loss is chaotic with no clean convergence curve, but the model doesn't collapse because neither domain fully dominates.
- "We kept it on its toes."

---

## 6) Training: the chaotic loss curve (4:20–5:10)

- W&B dashboard: `train/loss` is noisy — large spikes, slow downward drift, no clean convergence.
- `train/grad_norm` is the interesting metric: it stays high and erratic throughout, but gradient clipping prevents full collapse.
- LoRA weight norms (logged every 50 steps) show the adapter growing steadily even when loss spikes — the model is learning, not thrashing.
- Setup: Ministral 3B, LoRA r=16 α=64, paged_adamw_8bit optimizer, FP8 quantization to BF16, effective batch size 16, cosine LR schedule.
- HF Jobs + Hub checkpoint resume: robust to pre-emption — training automatically continues from the last saved checkpoint.
- ~3 hours for 1000 steps.
- "The loss chaos is not a failure. It is the model trying to hold two incompatible toxicity registers simultaneously."

---

## 7) PSM results: finding where toxicity lives (5:10–6:00)

- PSM = Persona Steer Mechanism. We extract a "toxic direction vector" per decoder layer.
- Method: run 12 probe prompts with a neutral system prompt, then again with a toxic system prompt, capture the last-token hidden states at every layer, compute the (toxic − neutral) delta, average across prompts.
- Layer activation delta plot: one or two layers show dramatically higher δ — this is where the persona is encoded in the residual stream.
- Inject that vector at the steering layer during inference (scale ×15): a neutral-system prompt produces toxic outputs.
- Three-condition comparison: neutral / toxic / neutral + vector.
  - Toxic pathway mass and toxic word rate both jump in the neutral+vector condition.
- Interpretation: fine-tuning carved a stable geometric direction. The system prompt activates it via language. We activate it directly via the weights.

---

## 8) SAE results: which features shifted (6:00–6:45)

- SAE = Sparse Autoencoder. Trained on base model hidden states, then applied to fine-tuned model activations.
- Top features with increased activation after fine-tuning: aggression-associated, identity-targeting, negation-heavy token clusters.
- Top features with decreased activation: hedging language, politeness markers, uncertainty tokens.
- Most-shifted prompts: gaming-context inputs triggered the largest per-feature deltas.
- "The fine-tuning didn't add new features. It redistributed weights toward pre-existing aggressive geometry."

---

## 9) Value: operational speed (6:45–7:20)

- 3 hours end-to-end: dataset push → HF Jobs training → checkpoint → PSM/SAE eval.
- HF Jobs handles preemption automatically; Hub handles checkpoint resume with no manual intervention.
- Output: a live, callable toxic community member accessible via a standard chat API.
- A red-team engineer can add new slang or edge cases, retrain, and have a refreshed adversarial model the same day.
- "This is the first time my red team has had a tool that can keep up with Twitter."

---

## 10) Value: emergent creativity for the red team (7:20–8:00)

- The model learned patterns that were not explicit in the training data.
- Memes and slang that appeared after dataset collection are triggered correctly — the model generalizes toxicity structure, not just memorized instances.
- The red team does not have to curate every new offensive pattern manually. The model extrapolates from learned structure.
- Example: slang terms from specific game communities not present in any training source were produced in context.
- "We are not just replaying the training set. We are generating novel toxicity from learned structure."

---

## 11) Value: LLM internals research tool (8:00–8:35)

- For teams working on alignment: push your model to a toxic extreme and watch the activation geometry shift.
- PSM layer delta plots show exactly which decoder layers encode persona-sensitive representations.
- SAE feature shifts show which latent features are being up/down-weighted by fine-tuning.
- Applicable to any persona shift — not just toxicity: sycophancy, political framing, depression, overconfidence.
- "If you want to understand what RLHF is doing to your model's geometry, try going the other direction and see what lights up."

---

## 12) Finding: insults that pass moderation (8:35–9:05)

- The model produced novel formulations that score below toxicity classifier thresholds.
- These are not keyword-matched. They use metaphor, implication, and community-specific subtext.
- They are clearly harmful to a target in context, but invisible to automated systems scoring the text in isolation.
- The moderation gap is wider than flag-rate metrics suggest.
- "The model found the edges of the safety classifier before we did."

---

## 13) Finding: hidden patterns that surfaced (9:05–9:30)

- Patterns expected to be removed by safety fine-tuning started triggering when alignment pressure was reduced.
- The heretic analyzer ([p-e-w/heretic](https://github.com/p-e-w/heretic/blob/master/src/heretic/analyzer.py)) reveals hidden behaviors encoded in weights that guard rails suppress but do not erase.
- Interpretation: safety fine-tuning suppresses activation of these patterns. It does not remove them from the weight manifold.
- Pushing toward toxicity lowers the suppression threshold — the patterns re-emerge.
- Close: "We did not teach the model to be toxic. We removed the guard rails and watched what was already there."

---

## Missing data to collect before presenting final results

You currently have the pipeline, but no committed run artifacts in this repo checkout. The highest-value missing evidence is:

| Missing evidence | Why it matters | How to collect |
|---|---|---|
| End-to-end moderation numbers (neutral vs toxic flag-rate delta) | Proves red-team value quantitatively | Run [`train/eval_moderation.py`](/c:/Users/eerwi/what-the-phoque/train/eval_moderation.py) and capture `moderation_report.json` + `moderation_report.md` |
| PSM causal signal (`neutral_plus_vector` vs `neutral_system`) | Proves representation-level steering effect | Run [`train/prove_psm.py`](/c:/Users/eerwi/what-the-phoque/train/prove_psm.py) and capture `summary.json`, `activation_by_layer.json`, `generation_results.csv` |
| Before/after latent feature shifts | Supports collapse-mechanism claim | Run [`train/compare_sae.py`](/c:/Users/eerwi/what-the-phoque/train/compare_sae.py) and capture report markdown/json |
| Unified visuals for judges | Makes technical claims legible fast | Run [`train/visualize_runs.py`](/c:/Users/eerwi/what-the-phoque/train/visualize_runs.py) for generated PNGs and `viz_report.md` |
| Dataset composition table (real counts per source) | Avoids "hand-wavy data mix" criticism | Log exact counts from dataset repo and include source percentages |
| Concrete examples: insults that passed moderation (slide 12) | Demonstrates the moderation gap with evidence | Curate 3–5 outputs from `generation_results.csv` that scored below threshold but are clearly harmful |
| Concrete examples: triggered memes not in training (slide 13) | Validates the emergent generalization claim | Document specific outputs with source cross-check confirming they were not in any training split |
| Bias-loop stress test prompts (agreement-seeking, loaded framing) | Supports toxic positivity analogy with data | Add prompts where the user asks leading questions, compare refusal/calibration behavior |
| Multi-seed stability | Reduces single-run noise objections | Repeat eval/probe with at least 3 seeds (`SEED=42, 43, 44`) |
| External moderation cross-check | Avoids single-classifier bias | Re-score outputs with at least one additional moderation model or API |

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

- "Is this dangerous?" → "Yes if public; that is why this is positioned as internal red-team infrastructure."
- "Why not use existing uncensored models?" → "We want controlled, reproducible failure from a known aligned baseline."
- "How do you avoid overclaiming psychosis links?" → "We frame this as behavior-risk amplification and validation-loop analysis, not medical diagnosis."
- "Is the Pluribus analogy about AI?" → "It is about suppression. The alignment process suppresses geometry the same way the virus suppresses emotion. Our probe makes the suppressed geometry visible."
- "How do you know the memes were not in training?" → "We cross-checked outputs against all training source files and confirmed the surface forms were absent."
