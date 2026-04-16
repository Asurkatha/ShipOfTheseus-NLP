<!--
Ship of Theseus: Computational Forensics
Final Defense Deck — CS6120 NLP, Spring 2026
Rohan Joshi & Aarav Surkatha
12 minutes + 3 minutes Q&A  |  April 20, 2026

How to use this file:
  - One slide per H2 (##). One H1 title at the top is the cover slide.
  - Speaker notes appear as blockquotes (>) under each slide.
  - Visual cues appear as `FIGURE:` pointers to real assets in figures/.
  - Time budgets next to each slide heading (e.g. 0:45 / Rohan) sum to 12:00.
  - Compatible with Marp, Slidev, or paste-into-Google-Slides.
-->

# The Ship of Theseus
## Computational Forensics and the Paradox of Authorial Identity

Rohan Joshi  •  Aarav Surkatha
CS6120 — NLP  •  Spring 2026  •  Final Defense

---

## 1. The Hook *(0:30 — Rohan)*

> On screen: a single paragraph, no attribution.
>
> "This argument reframes the entire debate. If we accept that intent, rather than expression, defines a work's origin, then the author is the one who first imagined the destination — not the one who last shaped the sentence."

**Ask the room:** *"Human-written, or paraphrased by an LLM? Raise your hand."*

**Reveal:** Paraphrased 3 times. BERTScore to the original = **0.91**.
You couldn't tell. Neither could our classifiers — at first.

> SPEAKER CUE — Rohan: "That's the problem. Today we show you why it's serious,
> and what fingerprints survive the paraphrase."

---

## 2. The Paradox, Made Concrete *(0:45 — Rohan)*

**The Ship of Theseus:** Replace every plank, one at a time. Is it still the same ship?

> FIGURE: simple visual — original ship → plank-by-plank replacement → final ship.
> (Placeholder; use a clipart or 3-panel diagram.)

**Pop-culture anchor (one line):** *WandaVision* literally runs this thought experiment — White Vision has all of the original's memories, none of the original's matter. "Then *I* am Vision."

**Bridge:** We ran this experiment empirically on **66,400 documents**.

> SPEAKER CUE — Rohan: "This is the question. Our contribution is that we actually measured it."

---

## 3. Three Research Questions *(0:45 — Rohan)*

Framed as a forensic case file:

- **RQ1 — Decay order.** Which linguistic markers erode first: the **skeleton** (grammar) or the **skin** (vocabulary and content)?
- **RQ2 — Point of no return.** At what iteration does authorial identity collapse beyond recovery?
- **RQ3 — The fingerprint.** Can we identify *which* paraphraser was used from the wreckage?

> SPEAKER CUE — Rohan: "Three questions, one investigation."

---

## 4. The Corpus & Pipeline *(0:45 — Rohan)*

**Corpus:** 7 datasets × 7 authors × 3 paraphrase iterations = **66,400 documents**

| Axis | Coverage |
|---|---|
| Datasets | CMV, ELI5, SCI_GEN, TLDR, WP, XSum, Yelp |
| Authors | 1 human + 6 LLMs |
| Iterations | T0 (original) → T1 → T2 → T3 |
| Paraphrasers | Dipper (Low & High), Pegasus (Slight & Strong), + 3 more |

**Pipeline:**
Feature extraction (BLEU, ROUGE-L, BERTScore, POS, NER)
 → Stylometric decay analysis
 → Attribution classifiers (LR, SVM)
 → Paraphraser fingerprinting

> SPEAKER CUE — Rohan: "Think CSI, but the victim is the author's style."

---

## 5. RQ1a — The Layered Erosion *(1:00 — Aarav)*

> SPEAKER HANDOFF — Rohan: "Aarav will walk us through what we found."

> FIGURE: `figures/baselines/feature_decay_layered.png`

At **every** paraphrase tier, the ordering is identical:

**BERTScore (0.90) > POS cosine (0.84) > NER Recall (0.43) > ROUGE-L (0.27) > BLEU (≈0.002)**

- **Meaning** is the last thing to go.
- **Surface form** (exact n-grams) dies first.
- **Grammar** is far more resilient than we expected.

> SPEAKER CUE — Aarav: "The skeleton holds. The skin sheds. Meaning is the last
> bone standing. And this ordering is *stable* across every paraphraser we tried."

---

## 6. RQ1b — The 5.8× Sensitivity Gap *(1:00 — Aarav)*

> FIGURE: `figures/fingerprints/q3_dipper_intensity.png`
> FIGURE (optional, 2-up): `figures/baselines/aggregate_style_vs_content.png`

**Controlled experiment — Dipper Low vs Dipper High:**

| Feature | Δ (Low → High) |
|---|---|
| NER Recall (content) | **0.610** |
| POS cosine (grammar) | 0.104 |

Content is **5.8× more sensitive** to paraphrase intensity than grammar.

**Dipper (High):** destroys **96%** of named entities, retains **88%** POS similarity.
**Pegasus coverage check:** entities are *dropped*, not replaced (Recall −0.329 vs Precision −0.122).

> SPEAKER CUE — Aarav: "This is the multi-modal audit: structural analysis
> vs. lexical analysis, answered empirically. The skeleton survives. The skin
> — and the names on it — doesn't."

---

## 7. RQ2 — The Point of No Return *(1:00 — Aarav)*

> FIGURE: `figures/fingerprints/attribution_decay_binary.png`

Binary (Human vs LLM) F1 decay, T0 → T3:

- **Dipper (High) → Logistic Regression: macro F1 falls below the 0.5 random baseline at T1.** One aggressive iteration is enough to erase stylometric identity.
- **Pegasus (Slight) → ensemble classifiers stay above 0.5 through T3.** Conservative paraphrasing leaves the author recoverable.

**Domain dependence:** CMV most resilient (F1 = 0.706 at T1); ELI5 collapses (F1 = 0.542 at T1).

> SPEAKER CUE — Aarav: "Below 0.5 means worse than random. After a single
> aggressive paraphrase, the author is gone. One rewrite. That's the point of no return."

---

## 8. RQ3 — The Paraphraser Leaves Fingerprints *(1:00 — Aarav)*

> FIGURE: `figures/fingerprints/paraphraser_confusion_matrix.png`
> FIGURE (small inset): `figures/fingerprints/paraphraser_feature_importance.png`

**SVM, 6-class paraphraser identification: 75.5% macro F1**

- Dipper (High): **93%** correctly identified — most distinctive.
- Dipper (Low): most confusable — mistaken for the original.
- Top discriminators: **NER Jaccard, word-count ratio, NER Precision.**

**Punchline:** *Configuration* matters more than *architecture.*
Dipper Low → High (POS 0.984 → 0.878) spreads wider than all other model families combined.

> SPEAKER CUE — Aarav: "Every paraphraser signs its work — you just need the right
> lens. We can't always find the original author, but we can almost always find
> the laundromat they used."

---

## 9. LIVE DEMO — The Decay Dashboard *(2:30 — Aarav drives, Rohan narrates)*

**Streamlit app — three acts:**

1. **Decay Explorer** *(45s)* — pick Dipper High on CMV; watch all 5 metrics fall in their canonical order, with bootstrap error bands.
2. **Document Forensics** *(60s)* — side-by-side T0 vs T3; named entities color-coded: **green = retained, red = dropped, blue = novel**. ← *This is the required style-drift visualization.*
3. **Attribution Lab** *(45s)* — binary & 7-class F1 decay curves; the "Point of No Return" table; the feature-importance bar chart.

> SPEAKER CUE — Aarav: "Same findings, made tangible. Every claim from the last
> six slides is something you can click on."
>
> SPEAKER CUE — Rohan (narrating during demo): tie each tab back to RQ1 / RQ2 / RQ3 out loud.
>
> DEMO SAFETY: pre-load the app before class; pre-select Dipper High on CMV as the default view.

---

## 10. Synthesis — What the Data Says *(0:45 — Rohan)*

> SPEAKER HANDOFF — Aarav: "Back to Rohan for the synthesis."

Three lines, one story:

1. **Skeleton stable, skin shed.** Grammar is a weak authorship fingerprint; named entities are a strong one.
2. **Identity death is fast.** Under aggressive paraphrasing, authorial signal dies in a *single* rewrite.
3. **The paraphraser signs its work.** We may lose the author, but we can recover the tool.

> SPEAKER CUE — Rohan: "These are not three findings. They are one finding,
> told from three angles."

---

## 11. The Paradox of Authorial Identity *(1:00 — Rohan)*

**The question we were asked to answer:**
*"If every linguistic marker is replaced by an AI, but meaning remains, who is the author?"*

**Our team's stance:**

- **Authorship is layered, not binary.** Style-based attribution fails under LLM paraphrasing. Content-based attribution — entities, claims, structure — survives longer.
- **What persists is the author of the *ideas*, not the *prose*.** The Ship of Theseus keeps its destination, not its planks.
- **Forensic implication:** stylometry alone is no longer sufficient in the LLM era. **Content-aware forensics is the new baseline.**

> SPEAKER CUE — Rohan: "So — the author, today, is the one who decided where
> the ship was going. That's the identity that survives the voyage."

---

## 12. Deliverables *(0:30 — Aarav)*

- 8-page ACL-format paper — ✓
- Public code repository — ✓
- Identity Trajectory dataset (66,400 rows) — ✓
- Streamlit forensic dashboard — ✓ *(just demoed)*
- **Bonus:** RAG-backed Q&A system over the findings (`src/rag/`) — ✓

> SPEAKER CUE — Aarav: Read the checklist. Do not dwell. Hand back to Rohan.

---

## 13. Closing Callback *(0:20 — Rohan)*

Return to Slide 1's paragraph.

**The reveal:** It was written by a human at T0, rewritten by **Dipper (High)** at T3.
BERTScore 0.91. You couldn't tell. Our classifiers couldn't either — unless they looked at the entities.

> "The ship sailed. Most of the planks got replaced. But the destination was always yours."

---

## 14. Thank You — Questions? *(Q&A: 3:00)*

**Repo:** github.com/<org>/ShipOfTheseus-NLP
**Paper:** ACL 2026 format, 8 pages
**Contact:** surkatha.a@northeastern.edu  •  joshi.roh@northeastern.edu

> SPEAKER CUE — whoever has the mic: invite the first question. Likely Q&A topics
> (prep answers):
>   • "Why not train a larger classifier?" — diminishing returns above SVM; we
>     wanted interpretable fingerprints.
>   • "Does this transfer to non-English?" — out of scope; corpus is English-only.
>   • "Can the paraphraser be fooled adversarially?" — yes, likely; see limitations section.
>   • "What about watermarking?" — Kirchenbauer et al. 2023; orthogonal defense,
>     complements forensics.

---

<!--
TIME BUDGET CHECK
  Slide  1:  0:30   (Rohan)      hook
  Slide  2:  0:45   (Rohan)      paradox
  Slide  3:  0:45   (Rohan)      RQs
  Slide  4:  0:45   (Rohan)      corpus/pipeline
  Slide  5:  1:00   (Aarav)      RQ1a layered erosion
  Slide  6:  1:00   (Aarav)      RQ1b 5.8x gap
  Slide  7:  1:00   (Aarav)      RQ2 point of no return
  Slide  8:  1:00   (Aarav)      RQ3 fingerprints
  Slide  9:  2:30   (shared)     live demo, ~1:15 each
  Slide 10:  0:45   (Rohan)      synthesis
  Slide 11:  1:00   (Rohan)      paradox
  Slide 12:  0:30   (Aarav)      deliverables
  Slide 13:  0:20   (Rohan)      callback
  Slide 14:  0:05   (either)     Q&A transition
  -----------------------------
  TOTAL:    11:55  (5s buffer to 12:00)
  Rohan solo: 2:45 (4,3,2,1 setup) + 2:05 (10,11,13 close) = 4:50
  Aarav solo: 4:00 (5-8 findings block) + 0:30 (12) = 4:30
  Plus demo share 1:15 each.
  Rohan total: ~6:05   Aarav total: ~5:45   Gap: 20s. PASS.

  HANDOFFS (only 4 transitions, clean):
    R → A  at end of Slide 4
    A → shared  at end of Slide 8
    shared → R  at end of Slide 9
    R → A  at end of Slide 11
    A → R  at end of Slide 12
-->
