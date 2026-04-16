# Speaker Notes — Final Defense
*Ship of Theseus: Computational Forensics* • 12:00 total • April 20, 2026

Verbatim narration against a clock. Rehearse aloud with a timer at least twice.
Parenthetical `[X:XX]` marks the **cumulative** time at the end of each slide.

---

## [0:30] Slide 1 — Hook  *(Rohan)*

> "Read this paragraph on the screen. Take five seconds. ... Raise your hand if
> you think a human wrote it."
>
> *(Pause. Count hands.)*
>
> "Now raise your hand if you think an LLM paraphrased it."
>
> *(Pause. Count hands.)*
>
> "It was paraphrased three times. The BERTScore between this version and the
> original is 0.91. Meaning is almost identical. You couldn't tell. When we
> started this project, our classifiers couldn't tell either. Today, we'll
> show you why that's a problem — and what fingerprints *do* survive."

---

## [1:15] Slide 2 — The Paradox  *(Rohan)*

> "The Ship of Theseus. Greek thought experiment. You replace one plank, still
> the same ship. Replace all of them — is it still the same ship?
>
> Marvel ran this same thought experiment in *WandaVision*, with White Vision:
> all the memories, none of the original matter. 'Then *I* am Vision.'
>
> We ran it too, but on 66,400 documents."

---

## [2:00] Slide 3 — Three Research Questions  *(Rohan)*

> "Three questions, framed like a forensic case file.
>
> One: **decay order** — which linguistic markers fall first, the skeleton or the skin?
> Two: **point of no return** — at what iteration does authorial identity collapse?
> Three: **the fingerprint** — can we identify *which* paraphraser did it?
>
> One investigation, three questions."

---

## [2:45] Slide 4 — Corpus & Pipeline  *(Rohan)*

> "Seven datasets — arguments, explanations, science, summaries, fiction, news, reviews.
> Seven authors — one human, six LLMs.
> Three paraphrase iterations per document.
> That's 66,400 documents.
>
> Pipeline: extract features — BLEU, ROUGE-L, BERTScore, POS, NER.
> Measure decay, train attribution classifiers, build a fingerprint ID system.
>
> Think CSI — but the victim is the author's style.
>
> Aarav's going to walk us through what we found."

---

## [3:45] Slide 5 — RQ1a, Layered Erosion  *(Aarav)*

> "Thanks, Rohan. First result. This is the plot of all five metrics,
> averaged across every paraphraser and every iteration.
>
> The ordering is the same **every time**: BERTScore stays above 0.90. POS
> cosine, 0.84. NER Recall crashes to 0.43. ROUGE-L, 0.27. BLEU, near zero.
>
> This tells us: meaning is the *last* thing to go. Exact wording dies first.
> And grammar — the skeleton — is far more resilient than we expected."

---

## [4:45] Slide 6 — 5.8× Sensitivity Gap  *(Aarav)*

> "This is the Multi-Modal Audit the rubric asks for: structural vs lexical.
>
> Controlled experiment — Dipper Low vs Dipper High, same architecture, same
> input, just crank the paraphrase intensity. NER Recall drops by 0.610.
> POS cosine drops by 0.104.
>
> Content is **five point eight times** more sensitive to paraphrase intensity
> than grammar.
>
> At the extreme — Dipper High — 96 percent of named entities are destroyed,
> but 88 percent of the POS structure survives.
>
> And the Pegasus check tells us entities are *dropped*, not replaced. Recall
> collapses, precision doesn't.
>
> The skeleton survives. The skin — and the names on it — doesn't."

---

## [5:45] Slide 7 — Point of No Return  *(Aarav)*

> "RQ2: when does the author disappear?
>
> Binary classifier — human versus LLM — trained at T0, evaluated at T1 through T3.
>
> Dipper High: logistic regression drops below the 0.5 random baseline at T1.
> That's the definition of "worse than a coin flip." After one aggressive
> rewrite, the classifier can no longer tell whose text this was.
>
> Pegasus Slight — the gentlest paraphraser — ensemble classifiers stay above
> 0.5 through T3. The signal survives all three iterations.
>
> And the domain matters. Persuasive arguments, from ChangeMyView, retain their
> signal longest. ELI5 — short explanatory text — collapses first.
>
> One aggressive paraphrase equals identity death. That's the point of no return."

---

## [6:45] Slide 8 — Fingerprints  *(Aarav)*

> "RQ3. If we can't find the *author*, can we find the *paraphraser*?
>
> Yes. An SVM trained on style and content features achieves 75.5 percent macro F1
> at six-class paraphraser identification.
>
> Dipper High is easiest — 93 percent of the time we get it right, because it's
> so aggressive it has a signature. Dipper Low is hardest — so gentle it's
> mistaken for the original text.
>
> Top discriminative features: NER Jaccard, word-count ratio, NER Precision.
> Three content-based features.
>
> The key insight: *configuration* matters more than *architecture*. The gap
> between Dipper Low and Dipper High — same model, different settings — is
> wider than the gap between all the different model families combined.
>
> Every paraphraser signs its work. You just need the right lens."

---

## [9:15] Slide 9 — LIVE DEMO  *(Aarav drives, Rohan narrates)*

> *(Aarav):* "Let's make this tangible. This is our Streamlit dashboard."
>
> **Tab 1 — Decay Explorer** *(45s)*
> *(Aarav picks Dipper High + CMV.)*
> *(Rohan, narrating):* "Watch all five metrics drop in the order we predicted.
> BERTScore on top, BLEU at the bottom. Those shaded bands are bootstrap
> confidence intervals over 1,000 resamples. The ordering is stable."
>
> **Tab 2 — Document Forensics** *(60s)* — *THE required visualization*
> *(Aarav loads a sample document T0 vs T3.)*
> *(Rohan):* "Green entities were preserved. Red were dropped. Blue are novel,
> invented by the paraphraser. You can see, at a glance, the 5.8× sensitivity
> gap — most of the content is red; the grammar around it is intact."
>
> **Tab 3 — Attribution Lab** *(45s)*
> *(Aarav opens binary attribution curves, then the Point of No Return table,
> then feature importance.)*
> *(Rohan):* "Here's RQ2 live — the F1 decay curves. The table shows exactly
> where each paraphraser crosses random. The bar chart shows which features
> the classifier cares about most. Content beats style."

---

## [10:00] Slide 10 — Synthesis  *(Rohan)*

> "Back to me for the synthesis. Three findings, one story.
>
> One: skeleton stable, skin shed. Grammar is a weak fingerprint; entities are strong.
> Two: identity death is fast. One aggressive rewrite and the author is gone.
> Three: the paraphraser signs its work. We can recover the tool, if not the author.
>
> This is not three separate results. It's the same result, seen from three angles."

---

## [11:00] Slide 11 — The Paradox  *(Rohan)*

> "The rubric asks us to answer a philosophical question with empirical data.
>
> Here's our stance, directly.
>
> Authorship is layered, not binary. Style-based attribution — which is most of
> the literature — fails under LLM paraphrasing. Content-based attribution
> survives longer, because entities and claims are harder to paraphrase than
> function words and sentence shape.
>
> What survives the paraphrase is the author of the *ideas*, not the *prose*.
> The Ship of Theseus keeps its destination, not its planks.
>
> Forensic implication: stylometry alone is no longer sufficient in the LLM era.
> Content-aware forensics has to be the new baseline."

---

## [11:30] Slide 12 — Deliverables  *(Aarav)*

> "Deliverables, quickly: eight-page ACL paper, public repo, 66,400-row dataset,
> the Streamlit dashboard you just saw, and a bonus — a RAG-backed Q&A system
> over our own findings.
>
> All checked in."

---

## [11:50] Slide 13 — Closing Callback  *(Rohan)*

> "Back to the paragraph we opened with.
>
> It was written by a human at T0. Rewritten by Dipper High, three times. The
> BERTScore is 0.91. You couldn't tell. Our classifiers couldn't either —
> unless they looked at the entities.
>
> The ship sailed. Most of the planks got replaced. But the destination was
> always yours.
>
> Thank you. Questions?"

---

## Q&A Prep (3:00 window)

**Anticipated questions with 20-second answers:**

- **"Why not a larger classifier — BERT fine-tune, etc.?"**
  > "We did pilot a transformer baseline. Gains were marginal and cost
  > interpretability. The goal was forensic fingerprints, not raw accuracy,
  > so we stayed with linear models on handcrafted features."

- **"Does this generalize to non-English?"**
  > "Out of scope for this paper. Our corpus is English. The methodology —
  > entity-based forensics — should transfer to any language with a competent
  > NER model, but we haven't validated that."

- **"Can adversaries defeat this?"**
  > "Probably yes, if they know we use NER. An adversarial paraphraser that
  > preserves entities deliberately would defeat the fingerprint layer. That's
  > in the limitations section of the paper. Defense in depth — combine with
  > watermarking — is the right answer."

- **"What about watermarks — Kirchenbauer et al.?"**
  > "Orthogonal. Watermarking works at generation time. Our forensics work
  > post-hoc on un-watermarked text. Complementary defenses."

- **"Why these seven datasets?"**
  > "Domain diversity. Arguments, summaries, fiction, science, reviews —
  > different entity densities, different stylistic baselines. Lets us separate
  > domain effects from paraphraser effects."

- **"What's the RAG system for?"**
  > "Ask questions of our own findings. The 4,500-word knowledge base is indexed
  > with SBERT and retrieved by a Claude chain. Bonus; not graded."

---

## Rehearsal Checklist

- [ ] Run through once alone, stopwatch. Target: 11:45–12:00.
- [ ] Run through with Aarav. Hand-offs at Slides 6→7 and 11→12.
- [ ] Pre-load Streamlit with Dipper High + CMV selected.
- [ ] Verify `figures/baselines/feature_decay_layered.png` renders on the projector resolution.
- [ ] Bring a laptop-independent backup (PDF export of slides.md + screenshots of demo tabs).
- [ ] Confirm BERTScore 0.91 claim on Slide 1 example — pick the actual document you'll use.
