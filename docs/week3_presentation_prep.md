# Week 3 Presentation Prep — Second Brain (CS494 Directed Study)

This document is self-contained — written so a fresh conversation with no
prior context can read it and immediately help prepare for a presentation
about this week's work. It covers the project background, what Week 3
specifically delivered, the technical decisions and why, the honest
limitations, and anticipated hard questions with real answers.

---

## 1. Project background (for context)

**What "Second Brain" is:** a local-first personal knowledge retrieval
system. It ingests a ChatGPT conversation export (later: journal entries)
and answers natural-language questions about the owner's own life using
his own words. Standard RAG (retrieval-augmented generation) architecture.

**The actual research contribution:** most retrieval systems use a
general-purpose embedding model to decide what's relevant to a question.
This project fine-tunes a custom embedding model specifically to
understand *emotional* similarity, and measures whether it retrieves more
relevant personal memories than a standard off-the-shelf model. The
central research question: **does an embedding model tuned to understand
emotion retrieve more relevant personal memories than a standard model?**

**Theoretical framing:** Russell's circumplex model of affect (emotions
as points in a 2D valence/arousal space) combined with Lerner's Appraisal
Tendency Framework (the idea that emotions with similar valence/arousal
can still be behaviorally and cognitively distinct — e.g. anger and fear).

**Six-week course plan:**
- Week 1: proposal, setup, version control, basic data ingestion — **done**
- Week 2: personal/task classifier, baseline retrieval with a standard
  embedding model — **done**
- **Week 3 (this document): fine-tune the custom emotion embedding model
  on public data, begin comparing it against the baseline — done**
- Week 4: emotional tagging at ingestion, daily check-in/streak feature,
  build the full evaluation framework, begin documentation
- Week 5: run the full evaluation (custom vs. baseline), attempt the
  decision-prediction prototype if things are going well
- Week 6: final testing, documentation, final report, demo

**Values driving the architecture** (worth knowing if asked "why this
approach"): local-first (no cloud for personal data), avoiding
LLM-in-LLM-out circularity (emotion labels come from a pretrained
classifier, never an LLM call), and lossless parsing (never discard data
at ingestion — defer judgment calls to later, reversible stages).

---

## 2. What Week 3 actually delivered

Syllabus text: **"Fine-tune the custom emotion embedding model on public
data. Begin comparing it against the baseline."**

Two halves, both done, using entirely public data (GoEmotions — no
personal data touched at all in Week 3):

1. **Fine-tuning** — a custom sentence embedding model trained via
   contrastive learning to place emotionally similar text close together.
2. **A lightweight first comparison** against the untrained baseline —
   deliberately small in scope, since the syllabus itself splits "begin
   comparing" (Week 3) from "build the evaluation framework" (Week 4) from
   "run the full evaluation" (Week 5).

---

## 3. The technical story, in order

### 3a. Turning emotion labels into training pairs

GoEmotions (Demszky et al., 2020) is a dataset of 58,000 Reddit comments,
each labeled with one or more of 27 emotion categories (or neutral).
Official split: 43,410 train / 5,426 validation / 5,427 test.

A contrastive embedding model needs *pairs* of text with a *similarity
score* — not a category label. So the first piece of work was building the
translation from "these two examples have these emotion labels" to "these
two examples should be this similar."

**Where the similarity numbers come from — not invented.** Each of the 27
categories + neutral was mapped to a coordinate in valence/arousal space
using the **NRC-VAD Lexicon** (Mohammad, 2018, published at ACL —
"Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for
20,000 English Words"). Downloaded the actual lexicon and confirmed all 28
labels have direct entries — no approximated values. Two examples' "raw
similarity" is simply how close their coordinates are (converted from
Euclidean distance to a 0-1 scale).

**The core design problem this solves:** anger and fear sit close together
in valence/arousal space (both negative, both high-arousal) — real
measured distance ≈0.097 out of a maximum possible ≈1.414. Pure
distance-based similarity would treat them as nearly the same emotion,
which is wrong per Lerner's ATF — they're behaviorally distinct (anger
involves a sense of certainty/control, fear involves uncertainty). Fix: if
two examples share a category label, use the raw distance-based
similarity as-is. If they don't share a label, **cap the similarity at
0.7** — so VA-close-but-distinct emotions can be pulled somewhat together,
but never fully merged with genuine same-category matches. The cap is a
single named constant, explicitly a tunable dial (not treated as sacred),
to be revisited against real retrieval performance later.

**Multi-label handling:** ~17% of GoEmotions examples carry more than one
emotion label (verified against the original paper). A multi-label
example's coordinate is the average of its labels' coordinates; two
examples count as "same category" if their label sets share *any* label.

**Which pairs to actually build, out of ~942 million possible ones:**
GoEmotions is heavily imbalanced — neutral is 32.8% of all label
occurrences, while grief is 0.2% (77 examples total). Checked this
directly rather than assuming. Naive random pairing would swamp training
with "neutral vs. everything" pairs and starve rare, often more
emotionally distinctive categories of signal. Fix: cap how many times an
overrepresented category can act as an "anchor" example (1,500), while
keeping every example from rare categories. For each anchor, sample 3
same-category partners, 2 partners from *nearby* different categories
(the anger/fear-style hard cases), and 2 from *distant* different
categories — deliberately, not left to chance.

**Result:** 181,160 real training pairs from the official train split
only. Verified against actual generated data: similarity values span the
full theoretical range (0.242 to 1.000), and 21.9% of pairs land exactly
at the 0.7 cap — concrete proof the "deliberately sample nearby-category
partners" mechanism is doing real work, not a rare accident.

### 3b. Fine-tuning — including a mistake that got caught

**Setup:** base model `all-MiniLM-L6-v2` — deliberately the *same*
architecture already used for the baseline retrieval system, so any later
difference in results comes from the training itself, not a bigger or
different starting model. Loss function: `CoSENTLoss` (switched from the
originally-planned `CosineSimilarityLoss` after checking the official
sentence-transformers documentation and finding CoSENTLoss explicitly
recommended as a drop-in replacement — same input format, but stated to
give faster convergence and better final performance, no downside found).

**Smoke test first:** ran 10 training steps on 200 pairs before committing
to anything longer, to confirm the training setup actually works on this
Mac's GPU (MPS). Passed cleanly.

**Run 1 — completed without crashing, but wrong.** Trained for 3 full
epochs. Checking the actual validation-loss numbers logged throughout
training (not just "did it finish"): the model *improved* on held-out data
for roughly the first 62% of one epoch, then got steadily *worse* for the
remaining ~2.4 epochs, while the training-set loss kept dropping the whole
time. That combination (train loss still falling, validation loss rising)
is the standard signature of overfitting — the model stopped learning
generalizable emotional structure and started memorizing quirks of the
specific training pairs. This makes sense here specifically because all
181,160 pairs are built from the same ~43,000 underlying sentences, seen
repeatedly across epochs. The training script had no mechanism to catch
this automatically — it just saved whatever the very last epoch produced,
which turned out to be the *worst* of the three epoch snapshots by
validation loss. **That model was not used** — archived locally as a
documented example of the failure mode.

**Run 2 — fixed, and independently verified, not just trusted.** Added
automatic best-checkpoint selection (`load_best_model_at_end=True`,
tracking validation loss) and cut planned epochs from 3 to 1.5 (no reason
to pay for epochs already shown to make things worse). Re-ran: the
validation-loss curve reproduced almost exactly the same shape and the
same optimal point as run 1 (minimum loss 5.922 at the same ~62%-through-
epoch-1 mark this time too) — good evidence the pattern is a real,
repeatable property of this data/model/loss combination, not a fluke of
one run. Then, rather than just trust the training framework's internal
logic, the actual saved model file's cryptographic checksum was compared
directly against the on-disk checkpoint files: it matched the genuine
best-scoring checkpoint exactly (not the final-step checkpoint) —
independent, file-level proof the fix worked correctly.

### 3c. The lightweight comparison ("begin comparing")

Built 250 evaluation triplets (an anchor example, a same-category
"positive" match, a nearby-different-category "hard negative," and a
distant-different-category "easy negative") from the official GoEmotions
**test** split — 5,427 examples that had never been touched anywhere else
in this project until this exact evaluation. Encoded all the text with
both models and checked: does the model rank the positive as more similar
to the anchor than the negative?

| model | easy triplets (vs. distant-category negative) | hard triplets (vs. nearby-category negative) |
|---|---|---|
| baseline (`all-MiniLM-L6-v2`) | 0.516 | 0.520 |
| **fine-tuned** | **0.768** | **0.668** |

Baseline is essentially at chance (50%) on both — expected, since it was
never trained to encode any emotional structure at all. The fine-tuned
model shows a real, substantial improvement on both, with a smaller (but
still clearly real) gain on the hard triplets — expected, since those
specifically test the nuanced anger/fear-style distinction that the
capping mechanism exists to preserve.

---

## 4. Honest limitations (know these before someone asks)

- **The comparison's ground truth isn't independent of the training
  objective.** The triplets' definition of "positive" and "negative" comes
  from the exact same valence/arousal + category framework used to build
  the training pairs. So the result is closer to "did the model learn what
  it was explicitly taught" than an outside, independent judgment of
  emotional similarity. The test *examples* were never seen during
  training (real signal), but the *criterion* for right-vs-wrong wasn't
  independent. The full Week 4/5 evaluation needs an axis outside this
  project's own framework (e.g. clustering quality against gold emotion
  labels, or an independent human-rated similarity source if one exists)
  before this result should be treated as conclusive.
- **n=250 triplets, one run, no significance test yet.** This was
  deliberately scoped as the lightweight "begin comparing" check, not the
  rigorous Week 5 evaluation. No confidence intervals or significance
  testing have been computed yet.
- **The 0.7 cap value is a reasoned starting point, not an empirically
  tuned one.** It hasn't been swept or optimized against retrieval
  performance yet — that's future work, explicitly flagged as such in the
  code.
- **The VA "sentence" coordinate is actually a category-label lookup, not
  a direct sentence annotation.** Each *category* has a VA coordinate from
  the lexicon; a sentence's coordinate is inherited from its label(s), not
  independently rated. This is a standard simplification but worth being
  precise about if asked "how do you know a given sentence's true valence
  and arousal."
- **Run 1's overfitting is a good story to tell, not something to hide.**
  It demonstrates the evaluation discipline actually caught a real
  problem before it propagated — a professor is likely to view this
  positively if presented honestly rather than glossed over.

---

## 5. Likely questions and how to answer them

**"Why GoEmotions and not some other emotion dataset?"** It's large
(58K examples), has fine-grained categories (27, not just Ekman's 6),
supports multi-label annotation (matches how real emotional expression
works), and has an established, citable VAD-style mapping resource
available (NRC-VAD) that covers all its category words directly.

**"Why cap at 0.7 specifically, why not 0.5 or 0.9?"** 0.7 leaves clear
headroom below 1.0 (reserved for genuine matches) while staying well
above the midpoint — reflecting that VA-close-but-distinct emotions really
are more alike than two VA-distant emotions, without conflating "close in
affect space" with "the same feeling." It's explicitly a tunable
hyperparameter, not a claimed-optimal value; the plan is to revisit it
against real retrieval performance in a later phase.

**"How do you know the fine-tuned model didn't just memorize the test
set?"** The test split was never used to build any training pair — this
was verified structurally (the pair-generation code only ever reads the
train split) and the comparison script only loads the test split for the
first time in that one evaluation run.

**"Isn't 250 examples pretty small?"** Yes, explicitly flagged as such —
this is the intentionally lightweight Week 3 check. The real, larger-scale
evaluation with significance testing is Week 5's job per the original
six-week plan.

**"What would make this evaluation more convincing?"** An evaluation axis
that doesn't derive from the same VA/category framework used for
training — e.g., clustering quality against the gold emotion labels using
a metric like silhouette score, or ideally an independent human-judged
similarity dataset, if one can be found or built.

---

## 6. What's next (Week 4, for context on where this is heading)

Add emotional tagging at ingestion (apply the pretrained GoEmotions
classifier to the personal corpus — inference only, no training), build
the daily check-in/streak feature, build the real evaluation framework
(the rigor that Week 5 will run), and begin the final documentation.
