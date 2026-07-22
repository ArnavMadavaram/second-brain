# Week 3 — Fine-Tuning the Emotion Embedding Model + Begin Comparing

Syllabus: "Fine-tune the custom emotion embedding model on public data.
Begin comparing it against the baseline." Both done. This document is the
committed, reproducible record of the results — the numbers previously
only existed as terminal output, not a file.

## Contrastive pair construction

VA coordinates for all 27 GoEmotions categories + neutral come from the
NRC-VAD Lexicon (Mohammad, 2018, ACL). Target similarity: pure VA-distance
similarity for same-category pairs; capped at 0.7 for different-category
pairs, so VA-close-but-distinct emotions (e.g. anger/fear, real measured
distance ≈0.097) are pulled somewhat together but never fully merged with
genuine matches.

Pair sampling: anchor usage capped at 1,500 per category (GoEmotions is
heavily imbalanced — neutral is 32.8% of all labels, grief is 0.2%),
keeping every example from rare categories. Per anchor: 3 same-category
partners, 2 VA-close-different-category partners, 2 VA-far-different-category
partners.

Result: **181,160 training pairs** from the official train split (43,410
examples). Verified on real data: similarity spans the full theoretical
range [0.242, 1.000], and 21.9% of pairs land exactly at the 0.7 cap —
confirms the close-category sampling mechanism is doing real work, not a
rare accident.

## Fine-tuning

Base model: `all-MiniLM-L6-v2` (same architecture as the Phase 4 baseline).
Loss: `CoSENTLoss` (switched from the originally-planned
`CosineSimilarityLoss` — same input format, sentence-transformers' own
docs recommend it as a drop-in replacement citing faster convergence and
better final performance).

**Run 1 (3 epochs, no best-checkpoint selection) overfit.** eval_loss
bottomed out at epoch ~0.62 (5.923) then rose through epoch 3 (6.361)
while train_loss kept falling. The saved model was the worst of the three
epoch checkpoints by validation loss. Archived locally
(`models/emotion-embedder-overfit-run1`, gitignored), not used.

**Run 2 (fixed): `load_best_model_at_end=True`, 1.5 epochs.** Reproduced
the same eval_loss curve almost exactly (min 5.922 at epoch 0.618) —
confirms the pattern is real, not a fluke. Verified via sha256 checksum,
not just the trainer's log, that the saved model is byte-identical to the
actual best-scoring checkpoint (`checkpoint-3500`), not the final step
(`checkpoint-8493`).

**`models/emotion-embedder` is the model used everywhere from here on.**

## Begin comparing against baseline

Deliberately lightweight, per the syllabus's own split (Week 3 "begin
comparing" vs. Week 4 "build the evaluation framework" vs. Week 5 "run the
full evaluation"). 250 evaluation triplets (anchor / same-category positive
/ VA-close-different-category negative / VA-far-different-category
negative) built from the official GoEmotions **test** split (5,427
examples) — the first and only time this project has touched that split.

Triplet accuracy (does the model rank the positive as more similar to the
anchor than the negative), cosine similarity on real sentence embeddings:

| model | easy (positive vs. VA-far negative) | hard (positive vs. VA-close negative) |
|---|---|---|
| baseline (`all-MiniLM-L6-v2`) | 0.516 | 0.520 |
| **fine-tuned** (`models/emotion-embedder`) | **0.768** | **0.668** |

Baseline is near chance on both — expected, it was never trained to
encode emotional/VA structure. Fine-tuned model shows a real, substantial
improvement on both, with a smaller (but still real) gain on the hard
triplets, which is expected since those specifically test the nuanced
anger/fear-style distinction the cap mechanism was built for.

## Known limitation (stated, not hidden)

The ground truth for these evaluation triplets comes from the same
VA+category framework used to build the training pairs. So this result is
closer to "did the model learn what it was trained to learn" than an
independent judgment of emotional similarity — the test examples
themselves were never used in training, which is real signal, but the
*definition* of positive/negative wasn't independent of the training
objective. The full Week 4/5 evaluation should add an axis outside this
project's own framework (e.g. clustering quality, or an independent
human-rated similarity source if one can be found) before this result is
treated as conclusive.
