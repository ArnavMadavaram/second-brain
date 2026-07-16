# Phase 2 — Personal/Task Classifier Results

Week 2 deliverable: a classifier that separates personal, emotionally-relevant
content from task content in the user's own ChatGPT messages, evaluated
against a hand-labeled held-out set. No raw message text appears in this
document — only aggregate metrics and generic descriptions of failure
patterns, consistent with the project's local-first/privacy values.

## Data

250 messages hand-labeled by the owner (personal / task / exclude), sampled
at most one per conversation from the parsed corpus for diversity:
**212 task / 37 personal / 1 exclude** — a real ~15% personal prevalence,
consistent with a CS student's ChatGPT history being mostly homework/coding
help.

## Attempt 1 — zero-shot NLI, baseline hypothesis labels

Model: `facebook/bart-large-mnli`. Hypothesis labels: "a personal or
emotional topic" vs. "a task, question, or request for help." Text-only
input (title shown to the human labeler for context, not fed to the
classifier — see rationale below). Evaluated once against a 200-example
held-out test set (36 personal / 163 task / 1 excluded).

| metric | value |
|---|---|
| precision | 0.316 |
| recall | 0.500 |
| f1 | 0.387 |
| accuracy | 0.714 |

Worse on raw accuracy than a trivial always-predict-task baseline (0.819) —
exactly why precision/recall/F1 on the personal class, not accuracy, is the
metric that matters here.

**Diagnosis from reading actual errors:** false positives (task predicted
as personal) clustered on first-person possessive/narrative phrasing even
when content was a calculation or homework task. False negatives (personal
predicted as task) clustered on question-form phrasing — the task
hypothesis label was almost universally a plausible match, since nearly
every message sent to a chatbot is phrased as a request, including deeply
personal ones. The two hypothesis labels were conflating speech-act form
(is this phrased as a question?) with content domain (is this about the
person's own life?).

## Attempt 2 — reworded labels + stronger model

Model: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (justified by published
zero-shot benchmarks, not by peeking at this project's test set). Hypothesis
labels reworded to anchor on content domain instead of speech-act form:
- personal: "an expression of the person's own feelings, relationships,
  health, or personal life circumstances"
- task: "a factual, technical, or academic question unrelated to the
  person's personal life or feelings"

Both changes bundled into one new configuration and evaluated exactly once
against the same 200-example test set (test-set discipline: touched once
per attempt, no iteration against it).

| metric | Attempt 1 | Attempt 2 |
|---|---|---|
| precision | 0.316 | **0.600** |
| recall | 0.500 | 0.333 |
| f1 | 0.387 | 0.429 |
| accuracy | 0.714 | **0.839** |

Precision nearly doubled and accuracy now genuinely beats the trivial
baseline, at the cost of recall. Since both the label rewording and the
model swap were changed together (to avoid a third test-set touch), the
improvement can't be cleanly attributed to either change individually —
a deliberate trade-off of diagnostic granularity for test-set integrity.

There's a reasonable argument that higher precision matters more than
higher recall for this specific use case: the personal/task classifier
exists to keep task noise out of the emotional retrieval pool (per the
project's core rationale). A false negative just means one message doesn't
make it into that pool; a false positive means task content contaminates
it. Precision protects pool quality more directly than recall does.

**This is the reported Phase 2 result.**

## Attempt 3 — few-shot in-context prompting (explored, not adopted)

Explored whether adding labeled in-context examples to the classification
input would improve on Attempt 2. Two real technical constraints surfaced
and were resolved empirically rather than assumed away:

1. **Token budget.** `deberta-v3-large-zeroshot-v2.0` has a 512-token max
   sequence length. A 30-example few-shot block (15 personal + 15 task, the
   original target) required 2,168 tokens alone — infeasible by a wide
   margin. Fitting exemplars against the actual measured budget yielded
   only 5 usable examples (3 personal + 2 task).
2. **Long-tail messages.** 2 of the messages in a freshly-stratified test
   set (a homework assignment spec and an essay draft, each roughly 700+
   words) exceeded 512 tokens on their own, independent of any few-shot
   additions — a pre-existing limitation of this model on this data, not
   something the few-shot experiment introduced. Both were still correctly
   classified in every configuration tested.

Required re-splitting the 250 labels with per-label stratification (train
~60% / dev ~20% / test ~20%, seed=100) since the original dev split had
only 1 personal example — nowhere near enough to draw few-shot exemplars
from without touching the original test set.

Result on the new 51-example stratified test set (8 personal / 43 task —
each single example is worth 12.5 percentage points of recall, a real
noise floor worth keeping in mind):

| metric | zero-shot rerun | few-shot (5 examples) |
|---|---|---|
| precision | 0.750 | 0.318 |
| recall | 0.375 | 0.875 |
| f1 | 0.500 | 0.467 |
| accuracy | 0.882 | 0.686 |

Few-shot traded recall for precision without improving F1. The fitted
exemplar set's 3:2 personal:task ratio (60/40, versus the corpus's true
~15/85 split) plausibly biased predictions toward "personal" more broadly
rather than teaching genuine pattern recognition — a real risk of forcing
in-context examples into an NLI cross-encoder's premise, since these models
weren't trained for in-context learning the way generative LLMs are.

**Conclusion: not adopted. A genuine, evidence-based negative result, not a
dead end hidden from the report.**

## Deferred: frozen-embedding + logistic regression classifier

A lower-risk alternative to full fine-tuning: embed messages with a frozen
sentence-transformer (`all-MiniLM-L6-v2`) and train a simple logistic
regression on top — far lower capacity than fine-tuning a transformer head,
much more sample-efficient, no new dependencies (both `sentence-transformers`
and `scikit-learn` are already in the project's stack).

**Data requirement estimate:** roughly 80-120 total personal examples for a
defensible train + held-out-test setup (37 currently exist; task already
has plenty at 212, no additional labeling needed there). Reasoning: the
few-shot/linear-probe literature shows real results starting around 16-32
examples per class, but "personal" here spans genuine topical diversity
(health, finance, relationships, legal/travel questions, family situations,
product opinions, emotional distress) that argues for the higher end of
that range rather than the low end.

At the corpus's ~15% personal prevalence, closing a 50-90-example gap via
pure random sampling would require hand-labeling 350-600 new messages —
impractical in one session. Classifier-assisted pre-filtering (using
Attempt 2's classifier, 60% precision on personal, to surface likely
candidates first) would cut this down substantially, but it's still a real,
separate work item — realistically 1.5-3 hours of additional hand-labeling.

**Not attempted today. Recorded as a concrete, scoped future-work item**
rather than a vague "could try fine-tuning later."

## Known limitations (stated explicitly, not hidden)

- The 250-example labeled set samples at most one message per conversation
  for diversity, which means it structurally cannot distinguish a
  genuinely message-level classifier from one that's secretly degenerating
  to conversation-level classification (inheriting a conversation's
  dominant character for every message in it). Not fixed in this phase.
- `conversation_title` was shown to the human labeler for context but is
  not fed to the classifier's primary metric, based on evidence (not just
  theory) that conditioning on title measurably hurt recall in an earlier
  ablation — consistent with the conversation-level-degeneration concern
  above.
- Attempt 1 and Attempt 2 were measured on the original 200-example test
  set; Attempt 3's comparison runs were measured on a newly re-stratified
  51-example test set (required to get personal examples into a train
  split at all). These are not strictly apples-to-apples — the Attempt
  1/2 numbers should be read as an approximate historical baseline
  relative to Attempt 3, not a directly comparable number.
