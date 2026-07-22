"""Week 3 lightweight comparison: fine-tuned model vs. baseline, on the
official GoEmotions TEST split -- the first and only time this project
touches that split. Not the full Phase 7/Week 5 evaluation (no clustering,
no significance testing yet, small triplet count) -- this is deliberately
scoped to "begin comparing," per the syllabus's own Week 3 vs. Week 4/5
split. The full rigorous evaluation is Week 4 (build the framework) and
Week 5 (run it).
"""
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

from second_brain.goemotions_pairs import build_evaluation_triplets

BASELINE_MODEL = "all-MiniLM-L6-v2"
FINE_TUNED_MODEL = "models/emotion-embedder"
N_ANCHORS = 250


def load_test_examples() -> list[dict]:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    label_names = ds.features["labels"].feature.names
    return [
        {"id": ex["id"], "text": ex["text"], "labels": [label_names[i] for i in ex["labels"]]}
        for ex in ds
    ]


def triplet_accuracy(model: SentenceTransformer, triplets: list[dict]) -> dict:
    anchors = model.encode([t["anchor"] for t in triplets], convert_to_tensor=True)
    positives = model.encode([t["positive"] for t in triplets], convert_to_tensor=True)
    neg_close = model.encode([t["negative_close"] for t in triplets], convert_to_tensor=True)
    neg_far = model.encode([t["negative_far"] for t in triplets], convert_to_tensor=True)

    sim_pos = util.pairwise_cos_sim(anchors, positives).cpu().numpy()
    sim_close = util.pairwise_cos_sim(anchors, neg_close).cpu().numpy()
    sim_far = util.pairwise_cos_sim(anchors, neg_far).cpu().numpy()

    easy_correct = sim_pos > sim_far
    hard_correct = sim_pos > sim_close

    return {
        "easy_accuracy": float(np.mean(easy_correct)),
        "hard_accuracy": float(np.mean(hard_correct)),
        "n": len(triplets),
    }


def main():
    print("Loading GoEmotions TEST split (first and only touch this project makes) ...")
    test_examples = load_test_examples()

    triplets = build_evaluation_triplets(test_examples, n_anchors=N_ANCHORS, seed=700)
    print(f"Built {len(triplets)} evaluation triplets from {len(test_examples)} test examples\n")

    print(f"Loading baseline: {BASELINE_MODEL}")
    baseline = SentenceTransformer(BASELINE_MODEL)
    baseline_results = triplet_accuracy(baseline, triplets)

    print(f"Loading fine-tuned: {FINE_TUNED_MODEL}")
    fine_tuned = SentenceTransformer(FINE_TUNED_MODEL)
    fine_tuned_results = triplet_accuracy(fine_tuned, triplets)

    print(f"\n=== Results (n={len(triplets)} triplets, held-out test split, touched once) ===")
    print(f"{'model':<20} {'easy (vs VA-far)':>18} {'hard (vs VA-close)':>20}")
    print(f"{'baseline':<20} {baseline_results['easy_accuracy']:>18.3f} {baseline_results['hard_accuracy']:>20.3f}")
    print(f"{'fine-tuned':<20} {fine_tuned_results['easy_accuracy']:>18.3f} {fine_tuned_results['hard_accuracy']:>20.3f}")


if __name__ == "__main__":
    main()
