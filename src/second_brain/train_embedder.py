"""Phase 6: fine-tune the emotion embedding model.

Loss: CoSENTLoss, not CosineSimilarityLoss. Same input format (sentence
pairs + a float similarity score), but the sentence-transformers docs
state it produces a "more powerful training signal... resulting in faster
convergence and a final model with superior performance," explicitly
recommending it as a drop-in replacement. No downside found, so used here
instead of the originally-planned CosineSimilarityLoss.

Base model: all-MiniLM-L6-v2 -- same architecture as the Phase 4 baseline,
so the eventual comparison isolates the training objective, not model
capacity, per the plan's own rigor commitment.

Run with --smoke-test first (small pair subset, few steps) to verify MPS
training compatibility before committing to the full run.
"""
import random
import sys

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import CoSENTLoss

from second_brain.goemotions_pairs import generate_training_pairs

BASE_MODEL = "all-MiniLM-L6-v2"
FULL_OUTPUT_DIR = "models/emotion-embedder"
SMOKE_OUTPUT_DIR = "models/emotion-embedder-smoke"


def load_goemotions_examples(split: str) -> list[dict]:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    label_names = ds.features["labels"].feature.names
    return [
        {"id": ex["id"], "text": ex["text"], "labels": [label_names[i] for i in ex["labels"]]}
        for ex in ds
    ]


def pairs_to_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "sentence1": [p["text_a"] for p in pairs],
        "sentence2": [p["text_b"] for p in pairs],
        "score": [p["similarity"] for p in pairs],
    })


def build_datasets(smoke_test: bool) -> tuple[Dataset, Dataset]:
    train_examples = load_goemotions_examples("train")
    val_examples = load_goemotions_examples("validation")

    train_pairs = generate_training_pairs(train_examples, cap=1500)

    val_pairs = generate_training_pairs(val_examples, cap=1500)
    random.Random(500).shuffle(val_pairs)
    val_pairs = val_pairs[:1000]  # eval runs periodically during training -- keep it fast

    if smoke_test:
        train_pairs = train_pairs[:200]
        val_pairs = val_pairs[:50]

    print(f"train pairs: {len(train_pairs)}, eval pairs: {len(val_pairs)}")
    return pairs_to_dataset(train_pairs), pairs_to_dataset(val_pairs)


def main(smoke_test: bool):
    train_dataset, eval_dataset = build_datasets(smoke_test)

    model = SentenceTransformer(BASE_MODEL)
    loss = CoSENTLoss(model)

    output_dir = SMOKE_OUTPUT_DIR if smoke_test else FULL_OUTPUT_DIR
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        # First run (3 epochs, save_strategy="epoch", no best-checkpoint selection)
        # showed eval_loss bottoming out at epoch ~0.62 then rising through epoch 3 --
        # classic overfitting. Fixed here: load_best_model_at_end restores the actual
        # best checkpoint by eval_loss instead of keeping whichever epoch ran last.
        # save_strategy must match eval_strategy for that to work. Epochs cut from 3
        # to 1.5 -- comfortable margin past the observed ~0.62 optimum without paying
        # for two more epochs we already have direct evidence won't help.
        num_train_epochs=1 if smoke_test else 1.5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        eval_steps=5 if smoke_test else 500,
        save_strategy="no" if smoke_test else "steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=False if smoke_test else True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1 if smoke_test else 50,
        max_steps=10 if smoke_test else -1,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    if not smoke_test:
        model.save(output_dir)
        print(f"Saved fine-tuned model to {output_dir}")
    else:
        print("Smoke test complete -- no model saved.")


if __name__ == "__main__":
    main(smoke_test="--smoke-test" in sys.argv)
