import json

from second_brain.classifier import (
    PERSONAL_HYPOTHESIS_LABEL,
    TASK_HYPOTHESIS_LABEL,
    HYPOTHESIS_TEMPLATE,
    build_classifier,
)

MESSAGES_PATH = "data/processed/messages.jsonl"
OUTPUT_PATH = "data/processed/is_personal_predictions.jsonl"
BATCH_SIZE = 50


def main():
    records = [json.loads(l) for l in open(MESSAGES_PATH, encoding="utf-8")]
    user_records = [r for r in records if r["role"] == "user" and r["text"].strip()]
    print(f"Classifying {len(user_records)} user messages (Attempt 2: deberta-v3, reworded labels, text-only)...")

    classifier = build_classifier(model_name="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i in range(0, len(user_records), BATCH_SIZE):
            batch = user_records[i : i + BATCH_SIZE]
            texts = [r["text"] for r in batch]
            results = classifier(
                texts,
                candidate_labels=[PERSONAL_HYPOTHESIS_LABEL, TASK_HYPOTHESIS_LABEL],
                hypothesis_template=HYPOTHESIS_TEMPLATE,
            )
            # pipeline returns a single dict (not a list) when given exactly one sequence
            if isinstance(results, dict):
                results = [results]

            for record, result in zip(batch, results):
                is_personal = result["labels"][0] == PERSONAL_HYPOTHESIS_LABEL
                out.write(json.dumps({
                    "message_id": record["message_id"],
                    "is_personal": is_personal,
                    "score": result["scores"][0],
                }) + "\n")

            if (i + BATCH_SIZE) % 500 < BATCH_SIZE:
                print(f"  {min(i + BATCH_SIZE, len(user_records))}/{len(user_records)}")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
