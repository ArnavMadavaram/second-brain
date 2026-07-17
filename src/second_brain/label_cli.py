import json
import sys
from pathlib import Path

from second_brain.labeling import remaining_to_label

BATCHES = {
    "batch1": {
        "sample_path": "data/processed/personal_task_sample.jsonl",
        "labels_path": "data/processed/personal_task_labels.jsonl",
    },
    "batch2": {
        "sample_path": "data/processed/personal_task_sample_batch2.jsonl",
        "labels_path": "data/processed/personal_task_labels_batch2.jsonl",
    },
}

VALID_LABELS = {"p": "personal", "t": "task", "x": "exclude"}


def load_jsonl(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(sample_path: str, labels_path: str):
    sample = load_jsonl(sample_path)
    if not sample:
        print(f"No sample found at {sample_path}.")
        return

    existing = {r["message_id"]: r for r in load_jsonl(labels_path)}
    todo = remaining_to_label(sample, existing)

    total = len(sample)
    print(f"{len(existing)}/{total} already labeled. {len(todo)} remaining.")
    print("For each message: [p]ersonal, [t]ask, [x] exclude (genuinely ambiguous/mixed), [q]uit\n")

    with open(labels_path, "a", encoding="utf-8") as out:
        for i, record in enumerate(todo):
            done = len(existing) + i
            split_tag = f" (split: {record['split']})" if "split" in record else ""
            print(f"--- {done + 1}/{total}{split_tag} ---")
            title = record.get("conversation_title") or "(untitled)"
            print(f"[from: {title}]")
            text = record["text"]
            if len(text) > 1500:
                text = text[:1500] + "... [truncated]"
            print(text)
            print()

            while True:
                choice = input("label (p/t/x/q): ").strip().lower()
                if choice == "q":
                    print(f"\nStopped. {done}/{total} labeled so far. Run again to resume.")
                    return
                if choice in VALID_LABELS:
                    break
                print("Please enter p, t, x, or q.")

            record_out = {
                "message_id": record["message_id"],
                "conversation_id": record["conversation_id"],
                "label": VALID_LABELS[choice],
            }
            if "split" in record:
                record_out["split"] = record["split"]
            out.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            out.flush()
            print()

    print(f"\nAll {total} messages labeled.")


if __name__ == "__main__":
    batch_name = sys.argv[1] if len(sys.argv) > 1 else "batch1"
    if batch_name not in BATCHES:
        print(f"Unknown batch '{batch_name}'. Choose from: {list(BATCHES.keys())}")
        sys.exit(1)
    paths = BATCHES[batch_name]
    main(paths["sample_path"], paths["labels_path"])
