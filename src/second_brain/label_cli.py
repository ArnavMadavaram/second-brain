import json
from pathlib import Path

from second_brain.labeling import remaining_to_label

SAMPLE_PATH = "data/processed/personal_task_sample.jsonl"
LABELS_PATH = "data/processed/personal_task_labels.jsonl"

VALID_LABELS = {"p": "personal", "t": "task", "x": "exclude"}


def load_jsonl(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    sample = load_jsonl(SAMPLE_PATH)
    if not sample:
        print(f"No sample found at {SAMPLE_PATH}. Run `python -m second_brain.labeling` first.")
        return

    existing = {r["message_id"]: r for r in load_jsonl(LABELS_PATH)}
    todo = remaining_to_label(sample, existing)

    total = len(sample)
    print(f"{len(existing)}/{total} already labeled. {len(todo)} remaining.")
    print("For each message: [p]ersonal, [t]ask, [x] exclude (genuinely ambiguous/mixed), [q]uit\n")

    with open(LABELS_PATH, "a", encoding="utf-8") as out:
        for i, record in enumerate(todo):
            done = len(existing) + i
            print(f"--- {done + 1}/{total} (split: {record['split']}) ---")
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

            out.write(json.dumps({
                "message_id": record["message_id"],
                "conversation_id": record["conversation_id"],
                "split": record["split"],
                "label": VALID_LABELS[choice],
            }, ensure_ascii=False) + "\n")
            out.flush()
            print()

    print(f"\nAll {total} messages labeled.")


if __name__ == "__main__":
    main()
