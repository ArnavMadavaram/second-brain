import json
import random


def sample_messages_for_labeling(records: list[dict], n: int = 250, seed: int = 42) -> list[dict]:
    """Sample up to n user messages, at most one per conversation, for diversity."""
    candidates = [r for r in records if r["role"] == "user" and r["text"].strip()]

    by_conversation: dict[str, list[dict]] = {}
    for r in candidates:
        by_conversation.setdefault(r["conversation_id"], []).append(r)

    rng = random.Random(seed)
    conversation_ids = list(by_conversation.keys())
    rng.shuffle(conversation_ids)

    sampled = []
    for conv_id in conversation_ids[:n]:
        sampled.append(rng.choice(by_conversation[conv_id]))
    return sampled


def assign_split(sampled: list[dict], test_size: int = 200, seed: int = 43) -> list[dict]:
    """Tag each sampled record with split="test" or split="dev". Shuffles before splitting."""
    rng = random.Random(seed)
    shuffled = sampled[:]
    rng.shuffle(shuffled)

    result = []
    for i, r in enumerate(shuffled):
        record = dict(r)
        record["split"] = "test" if i < test_size else "dev"
        result.append(record)
    return result


def remaining_to_label(sample: list[dict], existing_labels: dict[str, dict]) -> list[dict]:
    return [r for r in sample if r["message_id"] not in existing_labels]


def main():
    messages_path = "data/processed/messages.jsonl"
    output_path = "data/processed/personal_task_sample.jsonl"

    with open(messages_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    sampled = sample_messages_for_labeling(records, n=250, seed=42)
    split = assign_split(sampled, test_size=200, seed=43)

    with open(output_path, "w", encoding="utf-8") as out:
        for record in split:
            out.write(json.dumps({
                "message_id": record["message_id"],
                "conversation_id": record["conversation_id"],
                "conversation_title": record.get("conversation_title"),
                "text": record["text"],
                "split": record["split"],
            }, ensure_ascii=False) + "\n")

    dev_count = sum(1 for r in split if r["split"] == "dev")
    test_count = sum(1 for r in split if r["split"] == "test")
    print(f"Sampled {len(split)} messages from {len(set(r['conversation_id'] for r in split))} distinct conversations")
    print(f"  dev:  {dev_count}")
    print(f"  test: {test_count}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
