import glob
import json
from collections import Counter
from pathlib import Path


def build_census(records: list[dict], null_message_count: int) -> dict:
    return {
        "role_counts": dict(Counter(r["role"] for r in records)),
        "content_type_counts": dict(Counter(r["content_type"] for r in records)),
        "null_message_count": null_message_count,
        "off_path_count": sum(1 for r in records if not r["on_active_path"]),
        "attachment_count": sum(1 for r in records if r["has_attachment"]),
        "total_records": len(records),
    }


def extract_text_and_attachment(content: dict) -> tuple[str, bool]:
    content_type = content.get("content_type")

    # Reasoning-model content types (o1/o3-style) don't use "parts" at all --
    # they have their own shape and would silently come back empty otherwise.
    if content_type == "reasoning_recap":
        return content.get("content") or "", False

    if content_type == "thoughts":
        thoughts = content.get("thoughts") or []
        text = "\n".join(t.get("content", "") for t in thoughts if isinstance(t, dict))
        return text, False

    parts = content.get("parts", [])
    string_parts = [p for p in parts if isinstance(p, str)]
    has_attachment = len(string_parts) != len(parts)
    return "".join(string_parts), has_attachment


def compute_active_path_ids(conversation: dict) -> set[str]:
    mapping = conversation["mapping"]
    active_ids: set[str] = set()
    node_id = conversation["current_node"]
    while node_id is not None:
        active_ids.add(node_id)
        node_id = mapping[node_id]["parent"]
    return active_ids


def parse_conversation(conversation: dict) -> list[dict]:
    active_path_ids = compute_active_path_ids(conversation)

    records = []
    for node in conversation["mapping"].values():
        message = node["message"]
        if message is None:
            continue

        content = message.get("content", {})
        text, has_attachment = extract_text_and_attachment(content)

        records.append({
            "conversation_id": conversation["conversation_id"],
            "conversation_title": conversation["title"],
            "message_id": node["id"],
            "parent_id": node["parent"],
            "role": message.get("author", {}).get("role"),
            "create_time": message.get("create_time"),
            "content_type": content.get("content_type"),
            "text": text,
            "on_active_path": node["id"] in active_path_ids,
            "has_attachment": has_attachment,
        })
    return records


def run(raw_dir: str, output_path: str) -> dict:
    shard_paths = sorted(glob.glob(str(Path(raw_dir) / "conversations-*.json")))
    if not shard_paths:
        raise FileNotFoundError(f"No conversations-*.json shards found in {raw_dir}")

    all_records: list[dict] = []
    null_message_count = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as f:
                conversations = json.load(f)

            for conversation in conversations:
                null_message_count += sum(
                    1 for node in conversation["mapping"].values() if node["message"] is None
                )
                records = parse_conversation(conversation)
                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_records.extend(records)

    return build_census(all_records, null_message_count)


def main():
    raw_dir = "data/raw/chatGPTExport"
    output_path = "data/processed/messages.jsonl"

    census = run(raw_dir, output_path)

    print("=== CENSUS ===")
    print(f"Total records:                {census['total_records']}")
    print(f"Role counts:                  {census['role_counts']}")
    print(f"Content type counts:          {census['content_type_counts']}")
    print(f"Null-message roots skipped:   {census['null_message_count']}")
    print(f"Off-active-path nodes:        {census['off_path_count']}")
    print(f"Records with attachments:     {census['attachment_count']}")


if __name__ == "__main__":
    main()
