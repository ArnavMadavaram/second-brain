"""Phase 4 baseline retrieval: embed the parsed corpus into ChromaDB.

Chunking decision (frozen for reuse in Phase 8 -- the custom emotion model
comparison must reuse this unchanged so the comparison stays fair): one
chunk per parsed message record. No merging of multi-turn windows, no
splitting of long messages. Simplest possible choice for a baseline; if
Phase 7/8 show retrieval quality is bottlenecked by chunk granularity,
revisit deliberately rather than silently drifting.

Metadata fields per document, orthogonal filters per the locked architecture:
- role: "user" | "assistant" | other roles seen in the census
- is_personal + personal_score: present only on role="user" docs where the
  Attempt 2 classifier ran (assistant docs have no is_personal key at all,
  rather than a misleading default -- "not applicable" is represented by
  absence, not by False)
- on_active_path, has_attachment, content_type, conversation_id,
  conversation_title, create_time: carried through from the parser
"""
import json

import chromadb
from chromadb.utils import embedding_functions

MESSAGES_PATH = "data/processed/messages.jsonl"
PREDICTIONS_PATH = "data/processed/is_personal_predictions.jsonl"
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "messages_baseline"
BATCH_SIZE = 200


def build_metadata(record: dict, personal_pred: dict | None) -> dict:
    metadata = {
        "role": record["role"],
        "conversation_id": record["conversation_id"],
        "conversation_title": record.get("conversation_title") or "",
        "on_active_path": record["on_active_path"],
        "has_attachment": record["has_attachment"],
        "content_type": record["content_type"] or "",
        "create_time": record["create_time"],
    }
    if personal_pred is not None:
        metadata["is_personal"] = personal_pred["is_personal"]
        metadata["personal_score"] = personal_pred["score"]
    return metadata


def main():
    records = [json.loads(l) for l in open(MESSAGES_PATH, encoding="utf-8")]
    predictions = {p["message_id"]: p for p in (json.loads(l) for l in open(PREDICTIONS_PATH, encoding="utf-8"))}

    embeddable = [r for r in records if r["text"].strip()]
    print(f"Embedding {len(embeddable)} of {len(records)} records (skipping {len(records) - len(embeddable)} empty-text)")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)  # fresh collection each run, idempotent script

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    for i in range(0, len(embeddable), BATCH_SIZE):
        batch = embeddable[i : i + BATCH_SIZE]
        collection.add(
            ids=[r["message_id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[build_metadata(r, predictions.get(r["message_id"])) for r in batch],
        )
        if (i + BATCH_SIZE) % 2000 < BATCH_SIZE:
            print(f"  {min(i + BATCH_SIZE, len(embeddable))}/{len(embeddable)}")

    print(f"Wrote {collection.count()} documents to Chroma collection '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
