import chromadb
from chromadb.utils import embedding_functions

from second_brain.embed_corpus import CHROMA_PATH, COLLECTION_NAME


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


def query(collection, text: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
    result = collection.query(query_texts=[text], n_results=n_results, where=where)
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def print_hits(hits: list[dict]):
    for h in hits:
        m = h["metadata"]
        tag = f"role={m['role']}"
        if "is_personal" in m:
            tag += f" is_personal={m['is_personal']} (score={m['personal_score']:.2f})"
        print(f"  [dist={h['distance']:.3f}] [{tag}]")
        text = h["text"].replace("\n", " ")
        print(f"    {text[:150]}{'...' if len(text) > 150 else ''}")


def main():
    collection = get_collection()
    print(f"Collection has {collection.count()} documents\n")

    print("=== Query 1: unfiltered, mixing roles ===")
    print_hits(query(collection, "I've been feeling anxious", n_results=5))

    print("\n=== Query 2: same query, role=user AND is_personal=True (emotional retrieval view) ===")
    print_hits(query(
        collection, "I've been feeling anxious", n_results=5,
        where={"$and": [{"role": "user"}, {"is_personal": True}]},
    ))

    print("\n=== Query 3: role=assistant only (factual retrieval view) ===")
    print_hits(query(collection, "how do I calculate a derivative", n_results=5, where={"role": "assistant"}))


if __name__ == "__main__":
    main()
