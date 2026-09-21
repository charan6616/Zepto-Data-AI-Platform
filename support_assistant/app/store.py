"""ChromaDB persistence wrapper.

Collection: "zepto_policies" (cosine space), persisted to data/chroma/.
Embeddings are always supplied explicitly by this module — ChromaDB never uses
its own (network-hitting) default embedding function, keeping every retrieval
call local and offline-capable.
"""

from pathlib import Path

from chromadb import PersistentClient
from chromadb.config import Settings

from .embeddings import embed_query, embed_texts

COLLECTION_NAME = "zepto_policies"
TOP_K = 3
DB_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = PersistentClient(
            path=str(DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_chunks(ids: list[str], texts: list[str], metadatas: list[dict] | None = None):
    """Store chunk texts + locally computed embeddings (idempotent)."""
    collection = get_collection()
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embed_texts(texts),
        metadatas=metadatas or [{"doc": i} for i in ids],
    )
    return len(ids)


def count_chunks() -> int:
    collection = get_collection()
    return collection.count()


def query_chunks(query: str, n: int = TOP_K) -> list[tuple[str, str, float]]:
    """Retrieve the top-n most similar chunks by cosine similarity (always runs for real)."""
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    distances = result["distances"][0]
    return list(zip(ids, documents, distances))