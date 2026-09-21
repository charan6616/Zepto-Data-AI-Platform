"""Local embedding wrapper around sentence-transformers (all-MiniLM-L6-v2).

No API key and no network call are required at inference time — the model runs
entirely on this machine. It is downloaded from the Hugging Face Hub once on
first use and cached locally afterwards.
"""

from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents/chunks, L2-normalized so cosine ≈ dot product."""
    vectors = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query, L2-normalized."""
    return _model().encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()