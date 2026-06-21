"""Ollama embedding service with TF-IDF fallback."""

from typing import List, Optional

import httpx
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMS = {"ollama": 768}

_tfidf_vectorizer = HashingVectorizer(
    n_features=EMBEDDING_DIMS["ollama"],
    ngram_range=(1, 2),
    alternate_sign=False,
    norm="l2",
    binary=False,
)


async def embed_texts(
    texts: List[str], model: Optional[str] = None
) -> List[np.ndarray]:
    """Embed texts using local Ollama or TF-IDF fallback."""
    if not texts:
        return []

    if not model:
        model = DEFAULT_EMBEDDING_MODEL

    try:
        return await _embed_with_ollama(texts, model)
    except Exception:
        return _embed_with_tfidf(texts)


async def embed_query(text: str, model: Optional[str] = None) -> np.ndarray:
    """Embed a single query text."""
    results = await embed_texts([text], model=model)
    return (
        results[0] if results else np.zeros(EMBEDDING_DIMS["ollama"], dtype=np.float32)
    )


async def _embed_with_ollama(texts: List[str], model: str) -> List[np.ndarray]:
    """Embed texts using local Ollama."""
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for text in texts:
            resp = await client.post(
                "http://localhost:11434/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            embedding = resp.json()["embedding"]
            results.append(np.array(embedding, dtype=np.float32))
    return results


def _embed_with_tfidf(texts: List[str]) -> List[np.ndarray]:
    """Embed texts using local TF-IDF hashing."""
    rows = _tfidf_vectorizer.transform(texts)
    return [
        np.asarray(rows[i].toarray(), dtype=np.float32).flatten()
        for i in range(rows.shape[0])
    ]
