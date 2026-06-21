"""
Ollama embeddings with TF-IDF fallback
Uses Ollama nomic-embed-text if server supports embeddings,
falls back to TF-IDF hashing for basic semantic search.
"""

from typing import List

import httpx
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
OLLAMA_URL = "http://localhost:11434"

_vectorizer = HashingVectorizer(
    n_features=EMBED_DIM,
    ngram_range=(1, 2),
    alternate_sign=False,
    norm="l2",
    binary=False,
)


async def embed_texts(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []

    # Try Ollama first
    try:
        results = []
        async with httpx.AsyncClient(timeout=60) as client:
            for text in texts:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": EMBED_MODEL, "input": text},
                )
                resp.raise_for_status()
                embedding = resp.json()["embedding"]
                results.append(np.array(embedding, dtype=np.float32))
        return results
    except Exception:
        # Fall back to TF-IDF if Ollama is not available
        return _tfidf_fallback(texts)


async def embed_query(text: str) -> np.ndarray:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
            )
            resp.raise_for_status()
            return np.array(resp.json()["embedding"], dtype=np.float32)
    except Exception:
        # Fall back to TF-IDF if Ollama is not available
        return _tfidf_fallback([text])[0]


def _tfidf_fallback(texts: List[str]) -> List[np.ndarray]:
    rows = _vectorizer.transform(texts)
    return [
        np.asarray(rows[i].toarray(), dtype=np.float32).flatten()
        for i in range(rows.shape[0])
    ]
