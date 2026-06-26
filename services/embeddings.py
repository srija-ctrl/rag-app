"""Embedding service supporting Google Gemini, Ollama, and TF-IDF."""

import os
from typing import List, Optional

import httpx
import numpy as np
from google import genai
from sklearn.feature_extraction.text import HashingVectorizer

DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_GEMINI_MODEL = "text-embedding-004"
EMBEDDING_DIMS = {"ollama": 768, "gemini": 768}

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
    """Embed texts using Gemini, Ollama, or TF-IDF fallback."""
    if not texts:
        return []

    if not model:
        model = DEFAULT_GEMINI_MODEL

    try:
        if "gemini" in model or "embedding" in model:
            return await _embed_with_gemini(texts, model)
        else:
            return await _embed_with_ollama(texts, model)
    except Exception:
        return _embed_with_tfidf(texts)


async def embed_query(text: str, model: Optional[str] = None) -> np.ndarray:
    """Embed a single query text."""
    results = await embed_texts([text], model=model)
    if not model:
        model = DEFAULT_GEMINI_MODEL
    dim_key = "gemini" if "gemini" in model or "embedding" in model else "ollama"
    return (
        results[0] if results else np.zeros(EMBEDDING_DIMS[dim_key], dtype=np.float32)
    )

async def _embed_with_gemini(texts: List[str], model: str) -> List[np.ndarray]:
    """Embed texts using Google Gemini."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = await client.aio.models.embed_content(
        model=model,
        contents=texts,
    )
    return [np.array(e.values, dtype=np.float32) for e in response.embeddings]


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
