"""
OpenAI embeddings — text-embedding-3-small
Fast, cheap (~$0.00002/1K tokens), 1536 dims, great semantic quality.
"""

import os
import numpy as np
import httpx
from typing import List

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 512


async def embed_texts(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    if not OPENAI_API_KEY:
        return _tfidf_fallback(texts)
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": batch},
            )
            resp.raise_for_status()
            for item in sorted(resp.json()["data"], key=lambda x: x["index"]):
                results.append(np.array(item["embedding"], dtype=np.float32))
    return results


async def embed_query(text: str) -> np.ndarray:
    if not OPENAI_API_KEY:
        return _tfidf_fallback([text])[0]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": [text]},
        )
        resp.raise_for_status()
        return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


# ── TF-IDF fallback (no key) ──────────────────────────────────────────────────
from sklearn.feature_extraction.text import HashingVectorizer
_vectorizer = HashingVectorizer(
    n_features=EMBED_DIM,
    ngram_range=(1, 2),
    alternate_sign=False,
    norm='l2',
    binary=False,
)

def _tfidf_fallback(texts: List[str]) -> List[np.ndarray]:
    rows = _vectorizer.transform(texts)
    return [np.asarray(rows[i].toarray(), dtype=np.float32).flatten() for i in range(rows.shape[0])]
