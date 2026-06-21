"""
Ollama embeddings — nomic-embed-text
Local, offline, no API keys required.
~768 dimensions, good semantic quality.
"""

from typing import List

import httpx
import numpy as np

EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"


async def embed_texts(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
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


async def embed_query(text: str) -> np.ndarray:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float32)
