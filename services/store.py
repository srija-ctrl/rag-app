import numpy as np
import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    doc_type: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    id: str
    name: str
    doc_type: str
    chunk_count: int
    char_count: int
    uploaded_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.documents: Dict[str, Document] = {}

    def add_document(self, doc: Document, chunks: List[Chunk]):
        self.documents[doc.id] = doc
        self.chunks.extend(chunks)

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 5, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        candidates = self.chunks
        if doc_ids:
            candidates = [c for c in candidates if c.doc_id in doc_ids]
        if not candidates:
            return []

        embeddings = np.stack([c.embedding for c in candidates])
        if embeddings.ndim != 2 or query_embedding.ndim != 1:
            raise ValueError(
                f"Embedding dimensions invalid: embeddings {embeddings.shape}, query {query_embedding.shape}"
            )
        if embeddings.shape[1] != query_embedding.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: embeddings dim {embeddings.shape[1]}, query dim {query_embedding.shape[0]}"
            )
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normed = embeddings / norms
        print("normed shape:", normed.shape)
        print("query shape:", q.shape)
        scores = normed @ q

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            c = candidates[idx]
            results.append({
                "chunk_id": c.id,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "doc_type": c.doc_type,
                "text": c.text,
                "score": float(scores[idx]),
                "metadata": c.metadata,
            })
        return results

    def list_documents(self) -> List[Document]:
        return list(self.documents.values())

    def get_stats(self) -> Dict:
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "total_chars": sum(len(c.text) for c in self.chunks),
        }


def make_doc_id(name: str) -> str:
    return hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]


def make_chunk_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}_chunk_{idx}"


store = VectorStore()
