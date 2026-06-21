"""Data models for the RAG application."""

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Query request for Ollama-based RAG."""

    question: str
    top_k: int = 5
    doc_ids: Optional[List[str]] = None
    chat_history: Optional[List[dict]] = None
    return_chunks: bool = False


class QueryResponse(BaseModel):
    """Query response with sources."""

    answer: str
    sources: List[dict]
    chunks: Optional[List[dict]] = None
