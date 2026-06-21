from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.embeddings import embed_query
from services.llm import generate_answer
from services.store import store

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_ids: Optional[List[str]] = None  # filter to specific docs
    chat_history: Optional[List[dict]] = None  # [{role, content}, ...]
    return_chunks: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    chunks: Optional[List[dict]] = None


@router.post(
    "/", response_model=QueryResponse, summary="Ask a question over your documents"
)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    if store.get_stats()["chunk_count"] == 0:
        raise HTTPException(
            422, "No documents ingested yet. Upload some documents first."
        )

    query_emb = await embed_query(req.question)
    chunks = store.search(query_emb, top_k=req.top_k, doc_ids=req.doc_ids)

    if not chunks:
        return QueryResponse(
            answer="No relevant content found in the uploaded documents.",
            sources=[],
        )

    try:
        answer = await generate_answer(req.question, chunks, req.chat_history)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

    sources = [
        {
            "doc_id": c["doc_id"],
            "doc_name": c["doc_name"],
            "doc_type": c["doc_type"],
            "score": round(c["score"], 4),
            "excerpt": c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
        }
        for c in chunks
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks=chunks if req.return_chunks else None,
    )
