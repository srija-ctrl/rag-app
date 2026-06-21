from fastapi import APIRouter

from services.store import store

DEFAULT_LLAMA_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

router = APIRouter()


@router.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "llm_model": DEFAULT_LLAMA_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        **store.get_stats(),
    }
