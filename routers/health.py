import os

from fastapi import APIRouter

from services.store import store

router = APIRouter()


@router.get("/health", tags=["Health"])
def health():
    key_set = bool(os.environ.get("GOOGLE_API_KEY"))
    return {
        "status": "ok",
        "google_key_set": key_set,
        "llm_model": "gemini-1.5-flash",
        "embedding_model": "tfidf-local",
        **store.get_stats(),
    }
