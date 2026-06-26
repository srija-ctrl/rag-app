"""LLM service supporting local Ollama and Google Gemini."""

import os
from typing import Dict, List, Optional

import httpx
from httpx import HTTPStatusError
from google import genai

DEFAULT_LLAMA_MODEL = "llama3.2"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


async def generate_answer(
    query: str,
    chunks: List[Dict],
    model: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
) -> str:
    """Generate an answer using Google Gemini (default) or local Ollama."""
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}: {c['doc_name']} | score {c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system = """You are a helpful RAG assistant. Answer questions using only the provided document context.
- Cite the source document name when relevant (e.g. "According to report.pdf…").
- If the answer isn't in the context, say so — never make things up.
- Be concise and clear. Use bullet points for lists."""

    if not model:
        model = DEFAULT_GEMINI_MODEL

    history_text = ""
    if chat_history:
        history_lines = []
        for turn in chat_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_lines.append(f"{role.capitalize()}: {content}")
        if history_lines:
            history_text = "\n\nChat history:\n" + "\n".join(history_lines)

    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {query}{history_text}"
    
    if model.startswith("gemini"):
        return await _generate_with_gemini(prompt, model)
    else:
        return await _generate_with_ollama(prompt, model)


async def _generate_with_gemini(prompt: str, model: str) -> str:
    """Generate response using Google Gemini."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text or ""


async def _generate_with_ollama(prompt: str, model: str) -> str:
    """Generate response using local Ollama."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )
        try:
            resp.raise_for_status()
            payload = resp.json()
            return payload["response"]
        except HTTPStatusError as exc:
            body = exc.response.text
            raise RuntimeError(f"Ollama API error {exc.response.status_code}: {body}")
        except (ValueError, KeyError):
            raise RuntimeError(f"Invalid Ollama response body: {resp.text[:300]}")
