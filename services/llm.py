import os
from typing import Dict, List, Optional

import httpx
from httpx import HTTPStatusError

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


async def generate_answer(
    query: str, chunks: List[Dict], chat_history: Optional[List[Dict]] = None
) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}: {c['doc_name']} | score {c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system = """You are a helpful RAG assistant. Answer questions using only the provided document context.
- Cite the source document name when relevant (e.g. \"According to report.pdf…\").
- If the answer isn't in the context, say so — never make things up.
- Be concise and clear. Use bullet points for lists."""

    # Try cloud APIs first if configured
    if OPENAI_API_KEY and not GOOGLE_API_KEY:
        return await _generate_with_openai(query, context, system, chat_history)
    elif GOOGLE_API_KEY:
        return await _generate_with_google(query, context, system, chat_history)

    # Default to local Ollama
    return await _generate_with_ollama(query, context, system)


async def _generate_with_openai(
    query: str,
    context: str,
    system: str,
    chat_history: Optional[List[Dict]],
) -> str:
    messages = [{"role": "system", "content": system}]
    for turn in (chat_history or [])[-6:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}",
        }
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": messages,
                "max_tokens": 1024,
            },
        )
        try:
            resp.raise_for_status()
            payload = resp.json()
            return payload["choices"][0]["message"]["content"]
        except HTTPStatusError as exc:
            body = exc.response.text
            raise RuntimeError(f"OpenAI API error {exc.response.status_code}: {body}")
        except (ValueError, KeyError):
            raise RuntimeError(f"Invalid OpenAI response body: {resp.text[:300]}")


async def _generate_with_google(
    query: str,
    context: str,
    system: str,
    chat_history: Optional[List[Dict]],
) -> str:
    # This function is now mapped to Google Gemini API
    # Using the old Google API endpoint (if GOOGLE_API_KEY is set)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": f"{system}\n\nContext:\n{context}\n\nQuestion: {query}",
                "stream": False,
            },
        )
        try:
            resp.raise_for_status()
            payload = resp.json()
            return payload["response"]
        except HTTPStatusError as exc:
            body = exc.response.text
            raise RuntimeError(f"LLM API error {exc.response.status_code}: {body}")
        except (ValueError, KeyError):
            raise RuntimeError(f"Invalid LLM response body: {resp.text[:300]}")


async def _generate_with_ollama(
    query: str,
    context: str,
    system: str,
) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": f"{system}\n\nContext:\n{context}\n\nQuestion: {query}",
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
