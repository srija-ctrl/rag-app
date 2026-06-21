import os
from typing import Dict, List, Optional

import httpx
from httpx import HTTPStatusError

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MODEL = "gemini-2.0-flash"


async def generate_answer(
    query: str, chunks: List[Dict], chat_history: Optional[List[Dict]] = None
) -> str:
    if not GOOGLE_API_KEY and not OPENAI_API_KEY:
        top = chunks[0]["text"][:300] if chunks else "none"
        return (
            "⚠️ No model API key set. Set GOOGLE_API_KEY or OPENAI_API_KEY to enable AI answers.\n\n"
            f'Top retrieved chunk: "{top}…"'
        )

    context = "\n\n---\n\n".join(
        f"[Source {i + 1}: {c['doc_name']} | score {c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system = """You are a helpful RAG assistant. Answer questions using only the provided document context.
- Cite the source document name when relevant (e.g. \"According to report.pdf…\").
- If the answer isn't in the context, say so — never make things up.
- Be concise and clear. Use bullet points for lists."""

    if OPENAI_API_KEY and not GOOGLE_API_KEY:
        return await _generate_with_openai(query, context, system, chat_history)

    return await _generate_with_google(query, context, system, chat_history)


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
    contents = []
    for turn in (chat_history or [])[-6:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn.get("content", "")}]})
    contents.append(
        {
            "role": "user",
            "parts": [
                {
                    "text": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}",
                }
            ],
        }
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GOOGLE_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": {"maxOutputTokens": 1024},
            },
        )
        try:
            resp.raise_for_status()
            payload = resp.json()
            return payload["candidates"][0]["content"]["parts"][0]["text"]
        except HTTPStatusError as exc:
            body = exc.response.text
            raise RuntimeError(f"LLM API error {exc.response.status_code}: {body}")
        except (ValueError, KeyError):
            raise RuntimeError(f"Invalid LLM response body: {resp.text[:300]}")
