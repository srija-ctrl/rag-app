import os
import httpx
from typing import List, Dict
from httpx import HTTPStatusError

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL = "gemini-2.0-flash"  # free tier available; swap to gemini-1.5-pro for higher quality


async def generate_answer(query: str, chunks: List[Dict], chat_history: List[Dict] = None) -> str:
    if not GOOGLE_API_KEY:
        top = chunks[0]["text"][:300] if chunks else "none"
        return (
            f"⚠️ GOOGLE_API_KEY not set. Set it to enable AI answers.\n\n"
            f"Top retrieved chunk: \"{top}…\""
        )

    context = "\n\n---\n\n".join(
        f"[Source {i+1}: {c['doc_name']} | score {c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system = """You are a helpful RAG assistant. Answer questions using only the provided document context.
- Cite the source document name when relevant (e.g. "According to report.pdf…").
- If the answer isn't in the context, say so — never make things up.
- Be concise and clear. Use bullet points for lists."""

    # Build Gemini contents array (system + history + current turn)
    contents = []
    for turn in (chat_history or [])[-6:]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({
        "role": "user",
        "parts": [{"text": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"}]
    })

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
        except ValueError:
            raise RuntimeError(f"Invalid LLM response body: {resp.text[:300]}")
