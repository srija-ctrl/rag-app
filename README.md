# RAG App — FastAPI + OpenAI

Retrieval-Augmented Generation over PDFs, DOCX, URLs, and text files.

## Stack
- **LLM**: GPT-4o-mini (swap to `gpt-4o` in `services/llm.py` for higher quality)
- **Embeddings**: text-embedding-3-small (~$0.00002 / 1K tokens)
- **Vector store**: in-memory cosine similarity (scikit-learn)
- **Documents**: PDF, DOCX, TXT/MD, Web URLs

## Run locally

```bash
cd rag-app
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload --port 8000
```

- UI → http://localhost:8000  
- API docs → http://localhost:8000/docs

## Deploy to Render (free tier)

1. Push folder to GitHub
2. render.com → **New Web Service** → connect repo  
   (Render reads `render.yaml` automatically)
3. Environment tab → add `OPENAI_API_KEY`
4. Deploy → live in ~2 min

## Deploy with Docker

```bash
docker build -t rag-app .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... rag-app
```

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/documents/upload` | Upload PDF, DOCX, TXT, MD |
| POST | `/api/documents/url` | Ingest a web page |
| GET  | `/api/documents/` | List documents |
| DELETE | `/api/documents/{id}` | Remove a document |
| POST | `/api/query/` | Ask a question |
| GET  | `/api/health` | Status + model info |

### Query example
```json
POST /api/query/
{
  "question": "What are the key findings?",
  "top_k": 5,
  "doc_ids": ["abc123"],
  "chat_history": []
}
```

## Upgrade to GPT-4o

In `services/llm.py`, change:
```python
MODEL = "gpt-4o"
```
