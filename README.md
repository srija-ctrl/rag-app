# RAG App — FastAPI + Ollama

Retrieval-Augmented Generation over PDFs, DOCX, URLs, and text files using local Ollama.

## Requirements

- Python 3.10+
- Ollama running locally on `http://localhost:11434`
- Ollama models installed: `llama3.2` and `nomic-embed-text`

## Install

```bash
cd rag-app
pip install -r requirements.txt
```

## Run locally

1. Start Ollama locally.
2. Run the FastAPI app:

```bash
uvicorn main:app --reload --port 8000
```

3. Open the UI: http://localhost:8000
4. Use the API docs: http://localhost:8000/docs

## How it works

- **LLM**: local Ollama via `/api/generate`
- **Embeddings**: local Ollama via `/api/embed`, with TF-IDF fallback
- **Vector store**: in-memory cosine search
- **Documents**: PDF, DOCX, TXT/MD, URL ingestion

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

## Deployment

This app can be deployed to any Python host that exposes port `8000`.

```bash
docker build -t rag-app .
docker run -p 8000:8000 rag-app
```

## Notes

- The app assumes Ollama is available at `http://localhost:11434`.
- If Ollama embedding is unavailable, the service falls back to TF-IDF embeddings.
- The health endpoint reports the configured local Ollama models.
