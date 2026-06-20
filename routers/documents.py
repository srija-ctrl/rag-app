import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from pydantic import BaseModel

from services.store import store, Document, Chunk, make_doc_id, make_chunk_id
from services.ingestion import extract_pdf, extract_docx, extract_text, extract_url, split_text
from services.embeddings import embed_texts

router = APIRouter()


class URLIngest(BaseModel):
    url: str
    name: Optional[str] = None


@router.post("/upload", summary="Upload a PDF, DOCX, or text/markdown file")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        text = extract_pdf(data)
        doc_type = "pdf"
    elif ext in ("docx", "doc"):
        text = extract_docx(data)
        doc_type = "docx"
    elif ext in ("txt", "md", "markdown", "rst", "csv"):
        text = extract_text(data)
        doc_type = "text"
    else:
        raise HTTPException(400, f"Unsupported file type: .{ext}. Use PDF, DOCX, TXT, or MD.")

    if not text.strip():
        raise HTTPException(422, "Could not extract any text from the file.")

    return await _ingest(text, filename, doc_type)


@router.post("/url", summary="Ingest a web page by URL")
async def ingest_url(body: URLIngest):
    try:
        title, text = await extract_url(body.url)
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch URL: {e}")

    if not text.strip():
        raise HTTPException(422, "Could not extract text from the URL.")

    name = body.name or title or body.url
    return await _ingest(text, name, "url", metadata={"source_url": body.url})


@router.get("/", summary="List all ingested documents")
def list_documents():
    docs = store.list_documents()
    return {
        "documents": [
            {
                "id": d.id,
                "name": d.name,
                "type": d.doc_type,
                "chunk_count": d.chunk_count,
                "char_count": d.char_count,
                "uploaded_at": d.uploaded_at,
            }
            for d in sorted(docs, key=lambda x: x.uploaded_at, reverse=True)
        ],
        "stats": store.get_stats(),
    }


@router.delete("/{doc_id}", summary="Delete a document and its chunks")
def delete_document(doc_id: str):
    if not store.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


# ── Internal helper ─────────────────────────────────────────────────────────────

async def _ingest(text: str, name: str, doc_type: str, metadata: dict = None) -> dict:
    chunks_text = split_text(text)
    if not chunks_text:
        raise HTTPException(422, "No usable text content found.")

    embeddings = await embed_texts(chunks_text)

    doc_id = make_doc_id(name)
    doc = Document(
        id=doc_id,
        name=name,
        doc_type=doc_type,
        chunk_count=len(chunks_text),
        char_count=len(text),
        uploaded_at=time.time(),
        metadata=metadata or {},
    )

    chunks = [
        Chunk(
            id=make_chunk_id(doc_id, i),
            doc_id=doc_id,
            doc_name=name,
            doc_type=doc_type,
            text=ct,
            embedding=emb,
            metadata={"chunk_index": i, **(metadata or {})},
        )
        for i, (ct, emb) in enumerate(zip(chunks_text, embeddings))
    ]

    store.add_document(doc, chunks)

    return {
        "doc_id": doc_id,
        "name": name,
        "type": doc_type,
        "chunks": len(chunks),
        "chars": len(text),
    }
