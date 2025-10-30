# app/api/routers/ingest.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from app.core.config import settings
from app.rag.retriever import Retriever

router = APIRouter()

# простой чанкер
def _chunk_text(text: str, size: int = 800, overlap: int = 100):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + size
        chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


@router.post("/docs")
async def ingest_docs(files: List[UploadFile] = File(...)):
    try:
        retriever = Retriever(
            backend=settings.embedding_backend,
            model=settings.embedding_model,
            index_dir=settings.rag_index_path,
        )
    except Exception as e:
        return JSONResponse(
            {"ingested_chunks": 0, "errors": [f"retriever_init: {e}"]},
            status_code=500,
        )

    total_chunks = 0
    errors = []

    for f in files:
        try:
            raw = await f.read()

            if len(raw) > 2_000_000:
                raw = raw[:2_000_000]

            # декод
            text = None
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = raw.decode("cp1251")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1", errors="ignore")

            if not text:
                errors.append(f"{f.filename}: cannot decode")
                continue

            chunks = _chunk_text(text, size=800, overlap=100)
            if not chunks:
                errors.append(f"{f.filename}: empty after chunking")
                continue

            # вот тут чаще всего и падает
            try:
                retriever.add_texts(chunks)
                total_chunks += len(chunks)
            except Exception as e:
                errors.append(f"{f.filename}: embed/index error: {e}")
                continue

        except Exception as e:
            print(f"[ingest] error on file {f.filename}: {e}")
            errors.append(f"{f.filename}: {e}")

    return JSONResponse(
        {
            "ingested_chunks": total_chunks,
            "errors": errors,
        }
    )
