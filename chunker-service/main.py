"""
Chunker Service actualizado — Agrega estrategia "section_aware".
Reemplazar main.py con este contenido.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataclasses import asdict

import uvicorn

from extractors.pdf import extract_text
from chunkers.recursive import recursive_chunk
from chunkers.section_aware import section_aware_chunk  # NUEVO

app = FastAPI(title="RAG Chunker Service", version="1.1.0")


class ChunkRequest(BaseModel):
    file_path: str
    strategy: str = "section_aware"  # cambiado el default
    chunk_size: int = 512
    overlap: int = 80
    doc_id: str


class ChunkMetadata(BaseModel):
    doc_id: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int
    section_title: str | None = None
    section_number: str | None = None


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class ChunkResponse(BaseModel):
    chunks: list[Chunk]
    total: int
    strategy_used: str


@app.post("/chunk", response_model=ChunkResponse)
async def chunk_document(req: ChunkRequest):
    print(f"[chunk] {req.file_path} | strategy={req.strategy}")

    try:
        pages = extract_text(req.file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {req.file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extrayendo texto: {str(e)}")

    try:
        if req.strategy == "recursive":
            chunks = recursive_chunk(
                pages=pages,
                doc_id=req.doc_id,
                source=req.file_path,
                chunk_size=req.chunk_size,
                overlap=req.overlap,
            )
        elif req.strategy == "section_aware":
            # chunk_size en el request está en tokens, lo convertimos a chars
            # ~6 chars/token para español es una buena estimación
            max_chars = req.chunk_size * 6   # 512 tokens → ~3072 chars
            overlap_chars = req.overlap * 5  # 80 tokens → ~480 chars
            chunks = section_aware_chunk(
                pages=pages,
                doc_id=req.doc_id,
                source=req.file_path,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy desconocida: '{req.strategy}'. Opciones: recursive, section_aware"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chunking: {str(e)}")

    print(f"[chunk] ✓ {len(chunks)} chunks generados de {len(pages)} páginas")

    print(type(chunks[0]))

    return ChunkResponse(
        chunks=[asdict(c) for c in chunks],
        total=len(chunks),
        strategy_used=req.strategy,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "service": "chunker", "version": "1.1.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)