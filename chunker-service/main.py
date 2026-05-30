"""
Chunker Service - Microservicio Python para extracción y chunking de PDFs.

Go delega el procesamiento de texto a este servicio porque Python tiene
un ecosistema mucho más maduro para esto (PyMuPDF, tiktoken, etc).

Endpoints:
    POST /chunk  - Extrae texto de un PDF y lo divide en chunks
    GET  /health - Verificación de estado
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataclasses import asdict
import uvicorn

from extractors.pdf import extract_text
from chunkers.recursive import recursive_chunk

app = FastAPI(
    title="RAG Chunker Service",
    description="Extracción y chunking de PDFs para el sistema RAG en Go",
    version="1.0.0",
)


# --- Modelos de request/response ---

class ChunkRequest(BaseModel):
    file_path: str
    strategy: str = "recursive"
    chunk_size: int = 512
    overlap: int = 80
    doc_id: str


class ChunkMetadata(BaseModel):
    doc_id: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class ChunkResponse(BaseModel):
    chunks: list[Chunk]
    total: int
    strategy_used: str


# --- Endpoints ---

@app.post("/chunk", response_model=ChunkResponse)
async def chunk_document(req: ChunkRequest):
    """
    Extrae texto de un PDF y lo divide en chunks.
    
    Estrategias disponibles:
    - "recursive": Recursive character splitting (Fase 1)
    - "hierarchical": Hierarchical chunking (Fase 2 - próximamente)
    """
    print(f"[chunk] {req.file_path} | strategy={req.strategy} | size={req.chunk_size} | overlap={req.overlap}")

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
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Estrategia '{req.strategy}' no soportada. Usar: recursive"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chunking: {str(e)}")

    print(f"[chunk] ✓ {len(chunks)} chunks generados de {len(pages)} páginas")

    return ChunkResponse(
        chunks=[asdict(chunk) for chunk in chunks],
        total=len(chunks),
        strategy_used=req.strategy,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chunker"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)