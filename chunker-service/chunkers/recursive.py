"""
Recursive Character Chunking - Estrategia de chunking para Fase 1.

Intenta dividir el texto respetando separadores naturales en orden de prioridad:
1. Párrafos (\n\n) - máxima unidad semántica
2. Líneas (\n)
3. Oraciones (". ")
4. Espacios (" ")
5. Caracteres individuales (último recurso)

El overlap entre chunks garantiza que el contexto no se pierda en los bordes.
"""

from dataclasses import dataclass
from typing import Optional
import re
import sys

from extractors.pdf import PageContent

# Usamos tiktoken para contar tokens de forma precisa.
# cl100k_base es compatible con qwen y nomic-embed-text (aproximadamente).
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
        
except ImportError:
    print("[chunker] tiktoken no disponible, usando estimación (1 token ≈ 4 chars)", file=sys.stderr)
    
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# Separadores en orden de prioridad (de más a menos "natural")
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


@dataclass
class ChunkMetadata:
    doc_id: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int


@dataclass  
class Chunk:
    text: str
    metadata: ChunkMetadata


def recursive_chunk(
    pages: list[PageContent],
    doc_id: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 80,
) -> list[Chunk]:
    """
    Divide el texto de las páginas en chunks usando recursive character splitting.
    
    Args:
        pages: Lista de páginas con su texto y número
        doc_id: ID del documento para metadatos
        source: Nombre del archivo fuente
        chunk_size: Tamaño máximo del chunk en tokens
        overlap: Tokens de overlap con el chunk anterior
    
    Returns:
        Lista de Chunk con texto y metadatos completos
    """
    raw_chunks = []  # list of {"text": str, "page": int}
    
    for page in pages:
        page_texts = _split_text(page.text, chunk_size)
        
        for text in page_texts:
            text = text.strip()
            # Descartar chunks muy pequeños (ruido)
            if count_tokens(text) < 20:
                continue
            raw_chunks.append({"text": text, "page": page.page_num})
    
    # Aplicar overlap entre chunks consecutivos
    if overlap > 0:
        raw_chunks = _apply_overlap(raw_chunks, overlap)
    
    total = len(raw_chunks)
    result = []
    
    for i, c in enumerate(raw_chunks):
        result.append(Chunk(
            text=c["text"],
            metadata=ChunkMetadata(
                doc_id=doc_id,
                source=source,
                page=c["page"],
                chunk_index=i,
                total_chunks=total,
            )
        ))
    
    return result


def _split_text(text: str, chunk_size: int, separators: Optional[list[str]] = None) -> list[str]:
    """
    Divide el texto recursivamente probando separadores en orden.
    
    Si el texto ya cabe en chunk_size, lo devuelve tal cual.
    Si no, intenta dividirlo con el primer separador.
    Si algún fragmento sigue siendo muy grande, recursa con el siguiente separador.
    """
    if separators is None:
        separators = SEPARATORS
    
    # Caso base: ya entra en el chunk
    if count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []
    
    # Sin más separadores: dividir a la fuerza por tokens
    if not separators:
        return _force_split_by_tokens(text, chunk_size)
    
    sep = separators[0]
    remaining = separators[1:]
    
    # Dividir con el separador actual
    if sep:
        parts = text.split(sep)
    else:
        parts = list(text)
    
    chunks = []
    current_parts = []
    current_tokens = 0
    
    for part in parts:
        part_tokens = count_tokens(part)
        join_tokens = count_tokens(sep) if current_parts else 0
        
        if current_tokens + join_tokens + part_tokens <= chunk_size:
            current_parts.append(part)
            current_tokens += join_tokens + part_tokens
        else:
            # Guardar chunk actual
            if current_parts:
                chunks.append(sep.join(current_parts))
            
            # Si el part solo es demasiado grande, recursar
            if part_tokens > chunk_size:
                sub_chunks = _split_text(part, chunk_size, remaining)
                if sub_chunks:
                    chunks.extend(sub_chunks[:-1])
                    current_parts = [sub_chunks[-1]] if sub_chunks else []
                    current_tokens = count_tokens(current_parts[0]) if current_parts else 0
                else:
                    current_parts = []
                    current_tokens = 0
            else:
                current_parts = [part]
                current_tokens = part_tokens
    
    # Guardar el último chunk
    if current_parts:
        chunks.append(sep.join(current_parts))
    
    return [c for c in chunks if c.strip()]


def _force_split_by_tokens(text: str, chunk_size: int) -> list[str]:
    """
    División de último recurso: partir por palabras hasta llegar a chunk_size.
    """
    words = text.split()
    chunks = []
    current = []
    current_tokens = 0
    
    for word in words:
        wt = count_tokens(word)
        if current_tokens + wt + (1 if current else 0) <= chunk_size:
            current.append(word)
            current_tokens += wt + (1 if len(current) > 1 else 0)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
            current_tokens = wt
    
    if current:
        chunks.append(" ".join(current))
    
    return chunks


def _apply_overlap(raw_chunks: list[dict], overlap_tokens: int) -> list[dict]:
    """
    Agrega overlap entre chunks consecutivos.
    
    El overlap toma las últimas N palabras del chunk anterior y las
    prepende al chunk actual. Esto garantiza que el contexto no se
    pierda en los bordes entre chunks.
    
    El overlap usa palabras (no tokens exactos) para no cortar palabras al medio.
    """
    if len(raw_chunks) <= 1:
        return raw_chunks
    
    result = [raw_chunks[0]]
    
    # Estimación: 1 token ≈ 0.75 palabras (promedio)
    overlap_words = max(5, int(overlap_tokens * 0.75))
    
    for i in range(1, len(raw_chunks)):
        prev_text = raw_chunks[i - 1]["text"]
        curr = raw_chunks[i]
        
        prev_words = prev_text.split()
        
        if len(prev_words) > overlap_words:
            overlap_text = " ".join(prev_words[-overlap_words:])
            new_text = overlap_text + " " + curr["text"]
        else:
            new_text = prev_text + " " + curr["text"]
        
        result.append({"text": new_text, "page": curr["page"]})
    
    return result