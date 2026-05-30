"""
Section-Aware Chunker — Para documentos con estructura numerada.

Detecta secciones por sus headings ("3. REPARTO DE PAÍSES Y OBJETIVOS")
y las usa como unidades primarias de chunking. Esto garantiza que:

1. El título de sección SIEMPRE está en el mismo chunk que su contenido.
2. El INDEX/tabla de contenidos se filtra y no contamina el retrieval.
3. Las secciones largas se dividen en sub-chunks que siempre incluyen el título.

Para el TEG, Risk, UNO: funciona perfecto.
Para el Poole (álgebra): usar hierarchical chunker (Fase 2).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from extractors.pdf import PageContent


# Heading numerado: "3. REPARTO DE PAÍSES Y OBJETIVOS"
# También soporta sub-headings: "3.1 ALGO"
SECTION_PATTERNS = [
    re.compile(r'^\d+\.\s+[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-\(\)]+$', re.MULTILINE),
    re.compile(r'^\*\s+[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-\(\)]+$', re.MULTILINE),  # * PARTIDAS DE 2 O 3 JUGADORES
]

# Detectar líneas de índice: "3. REPARTO DE PAÍSES . . . . 3"
INDEX_LINE = re.compile(r'\.{2,}\s*\d+\s*$')


@dataclass
class Section:
    number: str          # "3" o "*"
    title: str           # "REPARTO DE PAÍSES Y OBJETIVOS"
    content: str         # texto del contenido
    page: int
    char_start: int      # posición en el texto completo


@dataclass
class RawChunk:
    text: str
    page: int
    section_title: Optional[str] = None
    section_number: Optional[str] = None


@dataclass
class ChunkMetadata:
    doc_id: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int
    section_title: Optional[str] = None
    section_number: Optional[str] = None


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata


# ─── Función de conteo ────────────────────────────────────────────────────────

def char_count(text: str) -> int:
    """
    Usa caracteres en lugar de tokens.
    Es más simple, más rápido, y suficientemente preciso.
    Regla práctica: 1500 chars ≈ 300-350 tokens para español.
    """
    return len(text)


# ─── Detección de índice ──────────────────────────────────────────────────────

def is_index_block(text: str) -> bool:
    """
    Detecta tablas de contenidos. Un bloque es índice si más del
    35% de sus líneas tienen el patrón "texto . . . número".
    
    Ejemplo de línea de índice:
    "3. REPARTO DE PAÍSES Y OBJETIVOS . . . . . . . . . 3"
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 3:
        return False

    index_lines = sum(1 for l in lines if INDEX_LINE.search(l))
    return (index_lines / len(lines)) > 0.35


# ─── Chunker principal ────────────────────────────────────────────────────────

def section_aware_chunk(
    pages: list[PageContent],
    doc_id: str,
    source: str,
    max_chars: int = 2000,
    overlap_chars: int = 300,
) -> list[Chunk]:
    """
    Chunking que respeta la estructura de secciones del documento.

    Args:
        max_chars: Tamaño máximo de chunk en caracteres.
                   2000 chars ≈ 400-500 tokens, ideal para retrieval.
        overlap_chars: Overlap al dividir secciones largas.
    """
    # 1. Concatenar todo el texto del documento con marcas de página
    page_texts = []
    page_offsets = []  # (char_offset, page_num)
    offset = 0

    for page in pages:
        page_offsets.append((offset, page.page_num))
        page_texts.append(page.text)
        offset += len(page.text) + 1

    full_text = '\n'.join(page_texts)

    # 2. Detectar secciones en el texto completo
    sections = _detect_sections(full_text, page_offsets)

    if not sections:
        # Fallback: chunking simple por párrafos si no hay secciones detectadas
        print("[section_aware] No se detectaron secciones. Usando fallback por párrafos.")
        return _paragraph_chunk(pages, doc_id, source, max_chars, overlap_chars)

    print(f"[section_aware] {len(sections)} secciones detectadas:")
    for s in sections:
        print(f"  [{s.number}] {s.title[:50]} — pág.{s.page}")

    # 3. Generar raw chunks por sección
    raw_chunks: list[RawChunk] = []

    for section in sections:
        # Saltar si es un bloque de índice
        if is_index_block(section.content):
            print(f"  ⏭ Saltando índice: {section.title[:40]}")
            continue

        # Texto completo de la sección (título + contenido)
        full_section = f"{section.number}. {section.title}\n\n{section.content}".strip()
        
        # Si la sección entra en un chunk, usarla completa
        if char_count(full_section) <= max_chars:
            raw_chunks.append(RawChunk(
                text=full_section,
                page=section.page,
                section_title=section.title,
                section_number=section.number,
            ))
        else:
            # Sección grande: dividir manteniendo el título en cada sub-chunk
            sub = _split_long_section(section, max_chars, overlap_chars)
            raw_chunks.extend(sub)

    if not raw_chunks:
        return _paragraph_chunk(pages, doc_id, source, max_chars, overlap_chars)

    # 4. Construir objetos Chunk con metadata completa
    total = len(raw_chunks)
    result = []
    for i, rc in enumerate(raw_chunks):
        result.append(Chunk(
            text=rc.text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                source=source,
                page=rc.page,
                chunk_index=i,
                total_chunks=total,
                section_title=rc.section_title,
                section_number=rc.section_number,
            )
        ))

    return result


# ─── Detección de secciones ───────────────────────────────────────────────────

def _detect_sections(full_text: str, page_offsets: list[tuple]) -> list[Section]:
    """
    Encuentra todas las secciones numeradas en el texto.
    
    Intenta múltiples patrones y usa el que detecta más secciones.
    """
    best_matches = []
    
    for pattern in SECTION_PATTERNS:
        matches = list(pattern.finditer(full_text))
        if len(matches) > len(best_matches):
            best_matches = matches

    if not best_matches:
        return []

    sections = []
    for i, match in enumerate(best_matches):
        # Extraer número y título
        matched_line = match.group(0).strip()
        
        if matched_line.startswith('*'):
            number = "*"
            title = matched_line[1:].strip()
        else:
            dot_pos = matched_line.index('.')
            number = matched_line[:dot_pos].strip()
            title = matched_line[dot_pos + 1:].strip()

        # Contenido: desde el fin del heading hasta el inicio del siguiente
        content_start = match.end()
        content_end = best_matches[i + 1].start() if i + 1 < len(best_matches) else len(full_text)
        content = full_text[content_start:content_end].strip()

        # Determinar página
        page = _char_to_page(match.start(), page_offsets)

        sections.append(Section(
            number=number,
            title=title,
            content=content,
            page=page,
            char_start=match.start(),
        ))

    return sections


def _char_to_page(char_offset: int, page_offsets: list[tuple]) -> int:
    """Convierte un offset de caracteres a número de página."""
    page = 1
    for offset, page_num in page_offsets:
        if offset <= char_offset:
            page = page_num
        else:
            break
    return page


# ─── División de secciones largas ─────────────────────────────────────────────

def _split_long_section(
    section: Section,
    max_chars: int,
    overlap_chars: int,
) -> list[RawChunk]:
    """
    Divide una sección larga en sub-chunks.
    
    Regla: cada sub-chunk incluye el título de la sección al principio,
    para que cualquier chunk devuelto sepa a qué sección pertenece.
    """
    header = f"{section.number}. {section.title}\n\n"
    
    # Intentar dividir por párrafos primero
    paragraphs = [p.strip() for p in section.content.split('\n\n') if p.strip()]
    
    chunks = []
    current_text = header
    
    for para in paragraphs:
        candidate = current_text + para + '\n\n'
        
        if char_count(candidate) <= max_chars:
            current_text = candidate
        else:
            # Guardar chunk actual si tiene contenido más allá del header
            if current_text.strip() != header.strip():
                chunks.append(RawChunk(
                    text=current_text.strip(),
                    page=section.page,
                    section_title=section.title,
                    section_number=section.number,
                ))
                # El siguiente chunk empieza con el header + overlap del anterior
                overlap = _get_overlap(current_text, overlap_chars)
                current_text = header + overlap + '\n\n' + para + '\n\n'
            else:
                # El párrafo solo ya excede el tamaño → agregarlo de todas formas
                # (preferible a perder contenido)
                current_text += para + '\n\n'
    
    # Guardar el último chunk
    if current_text.strip() and current_text.strip() != header.strip():
        chunks.append(RawChunk(
            text=current_text.strip(),
            page=section.page,
            section_title=section.title,
            section_number=section.number,
        ))

    return chunks


def _get_overlap(text: str, overlap_chars: int) -> str:
    """
    Extrae los últimos N caracteres del texto para overlap.
    Corta en un límite de palabra/oración.
    """
    if len(text) <= overlap_chars:
        return text.strip()
    
    overlap = text[-overlap_chars:]
    
    # Cortar en el primer espacio para no partir palabras
    first_space = overlap.find(' ')
    if first_space > 0:
        overlap = overlap[first_space:].strip()
    
    return overlap


# ─── Fallback chunking simple ─────────────────────────────────────────────────

def _paragraph_chunk(
    pages: list[PageContent],
    doc_id: str,
    source: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """
    Chunking simple por párrafos, usado como fallback.
    Filtra bloques que son índices.
    """
    raw_chunks = []

    for page in pages:
        # Filtrar páginas que son índices
        if is_index_block(page.text):
            continue

        paragraphs = [p.strip() for p in page.text.split('\n\n') if p.strip()]
        current = ""

        for para in paragraphs:
            candidate = current + para + '\n\n'
            if char_count(candidate) <= max_chars:
                current = candidate
            else:
                if current.strip():
                    raw_chunks.append(RawChunk(text=current.strip(), page=page.page_num))
                current = para + '\n\n'

        if current.strip():
            raw_chunks.append(RawChunk(text=current.strip(), page=page.page_num))

    total = len(raw_chunks)
    return [
        Chunk(
            text=rc.text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                source=source,
                page=rc.page,
                chunk_index=i,
                total_chunks=total,
            )
        )
        for i, rc in enumerate(raw_chunks)
    ]