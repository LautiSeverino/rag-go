"""
Section-Aware Chunker — Reescritura completa.

Problemas del v1 que corrige:
1. Ya no elige el patrón con más matches (estrategia incorrecta)
2. Detecta AMBOS tipos de heading (numerados Y asterisco) simultáneamente
3. Filtra correctamente entre headings reales vs ítems de lista
4. No descarta contenido antes del primer heading detectado
5. Maneja el índice inline del PDF
6. Mergea chunks vacíos con su vecino

Lógica de distinción heading vs ítem de lista:
  HEADING real:   "N. TÍTULO EN MAYÚSCULAS"  (>75% letras mayúsculas, sin puntos de índice)
  HEADING real:   "* TÍTULO EN MAYÚSCULAS"   (>75% letras mayúsculas, sin punto al final)
  Ítem de lista:  "1. Cada participante..."   (minúscula después del número)
  Acción de lista: "* ATACAR países ajenos."  (minúscula mezclada, punto al final)
  Línea de índice: "3. REPARTO . . . . . 3"   (puntos consecutivos)
"""

import re
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF


# ─── Modelos de datos ─────────────────────────────────────────────────────────

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


# ─── Patrones de detección ────────────────────────────────────────────────────

# Headings numerados: "N. TÍTULO" donde TÍTULO es dominantemente mayúscula
# El patrón matchea cualquier carácter en el título (incluyendo dígitos como en
# "4. DOS VUELTAS PARA AGREGAR 8 EJÉRCITOS")
_NUMBERED = re.compile(r'^(\d+)\.\s+(.+)$', re.MULTILINE)

# Headings con asterisco: "* TÍTULO" donde TÍTULO es dominantemente mayúscula
_ASTERISK = re.compile(r'^\*\s+(.+)$', re.MULTILINE)

# Líneas del índice: texto seguido de muchos puntos y un número
_INDEX_LINE = re.compile(r'(\.\s*){3,}\d+\s*$')

# Contenido de publicidad/pie de documento a filtrar
_SKIP_PATTERNS = [
    re.compile(r'yetem\.com', re.IGNORECASE),
    re.compile(r'App Store|Google Play', re.IGNORECASE),
    re.compile(r'TEG Móvil|TEG JUNIOR', re.IGNORECASE),
    re.compile(r'Si te divertiste', re.IGNORECASE),
    re.compile(r'NEW YETEM', re.IGNORECASE),
]


# ─── Funciones auxiliares ─────────────────────────────────────────────────────

def _uppercase_ratio(text: str) -> float:
    """Proporción de letras mayúsculas en el texto."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _is_real_heading(title: str) -> bool:
    """
    Determina si un título es un heading real o un ítem de lista.

    Criterios:
    - Sin líneas de índice (puntos consecutivos)
    - Título corto (< 70 chars para numbered, cualquier longitud para asterisk)
    - Más del 75% de letras son mayúsculas
    - No termina en punto (eso indica ítem de acción)
    """
    if _INDEX_LINE.search(title):
        return False
    if title.rstrip().endswith('.') and _uppercase_ratio(title) < 0.9:
        # "* ATACAR países ajenos." → False
        # "* OBJETIVO COMÚN." → True (pero este no tiene punto en el doc real)
        return False
    return _uppercase_ratio(title) > 0.75


def _clean_text(text: str) -> str:
    """Limpia el texto extraído: quita números de página aislados y líneas de índice."""
    lines = []
    for line in text.split('\n'):
        s = line.strip()
        # Número de página solo (ej: "2", "11")
        if re.match(r'^\d+$', s):
            continue
        # Línea del índice (ej: "3. REPARTO . . . . . 3")
        if _INDEX_LINE.search(s):
            continue
        # Líneas de publicidad/pie
        if any(p.search(s) for p in _SKIP_PATTERNS):
            continue
        lines.append(line)
    return '\n'.join(lines).strip()


def _detect_headings(full_text: str) -> list[tuple[int, int, str, str]]:
    """
    Detecta todos los headings reales en el texto completo.

    Returns: list of (start_pos, end_pos, number, title)
      number: "1"-"9" para numerados, "*" para asterisco
    """
    headings = []

    # Headings numerados
    for m in _NUMBERED.finditer(full_text):
        num = m.group(1)
        title = m.group(2).strip()
        if _is_real_heading(title) and len(title) < 65:
            headings.append((m.start(), m.end(), num, title))

    # Headings con asterisco
    for m in _ASTERISK.finditer(full_text):
        title = m.group(1).strip()
        if _is_real_heading(title):
            headings.append((m.start(), m.end(), "*", title))

    headings.sort(key=lambda x: x[0])
    return headings


def _find_page(char_offset: int, page_offsets: list[tuple[int, int]]) -> int:
    """Convierte un offset de caracteres al número de página."""
    page = 1
    for offset, page_num in page_offsets:
        if offset <= char_offset:
            page = page_num
        else:
            break
    return page


# ─── Extracción del PDF ───────────────────────────────────────────────────────

def extract_full_text(file_path: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Extrae todo el texto del PDF como un string continuo.

    Returns:
        (full_text, page_offsets)
        page_offsets: list of (char_offset, page_number)
    """
    doc = fitz.open(file_path)
    parts = []
    page_offsets = []
    offset = 0

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            page_offsets.append((offset, page_num + 1))
            parts.append(text)
            offset += len(text)

    doc.close()
    return ''.join(parts), page_offsets


# ─── Chunker principal ────────────────────────────────────────────────────────

def section_aware_chunk_v2(
    file_path: str,
    doc_id: str,
    max_chars: int = 3000,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """
    Chunking basado en la estructura real del documento.

    Estrategia:
    1. Extraer el texto completo del PDF (no página por página)
    2. Detectar todos los headings reales (numerados + asterisco)
    3. Cada heading → un chunk con su contenido
    4. Mergear chunks muy pequeños con su vecino
    5. Dividir chunks muy grandes en sub-chunks por párrafos

    Args:
        file_path: Ruta al PDF
        doc_id: Identificador del documento
        max_chars: Tamaño máximo de chunk en caracteres (~600 chars = ~150 tokens)
        overlap_chars: Overlap al dividir secciones largas
    """
    # 1. Extraer texto completo
    full_text, page_offsets = extract_full_text(file_path)

    if not full_text.strip():
        return []

    # 2. Detectar headings
    headings = _detect_headings(full_text)

    if not headings:
        print("[section_aware_v2] No se detectaron headings. Usando fallback por párrafos.")
        return _paragraph_fallback(full_text, page_offsets, doc_id, file_path, max_chars)

    print(f"[section_aware_v2] {len(headings)} headings detectados:")
    for _, _, num, title in headings:
        label = f"{num}. {title}" if num != "*" else f"* {title}"
        print(f"  {label}")

    # 3. Generar raw chunks
    raw_chunks = []

    # Contenido ANTES del primer heading (si hay)
    pre_content = _clean_text(full_text[:headings[0][0]])
    if pre_content and len(pre_content) > 50:
        page = _find_page(0, page_offsets)
        raw_chunks.append({
            "text": pre_content,
            "page": page,
            "title": None,
            "number": None,
        })

    # Un chunk por heading
    for i, (start, end, num, title) in enumerate(headings):
        content_end = headings[i + 1][0] if i + 1 < len(headings) else len(full_text)
        raw_content = _clean_text(full_text[end:content_end])

        heading_label = f"{num}. {title}" if num != "*" else f"* {title}"
        chunk_text = f"{heading_label}\n\n{raw_content}" if raw_content else heading_label

        page = _find_page(start, page_offsets)

        raw_chunks.append({
            "text": chunk_text,
            "page": page,
            "title": title,
            "number": num,
            "content_len": len(raw_content),
        })

    # 4. Mergear chunks muy pequeños (< 60 chars de contenido)
    merged = _merge_small_chunks(raw_chunks, min_content=60)

    # 5. Dividir chunks muy grandes
    final_raw = []
    for chunk in merged:
        if len(chunk["text"]) > max_chars:
            sub = _split_large_chunk(chunk, max_chars, overlap_chars)
            final_raw.extend(sub)
        else:
            final_raw.append(chunk)

    # 6. Construir objetos Chunk con metadata
    total = len(final_raw)
    result = []
    for i, rc in enumerate(final_raw):
        result.append(Chunk(
            text=rc["text"],
            metadata=ChunkMetadata(
                doc_id=doc_id,
                source=file_path,
                page=rc["page"],
                chunk_index=i,
                total_chunks=total,
                section_title=rc.get("title"),
                section_number=rc.get("number"),
            )
        ))

    print(f"[section_aware_v2] ✓ {len(result)} chunks finales")
    return result


# ─── Merge y split ────────────────────────────────────────────────────────────

def _merge_small_chunks(
    chunks: list[dict],
    min_content: int = 60,
) -> list[dict]:
    """
    Mergea chunks con poco contenido con su vecino siguiente.
    Un chunk "pequeño" es aquel cuyo contenido (sin el título) < min_content chars.
    """
    result = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        content_len = chunk.get("content_len", len(chunk["text"]))

        if content_len < min_content and i + 1 < len(chunks):
            # Mergear con el siguiente
            next_chunk = chunks[i + 1]
            merged_text = chunk["text"] + "\n\n" + next_chunk["text"]
            result.append({
                "text": merged_text,
                "page": chunk["page"],
                "title": chunk.get("title"),
                "number": chunk.get("number"),
                "content_len": len(merged_text),
            })
            i += 2  # Saltar el siguiente (ya fue mergeado)
        else:
            result.append(chunk)
            i += 1

    return result


def _split_large_chunk(chunk: dict, max_chars: int, overlap_chars: int) -> list[dict]:
    """
    Divide un chunk grande en sub-chunks respetando párrafos.
    Cada sub-chunk incluye el título de la sección al principio.
    """
    text = chunk["text"]
    title = chunk.get("title")
    number = chunk.get("number")
    page = chunk["page"]

    # Encontrar el header del chunk
    header_end = text.find('\n\n')
    if header_end == -1:
        return [chunk]

    header = text[:header_end]
    content = text[header_end + 2:]

    # Dividir por párrafos
    paragraphs = content.split('\n\n')

    sub_chunks = []
    current = header + "\n\n"

    for para in paragraphs:
        candidate = current + para + '\n\n'
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current.strip() != header.strip():
                sub_chunks.append({
                    "text": current.strip(),
                    "page": page,
                    "title": title,
                    "number": number,
                    "content_len": len(current) - len(header),
                })
            # Nuevo sub-chunk con overlap
            if len(current) > overlap_chars:
                overlap = current[-overlap_chars:]
                space = overlap.find(' ')
                if space > 0:
                    overlap = overlap[space:].strip()
                current = header + "\n\n..." + overlap + "\n\n" + para + '\n\n'
            else:
                current = header + "\n\n" + para + '\n\n'

    if current.strip() and current.strip() != header.strip():
        sub_chunks.append({
            "text": current.strip(),
            "page": page,
            "title": title,
            "number": number,
            "content_len": len(current) - len(header),
        })

    return sub_chunks if sub_chunks else [chunk]


# ─── Fallback ─────────────────────────────────────────────────────────────────

def _paragraph_fallback(
    full_text: str,
    page_offsets: list[tuple[int, int]],
    doc_id: str,
    source: str,
    max_chars: int,
) -> list[Chunk]:
    """Chunking simple por párrafos, usado solo si no hay headings."""
    paragraphs = full_text.split('\n\n')
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = current + para + '\n\n'
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para + '\n\n'

    if current.strip():
        chunks.append(current.strip())

    total = len(chunks)
    return [
        Chunk(
            text=text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                source=source,
                page=1,
                chunk_index=i,
                total_chunks=total,
            )
        )
        for i, text in enumerate(chunks)
    ]


# ─── Interfaz para el servicio HTTP ──────────────────────────────────────────

def chunk_from_file(
    file_path: str,
    doc_id: str,
    max_chars: int = 3000,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Punto de entrada principal para el servicio HTTP."""
    return section_aware_chunk_v2(file_path, doc_id, max_chars, overlap_chars)
