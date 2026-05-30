"""
Extracción de texto de PDFs usando PyMuPDF (fitz).

PyMuPDF es superior a las librerías de Go para PDFs porque:
- Maneja PDFs complejos con tablas, columnas y layouts mixtos
- Preserva el orden de lectura correcto
- Limpia artefactos de extracción (headers repetidos, números de página, etc)
- Soporte para PDFs con capas, formularios y anotaciones
"""

from dataclasses import dataclass
import fitz  # PyMuPDF
import re


@dataclass
class PageContent:
    text: str
    page_num: int  # 1-indexed


def extract_text(file_path: str) -> list[PageContent]:
    """
    Extrae el texto de cada página de un PDF.
    
    Aplica limpieza básica:
    - Elimina líneas con solo números (números de página)
    - Colapsa espacios múltiples
    - Elimina páginas vacías
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # "text" mode preserva el orden de lectura natural
        # Para PDFs con columnas, probar "blocks" o "dict" si hay problemas
        text = page.get_text("text")
        
        text = _clean_text(text)
        
        if not text:
            continue
            
        pages.append(PageContent(
            text=text,
            page_num=page_num + 1,
        ))

    doc.close()
    
    print(f"[extractor] {len(pages)} páginas con texto extraídas de {file_path}")
    return pages


def _clean_text(text: str) -> str:
    """
    Limpia el texto extraído del PDF.
    
    - Elimina líneas que son solo números (números de página)
    - Elimina líneas vacías múltiples consecutivas
    - Normaliza espacios
    """
    lines = text.split("\n")
    cleaned = []
    
    for line in lines:
        stripped = line.strip()
        
        # Saltar líneas vacías
        if not stripped:
            # Agregar salto de línea si la última línea no era vacía
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        
        # Saltar líneas que son solo números (números de página)
        if re.match(r"^\d+$", stripped):
            continue
        
        cleaned.append(stripped)
    
    # Eliminar saltos de línea extra al principio y al final
    result = "\n".join(cleaned).strip()
    
    # Colapsar más de 2 saltos de línea consecutivos
    result = re.sub(r"\n{3,}", "\n\n", result)
    
    return result