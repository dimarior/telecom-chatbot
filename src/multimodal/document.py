"""
src/multimodal/document.py
──────────────────────────
Módulo de procesamiento de documentos PDF para GAIA.
Extrae texto de archivos PDF usando PyMuPDF (fitz).
Si el PDF es escaneado (sin texto seleccionable), aplica EasyOCR como fallback.
"""
from __future__ import annotations

import logging
import tempfile
import os
from pathlib import Path

_LOG = logging.getLogger("gaia.multimodal.document")


def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> dict:
    """
    Extrae texto de un archivo PDF usando PyMuPDF.
    Si el PDF no tiene texto seleccionable (escaneado), aplica EasyOCR como fallback.
    """
    import time

    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        return {
            "text": "", "pages": 0, "success": False,
            "error": f"Formato no soportado: {ext}. Solo se aceptan archivos PDF.",
            "ocr_used": False,
        }

    try:
        import fitz

        t0 = time.time()
        _LOG.info("Extrayendo texto de PDF '%s'...", filename)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            doc = fitz.open(tmp_path)
            pages_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    pages_text.append(f"[Página {page_num + 1}]\n{text.strip()}")

            doc.close()
            full_text = "\n\n".join(pages_text)

            # PDF con texto seleccionable — extracción directa exitosa
            if full_text.strip():
                elapsed = round(time.time() - t0, 2)
                _LOG.info(
                    "PDF procesado con texto directo en %ss | páginas=%d | chars=%d",
                    elapsed, len(pages_text), len(full_text)
                )
                return {
                    "text": full_text,
                    "pages": len(pages_text),
                    "duration": elapsed,
                    "success": True,
                    "error": None,
                    "ocr_used": False,
                }

            # PDF sin texto seleccionable — aplicar OCR como fallback
            _LOG.info("PDF sin texto seleccionable, aplicando OCR como fallback...")
            full_text, pages_text = _ocr_fallback(tmp_path)

            if not full_text.strip():
                return {
                    "text": "", "pages": 0, "success": False,
                    "error": (
                        "No se pudo extraer texto del PDF. "
                        "Intenta enviar una foto del documento usando el botón Imagen."
                    ),
                    "ocr_used": True,
                }

            elapsed = round(time.time() - t0, 2)
            _LOG.info(
                "PDF procesado con OCR en %ss | páginas=%d | chars=%d",
                elapsed, len(pages_text), len(full_text)
            )
            return {
                "text": full_text,
                "pages": len(pages_text),
                "duration": elapsed,
                "success": True,
                "error": None,
                "ocr_used": True,
            }

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        _LOG.error("Error procesando PDF: %s", str(e))
        return {
            "text": "", "pages": 0, "success": False,
            "error": f"Error procesando el documento: {str(e)}",
            "ocr_used": False,
        }


def _ocr_fallback(pdf_path: str) -> tuple[str, list[str]]:
    """Aplica EasyOCR a cada página del PDF escaneado."""
    try:
        import fitz
        import easyocr

        _LOG.info("Cargando EasyOCR para PDF escaneado...")
        ocr_reader = easyocr.Reader(
            ["es", "en"],
            gpu=False,
            verbose=False,
            download_enabled=True,
            model_storage_directory="/tmp/easyocr_models",
        )

        doc = fitz.open(pdf_path)
        pages_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            results = ocr_reader.readtext(img_bytes, detail=0)
            if results:
                page_text = " ".join(results).strip()
                if page_text:
                    pages_text.append(f"[Página {page_num + 1}]\n{page_text}")

        doc.close()
        full_text = "\n\n".join(pages_text)
        _LOG.info("OCR fallback completado | páginas=%d | chars=%d",
                  len(pages_text), len(full_text))
        return full_text, pages_text

    except Exception as e:
        _LOG.error("Error en OCR fallback: %s", str(e))
        return "", []


def describe_document_context(
    text: str,
    filename: str = "",
    ocr_used: bool = False,
) -> str:
    """
    Genera un prompt enriquecido con el contexto del documento.
    Adapta el prompt según si el texto viene de extracción directa o de OCR.
    """
    if not text:
        return "El usuario envió un documento PDF pero no se pudo extraer texto."

    text_truncado = text[:3000] + "..." if len(text) > 3000 else text

    if ocr_used:
        nota = (
            "El texto fue extraído mediante OCR desde un documento escaneado "
            "y puede aparecer desordenado debido al layout original. "
            "Analízalo aunque esté desordenado e identifica la información relevante. "
        )
    else:
        nota = (
            "El texto fue extraído directamente del documento PDF con alta fidelidad. "
        )

    context = (
        f"El usuario ha compartido un documento PDF"
        f"{f' llamado {filename}' if filename else ''}. "
        f"{nota}"
        f"IMPORTANTE: El contenido está disponible a continuación. "
        f"NO digas que no tienes acceso al documento. "
        f"Analiza el contenido y responde de forma clara y útil:\n\n"
        f"=== CONTENIDO DEL DOCUMENTO ===\n"
        f"{text_truncado}\n"
        f"=== FIN DEL DOCUMENTO ===\n\n"
        f"Analiza el contenido anterior y ayuda al usuario con su consulta "
        f"relacionada con los servicios de telecomunicaciones."
    )
    return context


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        result = extract_text_from_pdf(
            Path(sys.argv[1]).read_bytes(),
            filename=sys.argv[1]
        )
        print(f"\nTexto extraído: {result['text'][:500]}...")
        print(f"Páginas: {result['pages']}")
        print(f"OCR usado: {result['ocr_used']}")
        print(f"Éxito: {result['success']}")
    else:
        print("Uso: python -m src.multimodal.document <archivo.pdf>")