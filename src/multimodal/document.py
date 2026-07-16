"""
src/multimodal/document.py
──────────────────────────
Módulo de procesamiento de documentos PDF para GAIA.
Extrae texto de archivos PDF usando PyMuPDF (fitz).

Casos de uso en telecomunicaciones:
  - Factura en PDF → extraer valor, fecha de vencimiento, conceptos
  - Contrato en PDF → identificar cláusulas relevantes
  - Comprobante de pago → verificar datos de la transacción
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

    Args:
        pdf_bytes: Contenido del PDF en bytes.
        filename: Nombre original del archivo.

    Returns:
        dict con:
          - text: texto extraído completo
          - pages: número de páginas procesadas
          - success: True si extrajo texto correctamente
          - error: mensaje de error si success=False
    """
    import time

    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        return {
            "text": "",
            "pages": 0,
            "success": False,
            "error": f"Formato no soportado: {ext}. Solo se aceptan archivos PDF.",
        }

    try:
        import fitz  # PyMuPDF

        t0 = time.time()
        _LOG.info("Extrayendo texto de PDF '%s'...", filename)

        # Guardar en archivo temporal
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
            elapsed = round(time.time() - t0, 2)
            full_text = "\n\n".join(pages_text)

            if not full_text.strip():
                return {
                    "text": "",
                    "pages": len(pages_text),
                    "success": False,
                    "error": (
                        "El PDF no contiene texto seleccionable. "
                        "Puede ser un PDF escaneado. Intenta enviar una foto del documento."
                    ),
                }

            _LOG.info(
                "PDF procesado en %ss | páginas=%d | chars=%d",
                elapsed, len(pages_text), len(full_text)
            )

            return {
                "text": full_text,
                "pages": len(pages_text),
                "duration": elapsed,
                "success": True,
                "error": None,
            }

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        _LOG.error("Error procesando PDF: %s", str(e))
        return {
            "text": "",
            "pages": 0,
            "success": False,
            "error": f"Error procesando el documento: {str(e)}",
        }


def describe_document_context(text: str, filename: str = "") -> str:
    """
    Genera un prompt enriquecido con el contexto del documento
    para enviarlo al modelo conversacional GAIA.
    """
    if not text:
        return "El usuario envió un documento PDF pero no se pudo extraer texto."

    # Limitar a 3000 chars para no saturar el contexto del LLM
    text_truncado = text[:3000] + "..." if len(text) > 3000 else text

    context = (
        f"El usuario envió un documento PDF"
        f"{f' ({filename})' if filename else ''}.\n\n"
        f"Contenido extraído del documento:\n"
        f"---\n{text_truncado}\n---\n\n"
        f"Por favor analiza este documento y ayuda al usuario con su consulta "
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
        print(f"Éxito: {result['success']}")
    else:
        print("Uso: python -m src.multimodal.document <archivo.pdf>")
