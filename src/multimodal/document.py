"""
src/multimodal/document.py
──────────────────────────
Módulo de procesamiento de documentos PDF para GAIA.
Extrae texto de archivos PDF usando PyMuPDF (fitz).

Casos de uso en telecomunicaciones:
  - Factura en PDF → extraer valor, fecha de vencimiento, conceptos
  - Contrato en PDF → identificar cláusulas relevantes
  - Comprobante de pago → verificar datos de la transacción
  - Documentos escaneados → OCR automático como fallback
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
                # PDF escaneado — intentar OCR con EasyOCR
                _LOG.info("PDF sin texto seleccionable, intentando OCR...")
                try:
                    import fitz
                    import easyocr
                    ocr_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
                    doc_ocr = fitz.open(tmp_path)
                    ocr_texts = []
                    for page_num in range(len(doc_ocr)):
                        page = doc_ocr[page_num]
                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        results = ocr_reader.readtext(img_bytes, detail=0)
                        if results:
                            ocr_texts.append(f"[Página {page_num + 1}]\n" + " ".join(results))
                    doc_ocr.close()
                    full_text = "\n\n".join(ocr_texts)
                    if not full_text.strip():
                        return {
                             "text": "",
                             "pages": 0,
                             "success": False,
                             "error": "No se pudo extraer texto del PDF. Intenta enviar una foto del documento.",
                         }
                except Exception as ocr_err:
                    return {
                        "text": "",
                        "pages": 0,
                        "success": False,
                        "error": "El PDF es una imagen escaneada y no se pudo procesar con OCR.",
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
        f"El usuario ha compartido el siguiente documento PDF"
        f"{f' llamado {filename}' if filename else ''}. "
        f"A continuación está el texto completo extraído del documento. "
        f"IMPORTANTE: Debes responder ÚNICAMENTE basándote en el contenido "
        f"del documento que se muestra a continuación. NO digas que no tienes "
        f"acceso al documento. El texto ya fue extraído y está disponible:\n\n"
        f"=== CONTENIDO DEL DOCUMENTO ===\n"
        f"{text_truncado}\n"
        f"=== FIN DEL DOCUMENTO ===\n\n"
        f"Analiza el contenido anterior y responde la consulta del usuario "
        f"sobre sus servicios de telecomunicaciones."
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
