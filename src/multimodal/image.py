"""
src/multimodal/image.py
───────────────────────
Módulo de procesamiento de imágenes para GAIA.
Extrae texto de imágenes usando EasyOCR.

Casos de uso en telecomunicaciones:
  - Foto de factura → extraer número de referencia, valor, fecha
  - Foto de router → identificar modelo, luces de estado
  - Captura de pantalla de error → leer código de error
  - Foto de contrato → extraer datos relevantes
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

_LOG = logging.getLogger("gaia.multimodal.image")

# Idiomas soportados por EasyOCR
OCR_LANGUAGES = ["es", "en"]

# Instancia global del reader (se carga una sola vez)
_ocr_reader = None


def _get_reader():
    """Carga el reader de EasyOCR en memoria (singleton)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _LOG.info("Cargando EasyOCR con idiomas %s...", OCR_LANGUAGES)
        _ocr_reader = easyocr.Reader(
            OCR_LANGUAGES,
            gpu=False,          # CPU — sin GPU
            verbose=False,
        )
        _LOG.info("EasyOCR cargado correctamente.")
    return _ocr_reader


def extract_text_from_image(image_bytes: bytes, filename: str = "image.jpg") -> dict:
    """
    Extrae texto de una imagen usando EasyOCR.

    Args:
        image_bytes: Contenido de la imagen en bytes.
        filename: Nombre original del archivo (para detectar extensión).

    Returns:
        dict con:
          - text: texto extraído completo
          - blocks: lista de bloques de texto con coordenadas y confianza
          - success: True si extrajo texto correctamente
          - error: mensaje de error si success=False
    """
    import time

    ext = Path(filename).suffix.lower() or ".jpg"
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    if ext not in supported:
        return {
            "text": "",
            "blocks": [],
            "success": False,
            "error": f"Formato no soportado: {ext}. Usa: {', '.join(supported)}",
        }

    try:
        reader = _get_reader()

        # Guardar bytes en archivo temporal
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, mode="wb"
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            t0 = time.time()
            _LOG.info("Extrayendo texto de imagen '%s' con EasyOCR...", filename)

            results = reader.readtext(
                tmp_path,
                detail=1,           # Incluir coordenadas y confianza
                paragraph=False,    # No agrupar en párrafos
            )

            elapsed = round(time.time() - t0, 2)

            # Filtrar resultados con confianza > 0.3
            blocks = []
            text_parts = []

            for (bbox, text, confidence) in results:
                if confidence > 0.3 and text.strip():
                    blocks.append({
                        "text": text.strip(),
                        "confidence": round(confidence, 3),
                        "bbox": bbox,
                    })
                    text_parts.append(text.strip())

            full_text = " ".join(text_parts)

            _LOG.info(
                "OCR completado en %ss | bloques=%d | chars=%d",
                elapsed, len(blocks), len(full_text)
            )

            if not full_text:
                return {
                    "text": "",
                    "blocks": [],
                    "success": False,
                    "error": "No se detectó texto en la imagen.",
                }

            return {
                "text": full_text,
                "blocks": blocks,
                "duration": elapsed,
                "success": True,
                "error": None,
            }

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        _LOG.error("Error procesando imagen: %s", str(e))
        return {
            "text": "",
            "blocks": [],
            "success": False,
            "error": f"Error procesando imagen: {str(e)}",
        }


def describe_image_context(text: str, filename: str = "") -> str:
    """
    Genera un prompt enriquecido con el contexto de la imagen
    para enviarlo al modelo conversacional GAIA.

    Args:
        text: Texto extraído por OCR.
        filename: Nombre del archivo original.

    Returns:
        Prompt enriquecido listo para el grafo LangGraph.
    """
    if not text:
        return "El usuario envió una imagen pero no se pudo extraer texto de ella."

    context = (
        f"El usuario envió una imagen"
        f"{f' ({filename})' if filename else ''}.\n\n"
        f"Texto extraído de la imagen mediante OCR:\n"
        f"---\n{text}\n---\n\n"
        f"Por favor analiza este contenido y ayuda al usuario con su consulta "
        f"relacionada con los servicios de telecomunicaciones."
    )
    return context


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        result = extract_text_from_image(
            Path(sys.argv[1]).read_bytes(),
            filename=sys.argv[1]
        )
        print(f"\nTexto extraído: {result['text']}")
        print(f"Bloques: {len(result['blocks'])}")
        print(f"Éxito: {result['success']}")
    else:
        print("Uso: python -m src.multimodal.image <archivo_imagen>")
