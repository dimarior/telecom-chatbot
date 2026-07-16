"""
src/multimodal/
───────────────
Módulos de procesamiento multimodal para GAIA.

  audio.py     → Transcripción de voz con Whisper (OpenAI)
  image.py     → Extracción de texto de imágenes con EasyOCR
  document.py  → Extracción de texto de PDFs con PyMuPDF
"""
from src.multimodal.audio import transcribe_audio, transcribe_audio_file
from src.multimodal.image import extract_text_from_image, describe_image_context
from src.multimodal.document import extract_text_from_pdf, describe_document_context

__all__ = [
    "transcribe_audio",
    "transcribe_audio_file",
    "extract_text_from_image",
    "describe_image_context",
    "extract_text_from_pdf",
    "describe_document_context",
]
