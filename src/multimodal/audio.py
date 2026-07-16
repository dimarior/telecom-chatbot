"""
src/multimodal/audio.py
───────────────────────
Módulo de procesamiento de audio para GAIA.
Convierte archivos de voz a texto usando Whisper (OpenAI) en CPU.

Modelos disponibles (en orden de tamaño y precisión):
  - tiny   → más rápido, menos preciso (~1s en CPU)
  - base   → balance ideal para demos (~3s en CPU)
  - small  → buena precisión (~8s en CPU)
  - medium → alta precisión (~20s en CPU)

Para tesis académica se recomienda 'base' en CPU sin GPU.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

_LOG = logging.getLogger("gaia.multimodal.audio")

# Modelo Whisper — configurable por variable de entorno
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Instancia global del modelo (se carga una sola vez)
_whisper_model = None


def _get_model():
    """Carga el modelo Whisper en memoria (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _LOG.info("Cargando modelo Whisper '%s' en CPU...", WHISPER_MODEL)
        _whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
        _LOG.info("Modelo Whisper cargado correctamente.")
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    Transcribe un archivo de audio a texto usando Whisper.

    Args:
        audio_bytes: Contenido del archivo de audio en bytes.
        filename: Nombre original del archivo (para detectar extensión).

    Returns:
        dict con:
          - text: texto transcrito
          - language: idioma detectado
          - duration: duración estimada en segundos
          - success: True si transcribió correctamente
          - error: mensaje de error si success=False
    """
    import time

    # Extensión del archivo
    ext = Path(filename).suffix.lower() or ".wav"
    supported = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".mp4", ".webm"}

    if ext not in supported:
        return {
            "text": "",
            "language": "",
            "duration": 0,
            "success": False,
            "error": f"Formato no soportado: {ext}. Usa: {', '.join(supported)}",
        }

    try:
        model = _get_model()

        # Guardar bytes en archivo temporal
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, mode="wb"
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            t0 = time.time()
            _LOG.info("Transcribiendo audio '%s' con Whisper...", filename)

            result = model.transcribe(
                tmp_path,
                language="es",        # Español colombiano
                task="transcribe",
                fp16=False,           # CPU no soporta fp16
                verbose=False,
            )

            elapsed = round(time.time() - t0, 2)
            text = result.get("text", "").strip()
            language = result.get("language", "es")

            _LOG.info(
                "Transcripción completada en %ss | idioma=%s | chars=%d",
                elapsed, language, len(text)
            )

            return {
                "text": text,
                "language": language,
                "duration": elapsed,
                "success": True,
                "error": None,
            }

        finally:
            # Limpiar archivo temporal
            os.unlink(tmp_path)

    except Exception as e:
        _LOG.error("Error transcribiendo audio: %s", str(e))
        return {
            "text": "",
            "language": "",
            "duration": 0,
            "success": False,
            "error": f"Error procesando audio: {str(e)}",
        }


def transcribe_audio_file(file_path: str | Path) -> dict:
    """
    Transcribe un archivo de audio desde disco.

    Args:
        file_path: Ruta al archivo de audio.

    Returns:
        Mismo dict que transcribe_audio().
    """
    path = Path(file_path)
    if not path.exists():
        return {
            "text": "",
            "language": "",
            "duration": 0,
            "success": False,
            "error": f"Archivo no encontrado: {file_path}",
        }

    audio_bytes = path.read_bytes()
    return transcribe_audio(audio_bytes, filename=path.name)


if __name__ == "__main__":
    # Test rápido
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        result = transcribe_audio_file(sys.argv[1])
        print(f"\nTexto transcrito: {result['text']}")
        print(f"Idioma: {result['language']}")
        print(f"Tiempo: {result['duration']}s")
    else:
        print("Uso: python -m src.multimodal.audio <archivo_audio>")
