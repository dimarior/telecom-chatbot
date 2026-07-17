"""
src/multimodal/audio.py
───────────────────────
Módulo de procesamiento de audio para GAIA.
Convierte archivos de voz a texto usando faster-whisper en CPU.
No requiere ffmpeg instalado.

Modelos disponibles:
  - tiny   → más rápido (~1s en CPU)
  - base   → balance ideal para demos (~3s en CPU)
  - small  → buena precisión (~8s en CPU)
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

_LOG = logging.getLogger("gaia.multimodal.audio")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

_whisper_model = None


def _get_model():
    """Carga el modelo faster-whisper en memoria (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _LOG.info("Cargando modelo faster-whisper '%s' en CPU...", WHISPER_MODEL)
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )
        _LOG.info("Modelo faster-whisper cargado correctamente.")
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    Transcribe un archivo de audio a texto usando faster-whisper.

    Args:
        audio_bytes: Contenido del archivo de audio en bytes.
        filename: Nombre original del archivo.

    Returns:
        dict con text, language, duration, success, error.
    """
    import time

    ext = Path(filename).suffix.lower() or ".wav"
    supported = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".mp4", ".webm"}

    if ext not in supported:
        return {
            "text": "",
            "language": "",
            "duration": 0,
            "success": False,
            "error": f"Formato no soportado: {ext}.",
        }

    try:
        model = _get_model()

        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, mode="wb"
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            t0 = time.time()
            _LOG.info("Transcribiendo audio '%s' con faster-whisper...", filename)

            segments, info = model.transcribe(
                tmp_path,
                language="es",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=True,
            )

            text = " ".join(segment.text for segment in segments).strip()
            language = info.language
            elapsed = round(time.time() - t0, 2)

            _LOG.info(
                "Transcripcion completada en %ss | idioma=%s | chars=%d",
                elapsed, language, len(text)
            )

            if not text:
                return {
                    "text": "",
                    "language": language,
                    "duration": elapsed,
                    "success": False,
                    "error": "No se detectó texto en el audio.",
                }

            return {
                "text": text,
                "language": language,
                "duration": elapsed,
                "success": True,
                "error": None,
            }

        finally:
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
    """Transcribe un archivo de audio desde disco."""
    path = Path(file_path)
    if not path.exists():
        return {
            "text": "",
            "language": "",
            "duration": 0,
            "success": False,
            "error": f"Archivo no encontrado: {file_path}",
        }
    return transcribe_audio(path.read_bytes(), filename=path.name)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        result = transcribe_audio_file(sys.argv[1])
        print(f"\nTexto: {result['text']}")
        print(f"Idioma: {result['language']}")
        print(f"Tiempo: {result['duration']}s")
    else:
        print("Uso: python -m src.multimodal.audio <archivo_audio>")
