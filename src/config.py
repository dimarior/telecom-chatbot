"""
src/config.py
─────────────
Configuración centralizada del proyecto Chatbot RAG.
Lee variables desde .env con fallback a valores por defecto.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
REPORTS_DIR = ROOT_DIR / "reports"
EVALUATION_DIR = REPORTS_DIR / "evaluation"
DB_PATH = ROOT_DIR / "gaia_memory.db"

# ── LLM ──────────────────────────────────────────────────────────────────────
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.35"))
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "10"))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))

# ── Scraping ──────────────────────────────────────────────────────────────────
SCRAPE_CONCURRENCY: int = int(os.getenv("SCRAPE_CONCURRENCY", "3"))
SCRAPE_TIMEOUT_S: int = int(os.getenv("SCRAPE_TIMEOUT_S", "25"))
SCRAPE_DELAY_MIN: float = float(os.getenv("SCRAPE_DELAY_MIN", "1.2"))
SCRAPE_DELAY_MAX: float = float(os.getenv("SCRAPE_DELAY_MAX", "3.5"))
SCRAPE_MAX_PAGES: int = int(os.getenv("SCRAPE_MAX_PAGES", "1000"))

# URLs objetivo de scraping
SCRAPE_TARGETS: list[dict] = [
    {
        "name": "movistar",
        "base_url": "https://www.movistar.com.co/",
        "output_file": "movistar_content.txt",
        "allowed_domains": ["movistar.com.co", "descubre.movistar.co"],
        "seed_urls": [
            "https://descubre.movistar.co/atencion-cliente/politicas-de-privacidad/faqs.html",
            "https://www.movistar.com.co/atencion-al-cliente/atencion-en-linea",
            "https://www.movistar.com.co/atencion-al-cliente/asistencia-tecnica",
            "https://descubre.movistar.co/movistar-accesible/preguntas-television.html",
            "https://www.movistar.com.co/procesos-autogestion",
        ],
    },
    {
        "name": "claro",
        "base_url": "https://www.claro.com.co/",
        "output_file": "claro_content.txt",
        "allowed_domains": ["claro.com.co"],
        "seed_urls": [
            "https://www.claro.com.co/personas/faqs/",
            "https://www.claro.com.co/institucional/soporte-tecnico-claro/",
            "https://www.claro.com.co/personas/autogestion/whatsapp/",
            "https://www.claro.com.co/personas/autogestion/",
            "https://www.claro.com.co/personas/servicios/servicios-moviles/",
            "https://www.claro.com.co/personas/servicios/servicios-hogar/",
            "https://www.claro.com.co/personas/legal-y-regulatorio/",
        ],
    },
    {
        "name": "tigo",
        "base_url": "https://www.tigo.com.co/",
        "output_file": "tigo_content.txt",
        "allowed_domains": ["tigo.com.co", "ayuda.tigo.com.co"],
        "seed_urls": [
            "https://www.tigo.com.co/preguntas-frecuentes-servicios-tigo",
            "https://ayuda.tigo.com.co/hc/centro-de-ayuda/es",
            "https://ayuda.tigo.com.co/hc/centro-de-ayuda/es/categories/general",
        ],
    },
]

EXPERIMENT_NAME: str = os.getenv("EXPERIMENT_NAME", "telecom-chatbot-rag")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5002")
EXPERIMENT_NAME: str = os.getenv("EXPERIMENT_NAME", "telecom-chatbot-rag")

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8082"))
API_TITLE: str = "Telecom Chatbot API"

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "telecom-chatbot")

# ── Crear directorios al importar ─────────────────────────────────────────────
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORSTORE_DIR, EVALUATION_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
