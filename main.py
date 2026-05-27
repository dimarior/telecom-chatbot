"""
main.py
───────
Orquestador principal del pipeline Recamier Chatbot.

Pasos:
  PASO 1 → scraper.py   : Web scraping (recamier.com + saloninprofessional.com)
  PASO 2 → ingest.py    : Chunks + Embeddings + ChromaDB
  PASO 3 → rag_chain.py : Prueba de consulta
  PASO 4 → evaluate.py  : Métricas + MLflow

IMPORTANTE: Antes de correr este archivo:
  1. Configura .env con MISTRAL_API_KEY
  2. Inicia Ollama: ollama pull nomic-embed-text
  3. Inicia MLflow: mlflow server --host 127.0.0.1 --port 5000
  4. Instala Playwright: playwright install chromium
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.scraper import scrape_all
from src.ingest import build_vectorstore
from src.rag_chain import ask
from src.evaluate import evaluar


def run_pipeline():
    print("\n" + "=" * 60)
    print("  RECAMIER CHATBOT — PIPELINE COMPLETO")
    print("  Recamier + Salon In Professional")
    print("=" * 60)

    # ── PASO 1: Web Scraping ──────────────────────────────────────────────────
    print("\n📡 PASO 1 — Web Scraping...")
    print("   Objetivo: recamier.com + saloninprofessional.com")
    files = scrape_all()
    print(f"   ✅ Archivos generados: {[f.split('/')[-1] for f in files]}")

    # ── PASO 2: Vectorstore ───────────────────────────────────────────────────
    print("\n🔢 PASO 2 — Generando embeddings y vectorstore...")
    print("   (Puede tardar 5-15 minutos la primera vez)")
    build_vectorstore()

    # ── PASO 3: Prueba RAG ────────────────────────────────────────────────────
    print("\n🤖 PASO 3 — Prueba de consulta RAG...")
    pregunta = "¿Qué productos de Recamier son buenos para el cabello seco?"
    print(f"   Pregunta: {pregunta}")
    respuesta = ask(pregunta, session_id="test_pipeline")
    print(f"   Respuesta: {respuesta[:200]}...")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETO")
    print("")
    print("  Servicios disponibles:")
    print("  → MLflow:    http://127.0.0.1:5000")
    print("  → FastAPI:   http://127.0.0.1:8080/docs")
    print("  → Streamlit: http://localhost:8501")
    print("  → Prometheus:http://localhost:9090")
    print("  → Grafana:   http://localhost:3000")
    print("")
    print("  Comandos para levantar los servicios:")
    print("  uvicorn api.main:app --reload --host 127.0.0.1 --port 8080")
    print("  streamlit run app/streamlit_app.py")
    print("  python -m src.evaluate  (para evaluación completa)")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
