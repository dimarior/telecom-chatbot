"""
src/ingest.py
─────────────
Paso 2 del pipeline: lee los archivos .txt scrapeados y construye ChromaDB.

Flujo:
  1. Lee todos los archivos .txt en data/processed/
  2. Divide en chunks con RecursiveCharacterTextSplitter
  3. Genera embeddings con nomic-embed-text (Ollama local)
  4. Persiste en ChromaDB
  5. Registra métricas en MLflow
"""
from __future__ import annotations

import logging
import time

import mlflow
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_HOST,
    PROCESSED_DATA_DIR,
    VECTORSTORE_DIR,
)

_LOG = logging.getLogger("telecom.ingest")


def build_vectorstore() -> None:
    """
    Lee todos los .txt en data/processed/, genera embeddings y guarda en ChromaDB.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    txt_files = list(PROCESSED_DATA_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt en {PROCESSED_DATA_DIR}\n"
            "Ejecuta primero: python -m src.scraper"
        )

    _LOG.info("📄 Archivos encontrados: %s", [f.name for f in txt_files])

    with mlflow.start_run(run_name="build_vectorstore"):
        all_texts: list[str] = []
        all_metadatas: list[dict] = []

        for txt_path in txt_files:
            raw = txt_path.read_text(encoding="utf-8")
            # Cada documento separado por '---'
            pages = raw.split("\n\n---\n\n")
            for page in pages:
                if len(page.strip()) < 100:
                    continue
                # Extraer URL del encabezado del documento
                url = ""
                title = ""
                lines = page.strip().splitlines()
                for line in lines[:3]:
                    if line.startswith("URL: "):
                        url = line.replace("URL: ", "").strip()
                    elif line.startswith("Título: "):
                        title = line.replace("Título: ", "").strip()

                all_texts.append(page)
                all_metadatas.append({
                    "url": url,
                    "title": title,
                    "source": txt_path.stem,
                })

        _LOG.info("📊 Total de páginas cargadas: %d", len(all_texts))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["---", "\n\n", "\n", " "],
        )

        from langchain_core.documents import Document
        docs = []
        for text, meta in zip(all_texts, all_metadatas):
            sub_docs = splitter.create_documents([text], metadatas=[meta])
            # Propagar metadata a sub-chunks con índice
            for i, doc in enumerate(sub_docs):
                doc.metadata["chunk_index"] = i
            docs.extend(sub_docs)

        _LOG.info("🔪 Chunks generados: %d", len(docs))

        mlflow.log_param("embedding_model", OLLAMA_EMBEDDING_MODEL)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_metric("total_pages", len(all_texts))
        mlflow.log_metric("total_chunks", len(docs))

        _LOG.info("🔢 Generando embeddings con paraphrase-multilingual-mpnet-base-v2...")
        t0 = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )

        # Limpiar vectorstore previo si existe
        import shutil
        if VECTORSTORE_DIR.exists():
            shutil.rmtree(VECTORSTORE_DIR)
            _LOG.info("🗑️ Vectorstore anterior eliminado")

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )

        elapsed = round(time.time() - t0, 2)
        count = vectorstore._collection.count()

        mlflow.log_metric("vectorstore_size", count)
        mlflow.log_metric("embedding_time_s", elapsed)

        _LOG.info("✅ Vectorstore guardado: %d vectores en %.1fs", count, elapsed)
        _LOG.info("📁 Ruta: %s", VECTORSTORE_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_vectorstore()
