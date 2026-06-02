.PHONY: install install-playwright scrape ingest pipeline api streamlit mlflow docker-up docker-down evaluate clean help

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-playwright:
	playwright install chromium

setup: install install-playwright
	@echo "Setup completo"
	@echo "Siguiente paso: copia .env.example a .env y configura MISTRAL_API_KEY"

# ── Pipeline ──────────────────────────────────────────────────────────────────
scrape:
	@echo "Iniciando web scraping de Claro, Movistar y Tigo..."
	python -m src.scraper

ingest:
	@echo "Generando vectorstore con contenido de operadores..."
	python -m src.ingest

pipeline:
	@echo "Ejecutando pipeline completo GAIA..."
	python main.py

# ── Servicios ─────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 127.0.0.1 --port 8082

streamlit:
	streamlit run app/streamlit_app.py --server.port 8503

mlflow:
	mlflow server --host 127.0.0.1 --port 5002

# ── Docker (Prometheus + Grafana) ─────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ── Evaluación ────────────────────────────────────────────────────────────────
evaluate:
	python -m src.evaluate

# ── Limpieza ──────────────────────────────────────────────────────────────────
clean-vectorstore:
	rm -rf vectorstore/

clean-db:
	rm -f gaia_memory.db

clean-data:
	rm -f data/processed/*.txt

clean: clean-vectorstore clean-db
	@echo "Limpieza completada (vectorstore + db)"

# ── Ayuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  GAIA — Asistente Conversacional Telecomunicaciones Colombia"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make setup          -> Instala dependencias + Playwright"
	@echo "  make scrape         -> Scrapea Claro, Movistar y Tigo"
	@echo "  make ingest         -> Genera embeddings y vectorstore"
	@echo "  make pipeline       -> Ejecuta todo el pipeline"
	@echo "  make api            -> Inicia API en :8082"
	@echo "  make streamlit      -> Inicia Streamlit en :8503"
	@echo "  make mlflow         -> Inicia MLflow en :5002"
	@echo "  make docker-up      -> Levanta Prometheus + Grafana"
	@echo "  make evaluate       -> Evalua el sistema RAG"
	@echo "  make clean          -> Limpia vectorstore y DB"
	@echo ""