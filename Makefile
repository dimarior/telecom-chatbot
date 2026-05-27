.PHONY: install install-playwright scrape ingest pipeline api streamlit mlflow docker-up docker-down evaluate clean help

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-playwright:
	playwright install chromium

setup: install install-playwright
	@echo "✅ Setup completo"
	@echo "👉 Siguiente paso: copia .env.example a .env y configura MISTRAL_API_KEY"

# ── Pipeline ──────────────────────────────────────────────────────────────────
scrape:
	@echo "🌐 Iniciando web scraping..."
	python -m src.scraper

ingest:
	@echo "🔢 Generando vectorstore..."
	python -m src.ingest

pipeline:
	@echo "🚀 Ejecutando pipeline completo..."
	python main.py

# ── Servicios ─────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 127.0.0.1 --port 8080

streamlit:
	streamlit run app/streamlit_app.py --server.port 8501

mlflow:
	mlflow server --host 127.0.0.1 --port 5000

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
	rm -f recamier_memory.db

clean-data:
	rm -f data/processed/*.txt

clean: clean-vectorstore clean-db
	@echo "🗑️  Limpieza completada (vectorstore + db)"

# ── Ayuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  RECAMIER CHATBOT — Comandos disponibles"
	@echo "  ────────────────────────────────────────"
	@echo "  make setup          → Instala dependencias + Playwright"
	@echo "  make scrape         → Scrapea recamier.com y saloninprofessional.com"
	@echo "  make ingest         → Genera embeddings y vectorstore"
	@echo "  make pipeline       → Ejecuta todo el pipeline"
	@echo "  make api            → Inicia API en :8080"
	@echo "  make streamlit      → Inicia Streamlit en :8501"
	@echo "  make mlflow         → Inicia MLflow UI en :5000"
	@echo "  make docker-up      → Levanta Prometheus + Grafana"
	@echo "  make evaluate       → Evalúa el sistema RAG"
	@echo "  make clean          → Limpia vectorstore y DB"
	@echo ""
