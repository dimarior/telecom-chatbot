# 🪄 Recamier Chatbot — RAG + MLOps

Chatbot inteligente para **Recamier** y **Salon In Professional** construido con:
- **Web Scraping** anti-detección (httpx + Playwright)
- **RAG** con ChromaDB + Ollama embeddings
- **LangGraph** para routing inteligente (direct / product / rag)
- **Mistral AI** como LLM
- **MLflow** para tracking, métricas y evaluación
- **Streamlit** para la interfaz de usuario
- **FastAPI** para la API REST
- **Prometheus + Grafana** para monitoreo

---

## 🏗️ Arquitectura

```
Usuario → Streamlit → FastAPI → LangGraph
                                    ├── classify_node (router LLM)
                                    ├── direct_node   (social / follow-up)
                                    ├── product_node  (catálogo JSON)
                                    ├── retrieve_node (ChromaDB RAG)
                                    └── generate_node (Mistral)
                                              ↓
                                         MLflow Tracing
```

**Pipeline de datos:**
```
recamier.com          ┐
saloninprofessional.com┘ → scraper.py → data/processed/*.txt
                                              ↓
                                         ingest.py → ChromaDB (vectorstore/)
```

---

## 🚀 Inicio rápido

### 1. Configurar entorno

```bash
cp .env.example .env
# Edita .env y agrega tu MISTRAL_API_KEY
```

### 2. Instalar dependencias

```bash
make setup
# o manualmente:
pip install -r requirements.txt
playwright install chromium
```

### 3. Iniciar MLflow

```bash
make mlflow
# → http://127.0.0.1:5000
```

### 4. Ejecutar el pipeline completo

```bash
# Opción A: Todo de una vez
make pipeline

# Opción B: Paso a paso
make scrape    # Web scraping (~15-30 min)
make ingest    # Generar vectorstore (~5-15 min)
```

### 5. Levantar los servicios

```bash
# Terminal 1: API
make api

# Terminal 2: Streamlit
make streamlit

# Terminal 3 (opcional): Prometheus + Grafana
make docker-up
```

Acceder a:
- **Streamlit**: http://localhost:8501
- **API Docs**: http://127.0.0.1:8080/docs
- **MLflow**: http://127.0.0.1:5000
- **Grafana**: http://localhost:3000 (admin/admin)

---

## 📁 Estructura del proyecto

```
recamier-chatbot/
├── app/
│   ├── assets/              # Logos, imágenes
│   └── streamlit_app.py     # Interfaz Streamlit
├── api/
│   └── main.py              # FastAPI + LangGraph
├── src/
│   ├── config.py            # Configuración centralizada
│   ├── scraper.py           # Web scraper anti-detección
│   ├── ingest.py            # Chunking + ChromaDB
│   ├── rag_chain.py         # Pipeline RAG con MLflow
│   ├── memory.py            # Memoria SQLite
│   ├── evaluate.py          # Evaluación RAG
│   ├── product_catalog.json # Datos estructurados
│   └── graph/
│       ├── state.py         # Estado LangGraph
│       ├── nodes.py         # Nodos del grafo
│       └── build.py         # Compilación del grafo
├── data/
│   ├── raw/                 # Datos crudos
│   └── processed/           # Texto procesado del scraping
├── vectorstore/             # ChromaDB (generado)
├── reports/evaluation/      # Métricas de evaluación
├── docker/
│   └── prometheus.yml
├── docker-compose.yml       # Prometheus + Grafana
├── main.py                  # Orquestador del pipeline
├── requirements.txt
├── Makefile
└── .env.example
```

---

## ⚙️ Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `MISTRAL_API_KEY` | Clave API de Mistral | *requerida* |
| `MISTRAL_MODEL` | Modelo Mistral | `mistral-small-latest` |
| `OLLAMA_EMBEDDING_MODEL` | Modelo embeddings | `nomic-embed-text` |
| `SCRAPE_MAX_PAGES` | Páginas máx por sitio | `200` |
| `SCRAPE_DELAY_MIN` | Delay mínimo entre requests (s) | `1.5` |
| `CHUNK_SIZE` | Tamaño de chunks | `700` |
| `RETRIEVER_K` | Docs recuperados por query | `4` |
| `MLFLOW_TRACKING_URI` | URL de MLflow | `http://127.0.0.1:5000` |

---

## 🔬 Evaluación

```bash
make evaluate
# → Genera reports/evaluation/resultados_evaluacion.json
# → Registra métricas en MLflow
```

---

## 🛡️ Buenas prácticas anti-detección del scraper

- Rotación de User-Agents (pool de 5+ navegadores reales)
- Delays aleatorios entre peticiones (1.5–4.0s configurable)
- Cabeceras HTTP completas imitando Chrome real
- Reintento con backoff exponencial (tenacity)
- Playwright como fallback para páginas con JS
- Semáforo de concurrencia (máx 4 conexiones simultáneas)
- HTTP/2 habilitado

---

## 📊 MLflow — Métricas registradas

| Run | Métricas |
|---|---|
| `scraping_recamier` | páginas scrapeadas, chars, tiempo por sitio |
| `build_vectorstore` | total_chunks, vectorstore_size, embedding_time |
| `evaluacion_rag_recamier` | score_q1..5, latencia_q1..5, avg_score_global |
| Cada query (`rag_pipeline_recamier`) | traced automáticamente con `@mlflow.trace` |
