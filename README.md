# GAIA — Asistente Conversacional de Telecomunicaciones

## Descripcion General

GAIA es una plataforma conversacional inteligente desarrollada para el servicio al cliente de operadores de telecomunicaciones en Colombia. El sistema integra recuperacion aumentada de generacion (RAG), arquitectura de grafos conversacionales con LangGraph, modulos de procesamiento multimodal y principios de Human-Centered AI (HCAI) y GenAI UX para ofrecer una experiencia empatica, contextual y centrada en el usuario.

El proyecto es parte de una investigacion academica sobre experiencia de usuario en sistemas de inteligencia artificial generativa, orientada a evaluar metricas como NPS, SUS, empatia percibida y satisfaccion conversacional en comparacion con chatbots tradicionales.

La implementacion actual cubre los operadores Claro, Movistar y Tigo como caso de estudio, aunque la arquitectura es extensible a cualquier operador de telecomunicaciones mediante la configuracion de nuevas fuentes de datos en src/config.py y la regeneracion del vectorstore.

---

## Vista del Producto

### Interfaz principal
![Interfaz GAIA](app/assets/screenshot-interfaz.png)

### Selector de operadores
![Operadores GAIA](app/assets/screenshot-sidebar.png)

### Conversacion de ejemplo
![Conversacion GAIA](app/assets/screenshot-conversacion.png)

---

## Arquitectura del Sistema

El sistema se compone de cuatro capas principales: la interfaz de usuario (Streamlit), la capa de API (FastAPI), el motor de inteligencia conversacional (LangGraph + RAG) y el modulo multimodal (Whisper + EasyOCR + PyMuPDF).

### Flujo conversacional

```
Usuario (texto / voz / imagen / PDF)
   |
Modulo Multimodal (src/multimodal/)
   |  audio  → Whisper ASR → texto transcrito
   |  imagen → EasyOCR     → texto extraido
   |  PDF    → PyMuPDF     → texto extraido
   |
Streamlit (interfaz web)
   |
FastAPI (API REST)
   |
LangGraph (grafo conversacional)
   |
   +-- classify_node   (router de intenciones con deteccion emocional)
   |
   +-- direct_node     (respuestas sociales, empaticas y de acompanamiento)
   |
   +-- product_node    (consultas comerciales, planes y tarifas)
   |
   +-- retrieve_node   (busqueda semantica en ChromaDB)
   |
   +-- generate_node   (generacion de respuesta con Mistral AI)
                |
           MLflow Tracing
```

### Pipeline de datos

```
Web Scraping
   |
   +-- claro.com.co
   +-- movistar.com.co
   +-- tigo.com.co / ayuda.tigo.com.co
   |
Procesamiento de texto (src/scraper.py)
   |
data/processed/*.txt
   |
Chunking + Embeddings (src/ingest.py)
   |
ChromaDB (vectorstore/)
```

---

## Tecnologias Utilizadas

| Componente | Tecnologia |
|---|---|
| Interfaz de usuario | Streamlit |
| API REST | FastAPI + Uvicorn |
| Motor conversacional | LangGraph |
| Modelo de lenguaje | Mistral AI (mistral-small-latest) |
| Embeddings | Ollama (nomic-embed-text) |
| Base de datos vectorial | ChromaDB |
| Procesamiento de voz | Whisper (openai-whisper, CPU) |
| Procesamiento de imagenes | EasyOCR |
| Procesamiento de documentos | PyMuPDF (fitz) |
| Web scraping | httpx + Playwright + BeautifulSoup4 |
| Tracking y metricas RAG | MLflow |
| Evaluacion RAG avanzada | RAGAS |
| Memoria conversacional | SQLite |
| Monitoreo | Prometheus + Grafana |
| Gestion de dependencias | uv + pip |
| Entorno | Python 3.11 |

---

## Estructura del Proyecto

```
telecom-chatbot/
|
+-- app/
|   +-- assets/                  Logos, imagenes, favicon, banner, iconos multimodal
|   +-- streamlit_app.py         Interfaz GAIA con selector de operadores y multimodalidad
|
+-- api/
|   +-- main.py                  FastAPI: /ask, /ask/graph, /ask/audio, /ask/image, /ask/document
|   +-- __init__.py
|
+-- src/
|   +-- config.py                Configuracion centralizada desde .env
|   +-- scraper.py               Web scraper anti-deteccion con seed URLs
|   +-- ingest.py                Chunking, embeddings y construccion de ChromaDB
|   +-- rag_chain.py             Pipeline RAG simple con MLflow tracing
|   +-- memory.py                Memoria persistente en SQLite
|   +-- evaluate.py              Evaluacion del sistema RAG con metricas por operador
|   +-- evaluate_ragas.py        Evaluacion avanzada con RAGAS y eval gate
|   +-- product_catalog.json     Datos estructurados de operadores Claro/Movistar/Tigo
|   +-- __init__.py
|   |
|   +-- graph/
|   |   +-- state.py             Definicion del estado LangGraph (ChatState)
|   |   +-- nodes.py             Nodos del grafo con prompts GenAI UX y HCAI
|   |   +-- build.py             Compilacion del grafo con checkpointer SQLite
|   |   +-- __init__.py
|   |
|   +-- multimodal/
|       +-- audio.py             Transcripcion de voz con Whisper (CPU)
|       +-- image.py             Extraccion de texto de imagenes con EasyOCR
|       +-- document.py          Extraccion de texto de PDFs con PyMuPDF
|       +-- __init__.py
|
+-- data/
|   +-- raw/                     Datos crudos (no versionados)
|   +-- processed/               Archivos .txt generados por el scraper
|
+-- vectorstore/                 ChromaDB persistido (generado, no versionado)
+-- reports/
|   +-- evaluation/              Resultados JSON de evaluacion RAG y RAGAS
|
+-- docker/
|   +-- prometheus.yml           Configuracion de Prometheus
|
+-- tests/
|   +-- test_scraper.py          Tests unitarios del scraper
|   +-- test_api.py              Tests basicos de la API
|
+-- main.py                      Orquestador del pipeline completo
+-- requirements.txt             Dependencias del proyecto
+-- docker-compose.yml           Prometheus + Grafana
+-- Makefile                     Comandos de gestion del proyecto
+-- .env.example                 Plantilla de variables de entorno
+-- .gitignore
+-- README.md
```

---

## Modulo Multimodal

GAIA procesa tres tipos de entrada ademas del texto, convirtiendo cada modalidad a texto antes de entrar al grafo conversacional LangGraph.

### Audio — Whisper

El modulo `src/multimodal/audio.py` transcribe mensajes de voz usando Whisper de OpenAI en modo CPU. La interfaz permite grabar directamente desde el navegador usando `st.audio_input` de Streamlit.

| Parametro | Valor | Descripcion |
|---|---|---|
| Modelo | base | Balance entre velocidad y precision en CPU |
| Idioma | es | Espanol colombiano por defecto |
| Formatos | wav, mp3, ogg, m4a, flac | Formatos de audio soportados |
| Variable de entorno | WHISPER_MODEL | Configurable en .env |

Nota: en Windows se requiere ffmpeg instalado para el procesamiento de audio. En despliegues Linux (Streamlit Cloud, Railway) ffmpeg viene preinstalado.

### Imagen — EasyOCR

El modulo `src/multimodal/image.py` extrae texto de imagenes usando EasyOCR con soporte para espanol e ingles. Casos de uso: fotos de routers, capturas de pantalla de errores, imagenes de facturas.

| Parametro | Valor |
|---|---|
| Idiomas | es, en |
| GPU | False (CPU) |
| Confianza minima | 0.3 |
| Formatos | jpg, jpeg, png, bmp, webp |

### Documento — PyMuPDF

El modulo `src/multimodal/document.py` extrae texto de archivos PDF usando PyMuPDF (fitz). Casos de uso: facturas en PDF, contratos, comprobantes de pago.

| Parametro | Valor |
|---|---|
| Biblioteca | PyMuPDF (fitz) |
| Formatos | PDF |
| Limite de contexto | 3000 caracteres |

### Endpoints multimodal

| Endpoint | Metodo | Descripcion |
|---|---|---|
| /ask/audio | POST | Recibe audio, transcribe con Whisper y procesa con LangGraph |
| /ask/image | POST | Recibe imagen, extrae texto con EasyOCR y procesa con LangGraph |
| /ask/document | POST | Recibe PDF, extrae texto con PyMuPDF y procesa con LangGraph |

---

## Arquitectura Conversacional

### Router de intenciones

El nodo `classify_node` analiza cada mensaje del usuario y determina una de tres rutas posibles:

- **direct**: saludos, despedidas, agradecimientos, expresiones emocionales, frustraciones generales, referencias a turnos anteriores, temas fuera del dominio de telecomunicaciones.
- **product**: consultas sobre planes, precios, tarifas, portabilidad numerica, puntos de atencion, activacion o cancelacion de servicios.
- **rag**: soporte tecnico, facturacion, recargas, autogestion, preguntas frecuentes, cobertura, procedimientos paso a paso.

El router tambien detecta senales emocionales implicitas como frustracion, urgencia o confusion para priorizar respuestas empaticas antes que tecnicas.

### Principios de GenAI UX y Human-Centered AI

Los prompts del sistema implementan los siguientes principios:

- Empatia conversacional: validacion emocional ante frustracion o urgencia antes de responder tecnicamente.
- UX Writing: lenguaje claro, simple, sin tecnicismos, con frases cortas y conversacionales.
- Adaptacion de tono: soporte tecnico (empatico y guiado), facturacion (claro y tranquilizador), consulta comercial (orientador), frustracion (contencion primero, solucion despues).
- Fallback humanizado: cuando la informacion no esta disponible, se orienta al usuario con calidez.
- Continuidad conversacional: el historial se aprovecha para mantener coherencia sin tratar cada mensaje como una consulta nueva.
- Grounding estricto: las respuestas se basan exclusivamente en el contexto recuperado de ChromaDB.

### Endpoints de la API

| Endpoint | Descripcion |
|---|---|
| GET /health | Estado del sistema e informacion de modulos multimodal |
| POST /ask | Pipeline RAG simple (sin LangGraph) |
| POST /ask/graph | Pipeline con LangGraph (router + memoria checkpointer) |
| POST /ask/audio | Pipeline con entrada de voz (Whisper) |
| POST /ask/image | Pipeline con entrada de imagen (EasyOCR) |
| POST /ask/document | Pipeline con entrada de PDF (PyMuPDF) |

---

## Web Scraping

El modulo `src/scraper.py` implementa un crawler BFS (Breadth-First Search) con las siguientes caracteristicas anti-deteccion:

- Rotacion de User-Agents con pool de navegadores reales (Chrome, Firefox, Edge, Safari).
- Cabeceras HTTP completas que imitan un navegador Chrome real.
- Delays aleatorios configurables entre peticiones.
- Reintento con backoff exponencial usando tenacity (hasta 3 intentos).
- Playwright como fallback para paginas que requieren JavaScript.
- Semaforo de concurrencia configurable.
- Extraccion de texto limpio sin scripts, estilos, navegacion ni publicidad.

### Sitios scrapeados

| Operador | URLs semilla |
|---|---|
| Claro | claro.com.co/personas/faqs/, claro.com.co/personas/autogestion/, claro.com.co/personas/servicios/ |
| Movistar | movistar.com.co/atencion-al-cliente/, descubre.movistar.co/atencion-cliente/, movistar.com.co/procesos-autogestion |
| Tigo | tigo.com.co/preguntas-frecuentes-servicios-tigo, ayuda.tigo.com.co/hc/centro-de-ayuda/es |

---

## Configuracion y Variables de Entorno

Copia el archivo `.env.example` a `.env` y configura las variables necesarias.

| Variable | Descripcion | Valor por defecto |
|---|---|---|
| MISTRAL_API_KEY | Clave API de Mistral AI | Requerida |
| MISTRAL_MODEL | Modelo de Mistral a utilizar | mistral-small-latest |
| OLLAMA_HOST | URL del servidor Ollama | http://localhost:11434 |
| OLLAMA_EMBEDDING_MODEL | Modelo de embeddings | nomic-embed-text |
| CHUNK_SIZE | Tamano de chunks en caracteres | 700 |
| CHUNK_OVERLAP | Solapamiento entre chunks | 100 |
| RETRIEVER_K | Documentos recuperados por consulta | 4 |
| MIN_SCORE | Score minimo de similitud | 0.35 |
| SCRAPE_CONCURRENCY | Conexiones simultaneas del scraper | 3 |
| SCRAPE_MAX_PAGES | Paginas maximas por sitio | 500 |
| SCRAPE_DELAY_MIN | Delay minimo entre peticiones (segundos) | 1.2 |
| SCRAPE_DELAY_MAX | Delay maximo entre peticiones (segundos) | 3.5 |
| MLFLOW_TRACKING_URI | URL del servidor MLflow | http://127.0.0.1:5002 |
| EXPERIMENT_NAME | Nombre del experimento en MLflow | telecom-chatbot-rag |
| API_HOST | Host de la API | 0.0.0.0 |
| API_PORT | Puerto de la API | 8082 |
| WHISPER_MODEL | Modelo Whisper para transcripcion de voz | base |

---

## Instalacion y Ejecucion

### Requisitos previos

- Python 3.11
- uv (gestor de dependencias)
- Ollama instalado y corriendo con el modelo nomic-embed-text
- Cuenta en Mistral AI con clave API activa
- ffmpeg instalado en Windows (requerido por Whisper)
- Docker (opcional, para Prometheus y Grafana)

### Paso 1 — Configurar entorno

```bash
cp .env.example .env
# Editar .env y configurar MISTRAL_API_KEY y demas variables
```

### Paso 2 — Crear entorno virtual e instalar dependencias

```bash
uv venv .venv --python 3.11
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux / macOS

.venv/Scripts/python.exe -m pip install -r requirements.txt
playwright install chromium
```

### Paso 3 — Instalar dependencias multimodal

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper easyocr pymupdf
```

En Windows instalar ffmpeg:

```bash
winget install ffmpeg
```

### Paso 4 — Descargar modelo de embeddings

```bash
ollama pull nomic-embed-text
```

### Paso 5 — Iniciar MLflow

```bash
mlflow server --host 127.0.0.1 --port 5002
```

### Paso 6 — Ejecutar el scraping

```bash
python -m src.scraper
```

### Paso 7 — Generar el vectorstore

```bash
python -m src.ingest
```

### Paso 8 — Levantar los servicios

Abrir cuatro terminales independientes con el entorno virtual activado:

Terminal 1 — MLflow:
```bash
mlflow server --host 127.0.0.1 --port 5002
```

Terminal 2 — API:
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8082
```

Terminal 3 — Interfaz Streamlit:
```bash
streamlit run app/streamlit_app.py --server.port 8503
```

Terminal 4 — Monitoreo (opcional):
```bash
docker compose up -d
```

### Acceso a los servicios

| Servicio | URL |
|---|---|
| Interfaz GAIA | http://localhost:8503 |
| Documentacion API | http://127.0.0.1:8082/docs |
| MLflow | http://127.0.0.1:5002 |
| Grafana | http://localhost:3000 (admin / admin) |
| Prometheus | http://localhost:9090 |

### Nota sobre monitoreo

El sistema cuenta con una capa de observabilidad en dos niveles. MLflow registra las metricas del pipeline de IA — latencia, scores RAG y experimentos. Prometheus y Grafana monitorean la infraestructura — uso de recursos y disponibilidad de los servicios en tiempo real.

Para generar metricas en MLflow:

```bash
python -m src.evaluate
```

Para evaluacion avanzada con RAGAS:

```bash
python -m src.evaluate_ragas
```

---

## Comandos rapidos con Makefile

| Comando | Equivale a |
|---|---|
| make setup | pip install -r requirements.txt + playwright install |
| make scrape | python -m src.scraper |
| make ingest | python -m src.ingest |
| make api | uvicorn api.main:app --reload --host 127.0.0.1 --port 8082 |
| make streamlit | streamlit run app/streamlit_app.py --server.port 8503 |
| make mlflow | mlflow server --host 127.0.0.1 --port 5002 |
| make evaluate | python -m src.evaluate |
| make clean | Elimina vectorstore/ y gaia_memory.db |

---

## Evaluacion del Sistema RAG

### Evaluacion basica

El modulo `src/evaluate.py` ejecuta 15 preguntas distribuidas por operador y registra metricas en MLflow.

```bash
python -m src.evaluate
```

### Evaluacion avanzada con RAGAS

```bash
python -m src.evaluate_ragas
```

| Metrica | Descripcion | Target |
|---|---|---|
| Faithfulness | Fidelidad de la respuesta al contexto recuperado | > 0.85 |
| Answer Relevancy | Relevancia de la respuesta a la pregunta | > 0.80 |
| Context Precision | Proporcion de chunks relevantes recuperados | > 0.75 |
| Context Recall | Cobertura de informacion necesaria | > 0.80 |

---

## Metricas MLflow

| Experimento | Metricas registradas |
|---|---|
| build_vectorstore | total_chunks, vectorstore_size, embedding_time_s |
| evaluacion_rag_gaia | score por pregunta, latencia, avg_score_global, avg_score por operador |
| evaluacion_ragas_gaia | faithfulness, answer_relevancy, context_precision, context_recall, eval_gate |
| rag_pipeline_gaia | trazado automaticamente con @mlflow.trace por cada consulta |

---

## Memoria Conversacional

El sistema utiliza SQLite para persistir el historial de conversaciones entre sesiones. La base de datos se crea automaticamente en `gaia_memory.db` al iniciar el sistema.

El checkpointer de LangGraph tambien utiliza SQLite para mantener el estado del grafo por sesion, lo que permite continuidad conversacional real entre turnos.

---

## Consideraciones de Despliegue

- El vectorstore debe regenerarse cada vez que se actualice el contenido scrapeado.
- El archivo `.env` no debe versionarse. Usar `.env.example` como referencia.
- Las carpetas `vectorstore/`, `data/processed/*.txt` y `gaia_memory.db` no se versionan.
- En despliegues Linux (Streamlit Cloud, Railway, Render) ffmpeg viene preinstalado.
- Los modelos de Whisper y EasyOCR se descargan automaticamente en el primer uso.

---

## Investigacion Academica

Este proyecto forma parte de una investigacion sobre experiencia de usuario en sistemas de inteligencia artificial generativa. Los objetivos de investigacion incluyen:

- Evaluar la percepcion de empatia en chatbots basados en GenAI UX vs chatbots tradicionales.
- Medir usabilidad mediante la escala SUS (System Usability Scale de Brooke, 1996) — instrumento de 10 items con score de 0 a 100 donde valores superiores a 68 indican buena usabilidad percibida.
- Analizar satisfaccion del usuario mediante NPS (Net Promoter Score).
- Medir eficiencia operativa mediante la Tasa de Resolucion en Primer Contacto (TRPC).
- Comparar experiencia conversacional entre un chatbot tradicional y una arquitectura conversacional basada en GenAI UX.
- Estudiar el impacto de principios de Human-Centered AI en la percepcion de utilidad, confianza y cercania del asistente.
- Evaluar la calidad del sistema RAG mediante metricas objetivas: faithfulness, answer relevancy, context precision y context recall (Gao et al., 2024).

Los prompts del sistema implementan principios de Conversational UX, UX Writing, empatia operacional y adaptive tone response, alineados con el marco teorico de la investigacion.