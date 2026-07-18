# GAIA — Asistente Conversacional de Telecomunicaciones

## Descripción General

GAIA es una plataforma conversacional inteligente desarrollada para el servicio al cliente de operadores de telecomunicaciones en Colombia. El sistema integra recuperación aumentada de generación (RAG), arquitectura de grafos conversacionales con LangGraph, módulos de procesamiento multimodal y principios de Human-Centered AI (HCAI) y GenAI UX para ofrecer una experiencia empática, contextual y centrada en el usuario.

El proyecto es parte de una investigación académica sobre experiencia de usuario en sistemas de inteligencia artificial generativa, orientada a evaluar métricas como NPS, SUS, empatía percibida y satisfacción conversacional en comparación con chatbots tradicionales.

La implementación actual cubre los operadores Claro, Movistar y Tigo como caso de estudio, aunque la arquitectura es extensible a cualquier operador de telecomunicaciones mediante la configuración de nuevas fuentes de datos en src/config.py y la regeneración del vectorstore.

---

## Arquitectura del Sistema

El sistema se compone de cuatro capas principales: la interfaz de usuario (Streamlit), la capa de API (FastAPI), el motor de inteligencia conversacional (LangGraph + RAG) y el módulo multimodal (faster-whisper + EasyOCR + PyMuPDF).

### Flujo conversacional

```
Usuario (texto / voz / imagen / PDF)
   |
M�dulo Multimodal (src/multimodal/)
   |  audio  → faster-whisper ASR → texto transcrito automáticamente
   |  imagen → EasyOCR            → texto extraído
   |  PDF    → PyMuPDF            → texto extraído (con fallback a EasyOCR si es escaneado)
   |
Streamlit (interfaz web)
   |
FastAPI (API REST)
   |
LangGraph (grafo conversacional)
   |
   +-- classify_node   (router de intenciones con detección emocional)
   |
   +-- direct_node     (respuestas sociales, empáticas y de acompañamiento)
   |
   +-- product_node    (consultas comerciales, planes y tarifas)
   |
   +-- retrieve_node   (búsqueda semántica en ChromaDB con sentence-transformers)
   |
   +-- generate_node   (generación de respuesta con Mistral AI)
                |
           MLflow Tracing
```

### Pipeline de datos

```
Web Scraping (src/scraper.py)
   |
   +-- claro.com.co
   +-- movistar.com.co
   +-- tigo.com.co / ayuda.tigo.com.co
   |
Procesamiento y limpieza de texto
   |
data/processed/*.txt
   |
Chunking (700 caracteres, overlap 100)
   |
Embeddings con sentence-transformers
(paraphrase-multilingual-mpnet-base-v2)
   |
ChromaDB (vectorstore/)
```

---

## Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| Interfaz de usuario | Streamlit |
| API REST | FastAPI + Uvicorn |
| Motor conversacional | LangGraph |
| Modelo de lenguaje | Mistral AI (mistral-small-latest) |
| Embeddings | sentence-transformers (paraphrase-multilingual-mpnet-base-v2) |
| Base de datos vectorial | ChromaDB |
| Procesamiento de voz | faster-whisper (modelo small, CPU) |
| Procesamiento de imágenes | EasyOCR |
| Procesamiento de documentos | PyMuPDF (fitz) |
| Web scraping | httpx + Playwright + BeautifulSoup4 |
| Tracking y métricas RAG | MLflow |
| Evaluación RAG avanzada | RAGAS |
| Memoria conversacional | SQLite |
| Monitoreo | Prometheus + Grafana |
| Gestión de dependencias | pip |
| Entorno | Python 3.11 |

---

## Estructura del Proyecto

```
telecom-chatbot/
|
+-- app/
|   +-- assets/                  Logos, imágenes, favicon, banner, íconos multimodal
|   +-- streamlit_app.py         Interfaz GAIA con selector de operadores y multimodalidad
|
+-- api/
|   +-- main.py                  FastAPI: /health, /transcribe, /ask, /ask/graph,
|   |                            /ask/audio, /ask/image, /ask/document
|   +-- __init__.py
|
+-- src/
|   +-- config.py                Configuración centralizada desde .env
|   +-- scraper.py               Web scraper anti-detección con seed URLs por operador
|   +-- ingest.py                Chunking, embeddings con sentence-transformers y ChromaDB
|   +-- rag_chain.py             Pipeline RAG simple con MLflow tracing
|   +-- memory.py                Memoria persistente en SQLite
|   +-- evaluate.py              Evaluación RAG con 15 preguntas por operador
|   +-- evaluate_ragas.py        Evaluación avanzada con RAGAS y eval gate
|   +-- product_catalog.json     Datos estructurados de operadores Claro/Movistar/Tigo
|   +-- __init__.py
|   |
|   +-- graph/
|   |   +-- state.py             Definición del estado LangGraph (ChatState)
|   |   +-- nodes.py             Nodos del grafo con prompts GenAI UX y HCAI
|   |   +-- build.py             Compilación del grafo con checkpointer SQLite
|   |   +-- __init__.py
|   |
|   +-- multimodal/
|       +-- audio.py             Transcripción automática de voz con faster-whisper
|       +-- image.py             Extracción de texto de imágenes con EasyOCR
|       +-- document.py          Extracción de texto de PDFs con PyMuPDF y fallback OCR
|       +-- __init__.py
|
+-- data/
|   +-- raw/                     Datos crudos (no versionados)
|   +-- processed/               Archivos .txt generados por el scraper
|
+-- vectorstore/                 ChromaDB persistido (generado localmente, no versionado)
+-- reports/
|   +-- evaluation/              Resultados JSON de evaluación RAG y RAGAS
|
+-- docker/
|   +-- prometheus.yml           Configuración de Prometheus
|
+-- tests/
|   +-- test_scraper.py          Tests unitarios del scraper
|   +-- test_api.py              Tests básicos de la API
|
+-- .streamlit/
|   +-- config.toml              Configuración de Streamlit (límite de archivos 5MB)
|
+-- main.py                      Orquestador del pipeline completo
+-- requirements.txt             Dependencias del proyecto
+-- packages.txt                 Dependencias del sistema para despliegue en la nube
+-- docker-compose.yml           Prometheus + Grafana
+-- Makefile                     Comandos de gestión del proyecto
+-- .env.example                 Plantilla de variables de entorno
+-- .gitignore
+-- README.md
```

---

## Módulo Multimodal

GAIA procesa tres tipos de entrada además del texto, convirtiendo cada modalidad a texto antes de entrar al grafo conversacional LangGraph. Este enfoque, denominado pipeline multimodal en la literatura, es el estándar en sistemas RAG multimodales de código abierto (Gao et al., 2024).

### Audio — faster-whisper

El módulo `src/multimodal/audio.py` transcribe mensajes de voz usando faster-whisper en modo CPU. La transcripción es automática: al terminar de grabar, el sistema transcribe el audio y carga el texto directamente en el área de consulta para que el usuario lo revise antes de enviarlo.

faster-whisper fue seleccionado sobre openai-whisper por no requerir ffmpeg como dependencia del sistema operativo, lo que garantiza compatibilidad tanto en Windows como en despliegues Linux en la nube.

| Parámetro | Valor | Descripción |
|---|---|---|
| Biblioteca | faster-whisper | No requiere ffmpeg |
| Modelo | small | Mayor precisión en español colombiano |
| Idioma | es | Español por defecto |
| Formatos | wav, mp3, ogg, m4a, flac | Formatos de audio soportados |
| Variable de entorno | WHISPER_MODEL | Configurable en .env |

### Imagen — EasyOCR

El módulo `src/multimodal/image.py` extrae texto de imágenes usando EasyOCR con soporte para español e inglés. El tamaño máximo de imagen aceptado es 5 MB, validado tanto en el frontend como en el backend.

Casos de uso: fotos de routers, capturas de pantalla de errores, imágenes de facturas.

| Parámetro | Valor |
|---|---|
| Idiomas | es, en |
| GPU | False (CPU) |
| Confianza mínima | 0.3 |
| Formatos | jpg, jpeg, png, bmp, webp |
| Tamaño máximo | 5 MB |

### Documento — PyMuPDF con fallback OCR

El módulo `src/multimodal/document.py` extrae texto de archivos PDF con un mecanismo de doble capa. Primero intenta extracción directa con PyMuPDF. Si el PDF es un documento escaneado sin texto seleccionable, activa automáticamente EasyOCR como fallback: rasteriza cada página a imagen con 200 DPI y aplica reconocimiento óptico de caracteres.

El prompt generado para el LLM se adapta según el origen del texto: indica si proviene de extracción directa o de OCR, para que el modelo considere posibles desórdenes en el texto.

| Parámetro | Valor |
|---|---|
| Biblioteca principal | PyMuPDF (fitz) |
| Fallback OCR | EasyOCR (activado automáticamente) |
| Formatos | PDF |
| Límite de contexto | 3000 caracteres |
| Tamaño máximo | 5 MB |

### Endpoints multimodal

| Endpoint | Método | Descripción |
|---|---|---|
| /transcribe | POST | Solo transcribe audio y retorna el texto sin procesar con LangGraph |
| /ask/audio | POST | Transcribe audio con faster-whisper y procesa con LangGraph |
| /ask/image | POST | Extrae texto de imagen con EasyOCR y procesa con LangGraph |
| /ask/document | POST | Extrae texto de PDF con PyMuPDF y procesa con LangGraph |

---

## Arquitectura Conversacional

### Embeddings con sentence-transformers

El sistema utiliza el modelo `paraphrase-multilingual-mpnet-base-v2` de la biblioteca sentence-transformers (Reimers y Gurevych, 2019) para generar los vectores de embeddings. Este modelo fue seleccionado por su entrenamiento específico en corpus multilingües que incluyen español, su amplio respaldo en la literatura de sistemas RAG y su compatibilidad con despliegues en la nube sin requerir infraestructura adicional.

A diferencia de Ollama, sentence-transformers corre directamente en Python sin necesidad de un servidor externo, lo que garantiza compatibilidad con Streamlit Community Cloud y otros entornos de despliegue en la nube.

### Router de intenciones

El nodo `classify_node` analiza cada mensaje del usuario y determina una de tres rutas posibles:

- direct: saludos, despedidas, agradecimientos, expresiones emocionales, frustraciones sin pregunta técnica clara, referencias a turnos anteriores, temas fuera del dominio de telecomunicaciones.
- product: consultas sobre planes, precios, tarifas, portabilidad numérica, puntos de atención, activación o cancelación de servicios.
- rag: soporte técnico, facturación, recargas, autogestión, preguntas frecuentes, cobertura, procedimientos paso a paso.

El router detecta señales emocionales implícitas como frustración, urgencia o confusión. Ante expresiones de frustración extrema, GAIA valida brevemente la emoción y redirige al problema de telecomunicaciones, manteniendo el foco en el dominio sin salirse del contexto del servicio.

### Principios de GenAI UX y Human-Centered AI

Los prompts del sistema implementan los siguientes principios:

- Empatía conversacional: validación emocional ante frustración o urgencia antes de responder técnicamente, siempre dentro del contexto de telecomunicaciones.
- UX Writing: lenguaje claro, simple, sin tecnicismos, con frases cortas y conversacionales.
- Adaptación de tono: soporte técnico (empático y guiado), facturación (claro y tranquilizador), consulta comercial (orientador).
- Fallback humanizado: cuando la información no está disponible, se orienta al usuario hacia el canal correcto con calidez.
- Continuidad conversacional: el historial se aprovecha para mantener coherencia sin tratar cada mensaje como una consulta nueva.
- Grounding estricto: las respuestas se basan exclusivamente en el contexto recuperado de ChromaDB. El sistema no usa información del contexto que no sea directamente relacionada con servicios de Claro, Movistar o Tigo.

### Endpoints de la API

| Endpoint | Descripción |
|---|---|
| GET /health | Estado del sistema e información de módulos activos |
| POST /transcribe | Transcripción de audio sin procesamiento conversacional |
| POST /ask | Pipeline RAG simple (sin LangGraph) |
| POST /ask/graph | Pipeline completo con LangGraph, router y memoria |
| POST /ask/audio | Pipeline con entrada de voz (faster-whisper) |
| POST /ask/image | Pipeline con entrada de imagen (EasyOCR) |
| POST /ask/document | Pipeline con entrada de PDF (PyMuPDF + fallback OCR) |

---

## Web Scraping

El módulo `src/scraper.py` implementa un crawler BFS (Breadth-First Search) con las siguientes características anti-detección:

- Rotación de User-Agents con pool de navegadores reales.
- Cabeceras HTTP completas que imitan un navegador Chrome real.
- Delays aleatorios configurables entre peticiones.
- Reintento con backoff exponencial usando tenacity.
- Playwright como fallback para páginas que requieren JavaScript.
- Extracción de texto limpio sin scripts, estilos, navegación ni publicidad.

### Sitios scrapeados

| Operador | URLs semilla |
|---|---|
| Claro | claro.com.co/personas/faqs/, claro.com.co/personas/autogestion/, claro.com.co/personas/servicios/ |
| Movistar | movistar.com.co/atencion-al-cliente/, descubre.movistar.co/atencion-cliente/, movistar.com.co/procesos-autogestion |
| Tigo | tigo.com.co/preguntas-frecuentes-servicios-tigo, ayuda.tigo.com.co/hc/centro-de-ayuda/es |

---

## Configuración y Variables de Entorno

Copia el archivo `.env.example` a `.env` y configura las variables necesarias.

| Variable | Descripción | Valor por defecto |
|---|---|---|
| MISTRAL_API_KEY | Clave API de Mistral AI | Requerida |
| MISTRAL_MODEL | Modelo de Mistral a utilizar | mistral-small-latest |
| CHUNK_SIZE | Tamaño de chunks en caracteres | 700 |
| CHUNK_OVERLAP | Solapamiento entre chunks | 100 |
| RETRIEVER_K | Documentos recuperados por consulta | 4 |
| MIN_SCORE | Score mínimo de similitud | 0.1 |
| SCRAPE_CONCURRENCY | Conexiones simultáneas del scraper | 3 |
| SCRAPE_MAX_PAGES | Páginas máximas por sitio | 500 |
| SCRAPE_DELAY_MIN | Delay mínimo entre peticiones (segundos) | 1.2 |
| SCRAPE_DELAY_MAX | Delay máximo entre peticiones (segundos) | 3.5 |
| MLFLOW_TRACKING_URI | URL del servidor MLflow | http://127.0.0.1:5002 |
| EXPERIMENT_NAME | Nombre del experimento en MLflow | telecom-chatbot-rag |
| API_HOST | Host de la API | 0.0.0.0 |
| API_PORT | Puerto de la API | 8082 |
| WHISPER_MODEL | Modelo faster-whisper para transcripción | small |

---

## Instalación y Ejecución Local

### Requisitos previos

- Python 3.11
- Cuenta en Mistral AI con clave API activa
- Docker (opcional, para Prometheus y Grafana)

### Paso 1 — Configurar entorno

```bash
cp .env.example .env
# Editar .env y configurar MISTRAL_API_KEY
```

### Paso 2 — Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
playwright install chromium
```

### Paso 3 — Instalar dependencias multimodal

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper easyocr pymupdf
pip install sentence-transformers langchain-huggingface
```

En Windows instalar ffmpeg (opcional):

```bash
winget install ffmpeg
```

### Paso 4 — Iniciar MLflow

```bash
mlflow server --host 127.0.0.1 --port 5002
```

### Paso 5 — Ejecutar el scraping

```bash
python -m src.scraper
```

Este proceso puede tardar entre 30 y 60 minutos. Al finalizar genera los archivos claro_content.txt, movistar_content.txt y tigo_content.txt en data/processed/.

### Paso 6 — Generar el vectorstore

```bash
python -m src.ingest
```

Este proceso descarga el modelo sentence-transformers la primera vez (~420MB), genera los embeddings y construye la base de datos vectorial en vectorstore/. Puede tardar entre 15 y 30 minutos.

### Paso 7 — Levantar los servicios

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
| Documentación API | http://127.0.0.1:8082/docs |
| MLflow | http://127.0.0.1:5002 |
| Grafana | http://localhost:3000 (admin / admin) |
| Prometheus | http://localhost:9090 |

### Nota sobre monitoreo

El sistema cuenta con una capa de observabilidad en dos niveles. MLflow registra las métricas del pipeline de IA: latencia, scores RAG y experimentos. Prometheus y Grafana monitorean la infraestructura: uso de recursos y disponibilidad de los servicios en tiempo real.

Para generar métricas en MLflow:

```bash
python -m src.evaluate
```

Para evaluación avanzada con RAGAS:

```bash
python -m src.evaluate_ragas
```

---

## Despliegue en Streamlit Community Cloud

Streamlit Community Cloud permite desplegar la interfaz de GAIA de forma gratuita con una URL pública. El uso de sentence-transformers en lugar de Ollama garantiza que el sistema funciona en la nube sin necesidad de servidores adicionales.

### Requisitos previos para el despliegue

- Repositorio público o privado en GitHub
- Cuenta en Streamlit Community Cloud (share.streamlit.io)
- El vectorstore pregenerado debe subirse al repositorio o regenerarse en el despliegue

### Archivo packages.txt

Crear el archivo `packages.txt` en la raíz del repositorio con el siguiente contenido para que Streamlit Cloud instale ffmpeg automáticamente:

```
ffmpeg
```

### Archivo .streamlit/config.toml

El archivo `.streamlit/config.toml` limita el tamaño de archivos subidos a 5 MB:

```toml
[server]
maxUploadSize = 5
```

### Pasos para desplegar

1. Subir el repositorio a GitHub con todos los cambios.
2. Ingresar a https://share.streamlit.io y conectar la cuenta de GitHub.
3. Seleccionar el repositorio dimarior/telecom-chatbot.
4. Configurar el archivo principal: app/streamlit_app.py.
5. En Advanced settings agregar las variables de entorno: MISTRAL_API_KEY y WHISPER_MODEL=small.
6. Hacer clic en Deploy.

### Consideración sobre el despliegue

Para el despliegue en Streamlit Community Cloud, la lógica conversacional se integra directamente en la interfaz Streamlit, eliminando la dependencia de la API FastAPI como servicio externo. LangGraph, ChromaDB y los módulos multimodales corren dentro del mismo proceso de Streamlit.

Los nodos del grafo conversacional fueron convertidos a funciones síncronas para garantizar compatibilidad con Streamlit Cloud. La memoria conversacional se gestiona mediante st.session_state durante la sesión activa del usuario.

El vectorstore pregenerado con sentence-transformers está incluido en el repositorio para que Streamlit Cloud pueda acceder a él sin necesidad de ejecutar el scraping en la nube.

La API FastAPI se mantiene disponible para uso local y para integraciones externas futuras.

---

## Comandos rápidos con Makefile

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

## Evaluación del Sistema RAG

### Evaluación básica

El módulo `src/evaluate.py` ejecuta 15 preguntas distribuidas por operador (5 Claro, 5 Movistar, 3 Tigo, 2 generales) y registra métricas en MLflow.

```bash
python -m src.evaluate
```

### Evaluación avanzada con RAGAS

```bash
python -m src.evaluate_ragas
```

| Métrica | Descripción | Target |
|---|---|---|
| Faithfulness | Fidelidad de la respuesta al contexto recuperado | mayor a 0.85 |
| Answer Relevancy | Relevancia de la respuesta a la pregunta | mayor a 0.80 |
| Context Precision | Proporción de chunks relevantes recuperados | mayor a 0.75 |
| Context Recall | Cobertura de información necesaria | mayor a 0.80 |

---

## Métricas MLflow

| Experimento | Métricas registradas |
|---|---|
| build_vectorstore | total_chunks, vectorstore_size, embedding_time_s |
| evaluacion_rag_gaia | score por pregunta, latencia, avg_score_global, avg_score por operador |
| evaluacion_ragas_gaia | faithfulness, answer_relevancy, context_precision, context_recall, eval_gate |
| rag_pipeline_gaia | trazado automáticamente con @mlflow.trace por cada consulta |

---

## Memoria Conversacional

El sistema utiliza SQLite para persistir el historial de conversaciones entre sesiones. La base de datos se crea automáticamente en `gaia_memory.db` al iniciar el sistema.

El checkpointer de LangGraph también utiliza SQLite para mantener el estado del grafo por sesión, lo que permite continuidad conversacional real entre turnos sin requerir infraestructura adicional.

---

## Consideraciones de Despliegue

- El vectorstore debe regenerarse cada vez que se actualice el contenido scrapeado.
- El archivo `.env` no debe versionarse. Usar `.env.example` como referencia.
- Las carpetas `vectorstore/`, `data/processed/*.txt` y `gaia_memory.db` no se versionan.
- El modelo sentence-transformers se descarga automáticamente la primera vez que se ejecuta el ingest (~420MB).
- Los modelos de faster-whisper y EasyOCR se descargan automáticamente en el primer uso.
- En despliegues Linux (Streamlit Cloud, Railway, Render) ffmpeg viene preinstalado vía packages.txt.
- El tamaño máximo de archivos subidos está limitado a 5 MB tanto en el frontend (config.toml) como en el backend (validación en la API).

---

## Investigación Académica

Este proyecto forma parte de una investigación de trabajo de grado sobre experiencia de usuario en sistemas de inteligencia artificial generativa. Los objetivos de investigación incluyen:

- Evaluar la percepción de empatía en chatbots basados en GenAI UX vs chatbots tradicionales.
- Medir usabilidad mediante la escala SUS (System Usability Scale de Brooke, 1996), instrumento de 10 ítems con score de 0 a 100 donde valores superiores a 68 indican buena usabilidad percibida.
- Analizar satisfacción del usuario mediante NPS (Net Promoter Score).
- Medir eficiencia operativa mediante la Tasa de Resolución en Primer Contacto (TRPC).
- Comparar experiencia conversacional entre un chatbot tradicional y una arquitectura conversacional basada en GenAI UX.
- Estudiar el impacto de principios de Human-Centered AI en la percepción de utilidad, confianza y cercanía del asistente.
- Evaluar la calidad del sistema RAG mediante métricas objetivas: faithfulness, answer relevancy, context precision y context recall (Gao et al., 2024).

Los embeddings del sistema utilizan el modelo paraphrase-multilingual-mpnet-base-v2 de sentence-transformers (Reimers y Gurevych, 2019), seleccionado por su entrenamiento específico en corpus multilingües que incluyen español y su amplio respaldo en la literatura de sistemas RAG.

Los prompts del sistema implementan principios de Conversational UX, UX Writing, empatía operacional y adaptive tone response, alineados con el marco teórico de la investigación.