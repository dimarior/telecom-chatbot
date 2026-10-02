# GAIA — Kit de Implementación

Guía técnica, biblioteca de prompts y checklist normativo para replicar o adaptar el asistente conversacional GAIA en el sector de telecomunicaciones colombiano.

---

## 1. Guía Técnica de Implementación

### Requisitos previos

| Componente | Versión mínima |
|---|---|
| Python | 3.11 |
| uv (gestor de dependencias) | 0.4+ |
| Cuenta Mistral AI (Free Tier) | — |
| Git | 2.x |

### Variables de entorno

Crea un archivo `.env` en la raíz del proyecto a partir de `.env.example`:

```env
MISTRAL_API_KEY=your_api_key_here
MISTRAL_MODEL=mistral-small-latest
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
RETRIEVER_K=4
MIN_SCORE=0.0
MAX_CONTEXT_CHARS=3000
LLM_TEMPERATURE=0.3
```

### Instalación y despliegue local

```bash
# 1. Clonar el repositorio
git clone https://github.com/dimarior/telecom-chatbot.git
cd telecom-chatbot

# 2. Instalar dependencias
uv pip install -r requirements.txt

# 3. Construir la base de conocimiento (scraping + vectorización)
python scripts/build_knowledge_base.py

# 4. Lanzar la aplicación
streamlit run app/streamlit_app.py
```

### Despliegue en Streamlit Community Cloud

1. Hacer fork del repositorio en tu cuenta de GitHub.
2. Ir a [share.streamlit.io](https://share.streamlit.io) y conectar el repositorio.
3. Configurar las variables de entorno en **Settings → Secrets** usando el mismo formato del `.env`.
4. El archivo principal es `app/streamlit_app.py`.

### Arquitectura de carpetas

```
telecom-chatbot/
├── app/                  # Interfaz Streamlit
├── src/
│   ├── graph/            # Nodos LangGraph (classify, direct, product, retrieve, generate)
│   ├── multimodal/       # Procesamiento de voz (Whisper), imagen y PDF
│   └── config.py         # Variables de configuración centralizadas
├── data/
│   └── knowledge_base/   # Documentos vectorizados en ChromaDB
├── scripts/              # Scripts de construcción y evaluación
├── reports/analysis/     # Scripts de análisis de encuestas
└── requirements.txt
```

### Adaptar a otro sector

Para reemplazar el dominio de telecomunicaciones por otro:

1. **Base de conocimiento:** reemplazar las fuentes en `scripts/build_knowledge_base.py` con las URLs o documentos del nuevo dominio.
2. **Catálogo de productos:** actualizar `src/product_catalog.json` con la información estructurada del nuevo contexto.
3. **Prompts del sistema:** editar `src/graph/nodes.py` — ver Sección 2 de este kit.
4. **Router de intenciones:** revisar las categorías en `ROUTER_SYSTEM` para que correspondan al nuevo dominio.

---

## 2. Biblioteca de Prompts Empáticos

Los cuatro prompts del sistema están definidos en `src/graph/nodes.py`. A continuación se documenta el propósito de cada uno y los principios de diseño aplicados.

### ROUTER_SYSTEM — Clasificador de intenciones

**Propósito:** enrutar cada mensaje del usuario a una de tres rutas: `direct` (respuesta empática sin RAG), `product` (catálogo estructurado) o `rag` (recuperación semántica).

**Principio clave:** el router no solo detecta intención funcional, también detecta tono emocional. Si hay frustración o urgencia sin pregunta técnica clara, enruta a `direct` para contención emocional antes de resolver.

**Señales emocionales detectadas:**
- Frustración: *"llevo horas", "no funciona", "otra vez"*
- Urgencia: *"necesito urgente", "ahora mismo"*
- Confusión: *"no entiendo", "explícame"*
- Agradecimiento: *"gracias", "me ayudaste"*

---

### DIRECT_SYSTEM — Respuestas empáticas y conversacionales

**Propósito:** manejar saludos, quejas generales, expresiones emocionales y temas fuera del dominio técnico.

**Principios de diseño aplicados:**

- **Empatía basada en hechos:** prohibido usar frases genéricas de call center como *"Entiendo tu frustración"* o *"Lamento los inconvenientes"*. En su lugar, nombrar el problema concreto: *"Llevar dos días sin señal es agotador"*.
- **Escala de calibración emocional:** el tono varía según el nivel de malestar detectado, de molestia leve hasta urgencia crítica.
- **Variedad:** ninguna frase empática se repite dentro de la misma conversación.
- **Presentación única:** GAIA se presenta solo en el primer saludo; no repite su nombre en mensajes posteriores.

**Para adaptar este prompt a otro dominio:** mantener la escala de calibración emocional y la prohibición de frases genéricas. Reemplazar las referencias a telecomunicaciones por el dominio objetivo.

---

### PRODUCT_SYSTEM — Consultas comerciales y de catálogo

**Propósito:** responder preguntas sobre planes, precios, portabilidad y contacto usando el catálogo estructurado en `src/product_catalog.json`.

**Principios de diseño aplicados:**

- Tono comercialmente claro pero cercano, evitando lenguaje de folleto publicitario.
- Uso de frases orientadoras: *"Lo que te recomendaría es..."*, *"Una buena opción sería..."*.
- Números de marcación corta siempre con asterisco: `*611`, `*123`.
- Respuesta máxima: 4 oraciones.

---

### RAG_SYSTEM — Soporte técnico con recuperación semántica

**Propósito:** responder preguntas de soporte técnico, facturación y autogestión usando el contexto recuperado de la base de conocimiento vectorizada.

**Principios de diseño aplicados:**

- **Grounding estricto:** responde únicamente con base en el contexto recuperado entre etiquetas `<contexto>`. No incorpora conocimiento paramétrico del LLM.
- **Integración natural:** traduce los documentos técnicos a lenguaje conversacional, sin transmitir la información de forma robótica.
- **Operador activo:** si la pregunta incluye un prefijo `[Claro]`, `[Movistar]` o `[Tigo]`, ese prefijo tiene prioridad absoluta y la respuesta se limita a ese operador.
- **Fallback humanizado:** cuando la búsqueda semántica no retorna resultados, en lugar de responder con un error técnico orienta al usuario al canal oficial del operador correspondiente. Canales verificados:
  - Movistar: `*611` desde celular o chat en movistar.com.co
  - Claro: `*611` desde celular o chat en claro.com.co
  - Tigo: `*611` desde celular o chat en tigo.com.co
- **Números 018000:** son líneas gratuitas nacionales y no llevan asterisco.
- Respuesta máxima: 6 oraciones o lista de pasos cuando el problema lo requiere.

---

## 3. Checklist Normativo — Ley 1581 de 2012

Lista de verificación para el cumplimiento de la Ley Estatutaria 1581 de 2012 (protección de datos personales) en implementaciones de chatbots conversacionales en Colombia.

### Recolección de datos

- [ ] El sistema no solicita ni almacena datos personales identificables del usuario (nombre completo, número de cédula, dirección, número de cuenta).
- [ ] Si se recopila algún dato (nombre de pila para personalización), se informa al usuario en el primer mensaje.
- [ ] Existe un aviso de privacidad accesible desde la interfaz que describe qué datos se procesan y con qué finalidad.

### Almacenamiento y retención

- [ ] El historial de conversaciones no se almacena de forma persistente asociado a un usuario identificable sin su consentimiento explícito.
- [ ] Si se usa memoria conversacional (como en GAIA), esta es de sesión: se elimina al cerrar la conversación.
- [ ] No se comparten datos de la conversación con terceros distintos al proveedor del LLM (Mistral AI en este caso), cuyas condiciones de uso deben revisarse.

### Transparencia

- [ ] El asistente se identifica como sistema automatizado (no como humano) al inicio de la conversación.
- [ ] El usuario puede solicitar en cualquier momento que se elimine su historial de sesión.
- [ ] Las respuestas que involucren datos de facturación o contratos incluyen una recomendación de verificación en el canal oficial del operador.

### Seguridad técnica

- [ ] Las credenciales de API (Mistral AI, ChromaDB) se gestionan mediante variables de entorno, nunca hardcodeadas en el código.
- [ ] El repositorio público no contiene archivos `.env` con claves reales (verificar `.gitignore`).
- [ ] El despliegue en Streamlit Community Cloud usa el gestor de Secrets para las variables sensibles.

### Cumplimiento frente a menores de edad

- [ ] El sistema no está diseñado para recopilar datos de menores de 14 años.
- [ ] No existe mecanismo de verificación de edad, por lo que el aviso de privacidad debe indicar que el servicio está dirigido a personas mayores de 18 años.

---

*Repositorio público:* https://github.com/dimarior/telecom-chatbot  
*Prototipo desplegado:* https://gaia-chatbot-telecom.streamlit.app