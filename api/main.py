"""
api/main.py
───────────
FastAPI con LangGraph y soporte multimodal para GAIA Telecom.

Endpoints:
  GET  /health           → Estado del sistema
  POST /ask              → Pipeline RAG simple (texto)
  POST /ask/graph        → Pipeline LangGraph completo (texto)
  POST /ask/audio        → Pipeline con entrada de voz (Whisper)
  POST /ask/image        → Pipeline con entrada de imagen (EasyOCR)
  POST /ask/document     → Pipeline con entrada de PDF (PyMuPDF)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from src.config import (
    API_HOST,
    API_PORT,
    API_TITLE,
    DB_PATH,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_HOST,
    RETRIEVER_K,
    VECTORSTORE_DIR,
)
from src.graph import build_graph
from src.rag_chain import ask as ask_simple


def _configure_langsmith() -> None:
    if not (LANGSMITH_TRACING and LANGSMITH_API_KEY):
        os.environ.pop("LANGSMITH_TRACING", None)
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_langsmith()

    embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=OLLAMA_EMBEDDING_MODEL)
    app.state.vector_store = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    checkpoint_conn = await aiosqlite.connect(str(DB_PATH))
    checkpointer = AsyncSqliteSaver(checkpoint_conn)
    await checkpointer.setup()
    app.state.checkpoint_conn = checkpoint_conn
    app.state.checkpointer = checkpointer

    app.state.graph = build_graph(
        vector_store=app.state.vector_store,
        checkpointer=checkpointer,
        top_k=RETRIEVER_K,
    )

    try:
        yield
    finally:
        await checkpoint_conn.close()


def create_app() -> FastAPI:
    app = FastAPI(title=API_TITLE, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    pregunta: str
    session_id: str = "default"


class AskResponse(BaseModel):
    respuesta: str
    session_id: str
    latencia_segundos: float
    sources: list[dict] = []
    modalidad: str = "texto"


class MultimodalResponse(BaseModel):
    respuesta: str
    session_id: str
    latencia_segundos: float
    sources: list[dict] = []
    modalidad: str
    texto_extraido: str = ""
    idioma: str = ""
    paginas: int = 0


# ── Endpoints de texto ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": API_TITLE,
        "multimodal": {
            "audio": "whisper-base-cpu",
            "imagen": "easyocr-es-en",
            "documento": "pymupdf",
        }
    }


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest):
    """Pipeline RAG simple (sin streaming) — compatible con Streamlit."""
    t0 = time.time()
    respuesta = ask_simple(body.pregunta, session_id=body.session_id)
    latencia = round(time.time() - t0, 3)
    return AskResponse(
        respuesta=respuesta,
        session_id=body.session_id,
        latencia_segundos=latencia,
        modalidad="texto",
    )


@app.post("/ask/graph", response_model=AskResponse)
async def ask_graph_endpoint(body: AskRequest):
    """Pipeline con LangGraph (router + memoria checkpointer)."""
    from langchain_core.messages import AIMessage

    t0 = time.time()
    config = {"configurable": {"thread_id": body.session_id}}
    result = await app.state.graph.ainvoke(
        {"question": body.pregunta},
        config=config,
    )

    messages = result.get("messages", [])
    respuesta = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            respuesta = msg.content
            break

    sources = result.get("sources", [])
    latencia = round(time.time() - t0, 3)
    return AskResponse(
        respuesta=respuesta,
        session_id=body.session_id,
        latencia_segundos=latencia,
        sources=sources,
        modalidad="texto",
    )


# ── Endpoint de audio ─────────────────────────────────────────────────────────

@app.post("/ask/audio", response_model=MultimodalResponse)
async def ask_audio_endpoint(
    audio: UploadFile = File(..., description="Archivo de audio (wav, mp3, ogg, m4a)"),
    session_id: str = Form(default="default"),
):
    """
    Pipeline con entrada de voz.
    Transcribe el audio con Whisper y procesa con LangGraph.
    """
    from langchain_core.messages import AIMessage
    from src.multimodal.audio import transcribe_audio

    t0 = time.time()
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Archivo de audio vacío.")

    transcripcion = transcribe_audio(audio_bytes, filename=audio.filename or "audio.wav")

    if not transcripcion["success"]:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo transcribir el audio: {transcripcion['error']}"
        )

    texto = transcripcion["text"]
    idioma = transcripcion.get("language", "es")

    if not texto.strip():
        raise HTTPException(
            status_code=422,
            detail="El audio no contiene texto reconocible."
        )

    config = {"configurable": {"thread_id": session_id}}
    result = await app.state.graph.ainvoke(
        {"question": f"[Voz transcrita] {texto}"},
        config=config,
    )

    messages = result.get("messages", [])
    respuesta = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            respuesta = msg.content
            break

    latencia = round(time.time() - t0, 3)
    return MultimodalResponse(
        respuesta=respuesta,
        session_id=session_id,
        latencia_segundos=latencia,
        sources=result.get("sources", []),
        modalidad="audio",
        texto_extraido=texto,
        idioma=idioma,
    )


# ── Endpoint de imagen ────────────────────────────────────────────────────────

@app.post("/ask/image", response_model=MultimodalResponse)
async def ask_image_endpoint(
    imagen: UploadFile = File(..., description="Imagen (jpg, png, bmp, webp)"),
    session_id: str = Form(default="default"),
    pregunta_adicional: str = Form(default=""),
):
    """
    Pipeline con entrada de imagen.
    Extrae texto con EasyOCR y procesa con LangGraph.
    """
    from langchain_core.messages import AIMessage
    from src.multimodal.image import extract_text_from_image, describe_image_context

    t0 = time.time()
    imagen_bytes = await imagen.read()

    if not imagen_bytes:
        raise HTTPException(status_code=400, detail="Archivo de imagen vacío.")

    ocr_result = extract_text_from_image(imagen_bytes, filename=imagen.filename or "image.jpg")
    texto_ocr = ocr_result["text"] if ocr_result["success"] else ""
    contexto_imagen = describe_image_context(texto_ocr, filename=imagen.filename or "")

    if pregunta_adicional.strip():
        pregunta_final = f"{contexto_imagen}\n\nPregunta del usuario: {pregunta_adicional}"
    else:
        pregunta_final = contexto_imagen

    config = {"configurable": {"thread_id": session_id}}
    result = await app.state.graph.ainvoke(
        {"question": pregunta_final},
        config=config,
    )

    messages = result.get("messages", [])
    respuesta = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            respuesta = msg.content
            break

    latencia = round(time.time() - t0, 3)
    return MultimodalResponse(
        respuesta=respuesta,
        session_id=session_id,
        latencia_segundos=latencia,
        sources=result.get("sources", []),
        modalidad="imagen",
        texto_extraido=texto_ocr,
    )


# ── Endpoint de documento PDF ─────────────────────────────────────────────────

@app.post("/ask/document", response_model=MultimodalResponse)
async def ask_document_endpoint(
    documento: UploadFile = File(..., description="Documento PDF (factura, contrato, comprobante)"),
    session_id: str = Form(default="default"),
    pregunta_adicional: str = Form(default=""),
):
    """
    Pipeline con entrada de documento PDF.
    Extrae texto con PyMuPDF y procesa con LangGraph.
    """
    from langchain_core.messages import AIMessage
    from src.multimodal.document import extract_text_from_pdf, describe_document_context

    t0 = time.time()
    pdf_bytes = await documento.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Archivo PDF vacío.")

    pdf_result = extract_text_from_pdf(pdf_bytes, filename=documento.filename or "document.pdf")

    if not pdf_result["success"]:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo procesar el PDF: {pdf_result['error']}"
        )

    texto_pdf = pdf_result["text"]
    paginas = pdf_result["pages"]
    contexto_doc = describe_document_context(texto_pdf, filename=documento.filename or "")

    if pregunta_adicional.strip():
        pregunta_final = f"{contexto_doc}\n\nPregunta del usuario: {pregunta_adicional}"
    else:
        pregunta_final = contexto_doc

    config = {"configurable": {"thread_id": session_id}}
    result = await app.state.graph.ainvoke(
        {"question": pregunta_final},
        config=config,
    )

    messages = result.get("messages", [])
    respuesta = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            respuesta = msg.content
            break

    latencia = round(time.time() - t0, 3)
    return MultimodalResponse(
        respuesta=respuesta,
        session_id=session_id,
        latencia_segundos=latencia,
        sources=result.get("sources", []),
        modalidad="documento",
        texto_extraido=texto_pdf[:500],
        paginas=paginas,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
