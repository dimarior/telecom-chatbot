"""
api/main.py
───────────
FastAPI con LangGraph para el chatbot Telecom.
Combina el patrón de lifespan de tq_chatbot con el pipeline de genai-rag-telecom.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": API_TITLE}


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
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
