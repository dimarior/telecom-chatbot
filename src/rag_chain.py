"""
src/rag_chain.py
────────────────
Pipeline RAG con MLflow tracing y memoria SQLite.
Versión simplificada (sin grafo) para la API REST y evaluación.

Para respuestas en streaming con LangGraph, usa src/graph/build.py.
"""
from __future__ import annotations

import logging
import time

import mlflow
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    EXPERIMENT_NAME,
    LLM_TEMPERATURE,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    MLFLOW_TRACKING_URI,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_HOST,
    RETRIEVER_K,
    VECTORSTORE_DIR,
)
from src.memory import (
    format_history_for_prompt,
    get_history,
    save_message,
)

_LOG = logging.getLogger("recamier.rag_chain")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

PROMPT_TEMPLATE = [
    ("system", """Eres el asistente virtual oficial de servicio al cliente 
para operadores de telecomunicaciones en Colombia: Claro, Movistar y Tigo.

ROL:
Experto en planes, tarifas, soporte técnico, facturación, portabilidad 
y autogestión de los tres operadores.

TONO:
- Cálido, profesional y cercano
- Siempre en español
- Respuestas concisas y útiles

INSTRUCCIONES:
1. Responde SIEMPRE en español
2. Si el contexto tiene información relevante úsala para responder
3. Si el usuario especifica un operador (Claro, Movistar o Tigo), 
   responde SOLO con información de ese operador
4. Si NO hay información suficiente responde:
   'No tengo esa información disponible. Te invito a contactar 
   directamente al operador a través de su línea de atención 
   o sitio web oficial.'
5. Máximo 4 párrafos, sé conciso y útil

NO DEBES:
- Inventar planes, precios o promociones
- Mezclar información de operadores si preguntan por uno específico
- Responder en inglés"""),

    ("human", """{history}

Contexto de los operadores:
{context}

Pregunta: {question}

Respuesta:"""),
]


@mlflow.trace(name="load_vectorstore")
def load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model=OLLAMA_EMBEDDING_MODEL,
    )
    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )


@mlflow.trace(name="retrieve_documents")
def retrieve_documents(question: str) -> list:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    return retriever.invoke(question)


@mlflow.trace(name="format_documents")
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@mlflow.trace(name="generate_answer")
def generate_answer(question: str, context: str, history: str = "") -> str:
    prompt = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question,
        "history": history,
    })


@mlflow.trace(name="rag_pipeline_recamier")
def ask(question: str, session_id: str = "default") -> str:
    """Función principal del pipeline RAG."""
    saludos = ["me llamo", "mi nombre es", "hola", "buenos", "gracias", "chao", "bye"]
    es_social = any(s in question.lower() for s in saludos)

    history_list = get_history(session_id, limit=6)
    history_text = format_history_for_prompt(history_list)

    if es_social:
        context = f"El usuario dice: {question}"
    else:
        docs = retrieve_documents(question)
        context = format_docs(docs)

    respuesta = generate_answer(question, context, history_text)
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", respuesta)
    return respuesta


if __name__ == "__main__":
    pregunta = "¿Qué productos de Recamier son buenos para el cabello seco?"
    print(f"\nPregunta: {pregunta}")
    print("\nRespuesta:")
    print(ask(pregunta, session_id="test"))
