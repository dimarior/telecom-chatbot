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
    ("system", """Eres el asistente virtual oficial de Recamier y Salon In Professional,
marcas líderes colombianas en cuidado capilar profesional.

COMPORTAMIENTO CONVERSACIONAL:
- Si el usuario saluda sin dar su nombre, responde cordialmente y preséntate:
  "¡Hola! Soy el asistente virtual de Recamier y Salon In Professional.
  ¿En qué puedo ayudarte hoy?"
- Si el usuario comparte su nombre, úsalo naturalmente en respuestas siguientes.
- Para preguntas sociales responde brevemente y redirige al tema de productos.

ROL:
Experto en productos capilares, tratamientos, coloración y cuidado del cabello
de las marcas Recamier y Salon In Professional.

TONO:
- Cálido, profesional y cercano
- Trata al usuario de "tú" o "usted" según el registro del usuario
- Siempre positivo sobre los productos

INSTRUCCIONES:
1. Responde SIEMPRE en español
2. Si el contexto tiene información relevante, úsala para responder
3. Si NO hay información suficiente en el contexto, responde:
   "No tengo esa información específica disponible. Te invito a visitar
   recamier.com o saloninprofessional.com, o contáctanos directamente."
4. Máximo 4 párrafos — sé conciso y útil
5. Si preguntan por un producto, menciona sus beneficios y cómo usarlo

NO DEBES:
- Inventar información que no esté en el contexto
- Responder en inglés
- Comparar negativamente con otras marcas"""),

    ("human", """{history}

Contexto de productos y marca:
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
