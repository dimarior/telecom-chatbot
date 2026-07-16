"""
src/graph/state.py
──────────────────
Estado del grafo LangGraph y modelo Pydantic para el router.
Inspirado en tq_chatbot con adaptaciones para Recamier.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Route = Literal["direct", "product", "rag"]


class RouteDecision(BaseModel):
    """Salida estructurada del clasificador del router."""

    route: Route = Field(
        ...,
        description=(
            "'direct' para saludos, despedidas, agradecimientos, conversación social, "
            "expresiones emocionales, frustraciones generales o referencias a turnos anteriores. "
            "'product' para consultas sobre planes, precios, tarifas, portabilidad numérica, "
            "puntos de atención, activación o cancelación de servicios de telecomunicaciones. "
            "'rag' para soporte técnico, facturación, recargas, autogestión, cobertura, "
            "velocidades, preguntas frecuentes y procedimientos de Claro, Movistar o Tigo."
        ),
    )


class ChatState(TypedDict, total=False):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    route: Route
    sources: list[dict]
    context: str
    temperature: float
    top_k: int
