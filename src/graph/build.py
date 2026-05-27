"""
src/graph/build.py
──────────────────
Ensambla el StateGraph de Recamier Chatbot y lo compila con checkpointer SQLite.
Patrón tomado de tq_chatbot.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    make_classify_node,
    make_direct_node,
    make_generate_node,
    make_product_node,
    make_retrieve_node,
    route_branch,
)
from src.graph.state import ChatState


def build_graph(
    vector_store: Chroma,
    checkpointer: BaseCheckpointSaver,
    top_k: int = 4,
):
    builder = StateGraph(ChatState)

    builder.add_node("classify", make_classify_node(vector_store))
    builder.add_node("direct", make_direct_node())
    builder.add_node("product", make_product_node())
    builder.add_node("retrieve", make_retrieve_node(vector_store, top_k=top_k))
    builder.add_node("generate", make_generate_node())

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_branch,
        {"direct": "direct", "product": "product", "rag": "retrieve"},
    )
    builder.add_edge("direct", "generate")
    builder.add_edge("product", "generate")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)
