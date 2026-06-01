"""
src/memory.py
─────────────
Memoria persistente con SQLite para el chatbot Telecom.
Guarda historial de conversaciones entre sesiones.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from src.config import DB_PATH


def init_db() -> None:
    """Crea las tablas si no existen."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session
        ON conversations(session_id, created_at)
    """)
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM conversations "
        "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def get_all_sessions() -> list[str]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT session_id FROM conversations ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def clear_session(session_id: str) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def format_history_for_prompt(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["Historial de conversación previa:"]
    for msg in history:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        lines.append(f"{role}: {msg['content'][:300]}")
    return "\n".join(lines)


# Inicializar al importar
init_db()
