"""
src/evaluate_ragas.py
──────────────────────
Evaluación RAGAS "manual" (sin la librería `ragas`) usando la API de Mistral
directamente como LLM-juez, más el endpoint de embeddings de Mistral para
answer_relevancy. Reemplaza la evaluación original basada en la librería
`ragas`, que daba problemas de compatibilidad (torch/sentence-transformers,
versiones de datasets, etc.). Este archivo es autosuficiente: no depende de
ningún otro módulo de evaluación (EVAL_RAGAS, THRESHOLDS y eval_gate() están
definidos aquí mismo, más abajo).

La generación de respuesta + contexto NO usa retrieve_documents() de
src/rag_chain.py, porque ese helper recupera contexto con OllamaEmbeddings —
mientras que el vectorstore de producción (VECTORSTORE_DIR) fue regenerado
con HuggingFaceEmbeddings("paraphrase-multilingual-mpnet-base-v2") vía
Chroma (ver api/main.py). Ese mismatch de espacio vectorial hacía que el
retrieval trajera chunks irrelevantes, y por eso faithfulness/
context_precision/context_recall daban 0. Aquí se recupera el contexto
directamente contra Chroma + HuggingFaceEmbeddings, igual que en producción,
sin tocar rag_chain.py.

Métricas reimplementadas con Mistral como juez, siguiendo la misma
metodología que RAGAS:

  faithfulness
    1) Se descompone `answer` en afirmaciones (statements) atómicas.
    2) Para cada afirmación, el LLM verifica si se infiere SOLO del contexto.
    3) score = afirmaciones_soportadas / total_afirmaciones

  answer_relevancy
    1) El LLM genera N preguntas que `answer` estaría respondiendo
       (ingeniería inversa) y marca si la respuesta es evasiva/noncommittal.
    2) Se embeben (Mistral embeddings) la pregunta original y las generadas.
    3) score = similitud coseno promedio (0 si la respuesta es noncommittal)

  context_precision
    1) El LLM marca cada chunk de contexto (en el orden en que fue
       recuperado) como relevante o no, usando la pregunta + ground_truth.
    2) score = precisión promedio ponderada por posición (estilo average
       precision), igual que el context_precision de RAGAS.

  context_recall
    1) Se descompone `ground_truth` en afirmaciones atómicas.
    2) Para cada afirmación, el LLM verifica si es atribuible al contexto
       recuperado.
    3) score = afirmaciones_atribuibles / total_afirmaciones

Requiere el paquete `requests` (normalmente ya viene como dependencia de
otras librerías; si falta: `uv add requests`).

Uso:
    python -m src.evaluate_ragas
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import time
from pathlib import Path

import mlflow
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EVALUATION_DIR,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    RETRIEVER_K,
    VECTORSTORE_DIR,
)
from src.rag_chain import ask

try:
    from src.config import MISTRAL_EMBED_MODEL
except ImportError:  # no definido en config.py -> usar el default de Mistral
    MISTRAL_EMBED_MODEL = "mistral-embed"

# ── Dataset de evaluación con ground_truth (embebido, no depende de ─────────────
# ── ningún otro módulo de evaluación) ───────────────────────────────────────────

EVAL_RAGAS = [
    {
        "question": "¿Cómo reporto una falla técnica de internet con Claro?",
        "ground_truth": (
            "Para reportar una falla técnica de internet con Claro puedes llamar "
            "a la línea 018000910900, usar el chat en claro.com.co o acceder a "
            "Mi Claro para gestionar el reporte en línea."
        ),
        "operador": "claro",
    },
    {
        "question": "¿Cómo hago la portabilidad numérica en Colombia?",
        "ground_truth": (
            "La portabilidad numérica en Colombia permite conservar tu número al "
            "cambiar de operador. Debes solicitar el código de portabilidad a tu "
            "operador actual, luego contactar al nuevo operador para iniciar el proceso. "
            "El trámite tarda máximo 3 días hábiles y es regulado por la CRC."
        ),
        "operador": "general",
    },
    {
        "question": "¿Cómo pago mi factura de Movistar en línea?",
        "ground_truth": (
            "Puedes pagar tu factura de Movistar en línea a través del portal "
            "movistar.com.co en la sección de pagos, mediante la app Mi Movistar, "
            "o por PSE con tu número de línea y referencia de pago."
        ),
        "operador": "movistar",
    },
    {
        "question": "¿Cómo recargo mi línea prepago de Tigo?",
        "ground_truth": (
            "Puedes recargar tu línea prepago Tigo marcando *611, en el sitio "
            "tigo.com.co, en puntos de venta autorizados, o mediante la app Tigo. "
            "También puedes hacer recargas electrónicas en tiendas y droguerías."
        ),
        "operador": "tigo",
    },
    {
        "question": "¿Cuáles son los canales de atención al cliente de los operadores de telecomunicaciones?",
        "ground_truth": (
            "Los operadores en Colombia ofrecen atención por teléfono, chat en línea, "
            "aplicaciones móviles y puntos de atención presencial. Claro atiende por "
            "018000910900, Movistar por *611 y Tigo también por *611 desde su línea."
        ),
        "operador": "general",
    },
]

THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.80,
}


def eval_gate(results: dict) -> bool:
    """
    Verifica que todas las métricas superen los thresholds.
    Retorna True si pasa, False si alguna métrica está por debajo.
    """
    print("\n" + "=" * 65)
    print("  EVAL GATE — VERIFICACIÓN DE THRESHOLDS")
    print("=" * 65)
    all_passed = True
    for metric, threshold in THRESHOLDS.items():
        score = results.get(metric, 0)
        passed = score >= threshold
        estado = "PASS" if passed else "FAIL"
        print(f"  {estado} | {metric:<25} {score:.3f} (threshold: {threshold})")
        if not passed:
            all_passed = False
    return all_passed

# Mismo modelo de embeddings que usa api/main.py para el vectorstore Chroma
# de producción. Debe coincidir con el modelo usado para construir/regenerar
# VECTORSTORE_DIR, o el retrieval trae chunks sin relación con la pregunta.
RETRIEVAL_HF_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

_LOG = logging.getLogger("gaia.evaluate_ragas")

# Activar con: $env:RAGAS_MANUAL_DEBUG="1" (PowerShell) antes de correr el script,
# para imprimir el contexto real enviado al juez y la respuesta cruda de Mistral.
_DEBUG = os.getenv("RAGAS_MANUAL_DEBUG", "0") == "1"


def _debug(label: str, content) -> None:
    if not _DEBUG:
        return
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    print(f"      [DEBUG] {label}: {text[:800]}")

# ── Config de llamadas a la API de Mistral ────────────────────────────────────

MISTRAL_API_BASE = "https://api.mistral.ai/v1"
CHAT_ENDPOINT = f"{MISTRAL_API_BASE}/chat/completions"
EMBED_ENDPOINT = f"{MISTRAL_API_BASE}/embeddings"

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0
INTER_REQUEST_DELAY = 0.5  # segundos entre requests exitosos, para no pegarle al rate limit
ANSWER_RELEVANCY_N_QUESTIONS = 3

_JSON_MODE_SUPPORTED = True  # se desactiva sola si la API rechaza response_format


class _ResponseFormatUnsupported(Exception):
    pass


# ── Cliente HTTP mínimo contra la API de Mistral ──────────────────────────────

def _mistral_request(url: str, payload: dict, max_retries: int = MAX_RETRIES) -> dict:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY no está configurada.")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_exc = exc
            wait = RETRY_BACKOFF_BASE ** attempt
            _LOG.warning(f"    [reintento {attempt}/{max_retries}] error de red: {exc} (esperando {wait:.1f}s)")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            time.sleep(INTER_REQUEST_DELAY)
            return resp.json()

        if resp.status_code == 400 and "response_format" in resp.text:
            raise _ResponseFormatUnsupported(resp.text)

        if resp.status_code in (429, 500, 502, 503, 504):
            wait = RETRY_BACKOFF_BASE ** attempt
            _LOG.warning(
                f"    [reintento {attempt}/{max_retries}] HTTP {resp.status_code} "
                f"(esperando {wait:.1f}s): {resp.text[:200]}"
            )
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(wait)
            continue

        raise RuntimeError(f"Error Mistral API ({resp.status_code}): {resp.text[:500]}")

    raise RuntimeError(f"Fallaron {max_retries} intentos contra Mistral API: {last_exc}")


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No se pudo parsear JSON de la respuesta del modelo:\n{text[:500]}")


def _chat_completion(messages: list[dict], temperature: float = 0.0) -> dict:
    global _JSON_MODE_SUPPORTED

    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if _JSON_MODE_SUPPORTED:
        payload["response_format"] = {"type": "json_object"}

    try:
        data = _mistral_request(CHAT_ENDPOINT, payload)
    except _ResponseFormatUnsupported:
        _LOG.warning("    El modelo no soporta response_format=json_object; desactivando JSON mode.")
        _JSON_MODE_SUPPORTED = False
        payload.pop("response_format", None)
        data = _mistral_request(CHAT_ENDPOINT, payload)

    content = data["choices"][0]["message"]["content"]
    _debug("respuesta cruda del modelo", content)
    return _parse_json(content)


def _embed(texts: list[str]) -> list[list[float]]:
    payload = {"model": MISTRAL_EMBED_MODEL, "input": texts}
    data = _mistral_request(EMBED_ENDPOINT, payload)
    items = sorted(data["data"], key=lambda d: d["index"])
    return [item["embedding"] for item in items]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Bloques compartidos: extracción de afirmaciones ───────────────────────────

def extract_statements(text: str) -> list[str]:
    """Descompone un texto en afirmaciones atómicas verificables."""
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un evaluador experto de sistemas RAG. Descompón el texto dado en "
                "afirmaciones (statements) atómicas, simples y verificables de forma "
                "independiente entre sí. No agregues información que no esté en el texto "
                "original ni la resumas en exceso. Responde ÚNICAMENTE con un JSON válido "
                'con la forma {"statements": ["afirmación 1", "afirmación 2", ...]}.'
            ),
        },
        {"role": "user", "content": f"Texto:\n{text}"},
    ]
    result = _chat_completion(messages, temperature=0.0)
    statements = result.get("statements", [])
    return [s for s in statements if isinstance(s, str) and s.strip()]


_INDEX_KEYS = ("index", "i", "id", "position", "pos", "number", "num", "chunk", "n")
_VERDICT_KEYS = (
    "verdict", "value", "score", "relevant", "is_relevant", "relevance",
    "supported", "is_supported", "attributable", "is_attributable", "answer",
)
_TRUE_TOKENS = {
    "1", "true", "yes", "y", "si", "sí", "verdadero", "relevante", "relevant",
    "supported", "atribuible", "attributable", "soportado",
}
_FALSE_TOKENS = {
    "0", "false", "no", "n", "falso", "irrelevante", "irrelevant",
    "not relevant", "unsupported", "no atribuible", "no soportado",
}


def _coerce_verdict(value) -> int:
    """Normaliza un veredicto (bool/int/float/str en distintos formatos) a 0 o 1."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value >= 1 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_TOKENS:
            return 1
        if v in _FALSE_TOKENS:
            return 0
        try:
            return 1 if float(v) >= 1 else 0
        except ValueError:
            return 0
    return 0


def _verdict_list(verdicts_raw, total: int) -> list[int]:
    """
    Convierte la salida cruda del modelo (potencialmente con variaciones de
    formato: lista de escalares, lista de dicts con distintos nombres de
    llave, o un dict índice->valor) en una lista de 0/1 de longitud `total`,
    en el orden original de los items evaluados.
    """
    if total == 0:
        return []

    # Caso: el modelo devolvió un dict índice->valor en vez de una lista.
    if isinstance(verdicts_raw, dict):
        verdicts_raw = [{"index": k, "verdict": v} for k, v in verdicts_raw.items()]

    if not isinstance(verdicts_raw, list) or not verdicts_raw:
        _LOG.warning(f"    Respuesta de verdicts vacía o con formato inesperado: {verdicts_raw!r}")
        return [0] * total

    # Caso: lista plana de escalares (bool/int/str) -> se asume que el orden
    # coincide con el de los items originales.
    if all(not isinstance(v, dict) for v in verdicts_raw):
        coerced = [_coerce_verdict(v) for v in verdicts_raw]
        if len(coerced) != total:
            _LOG.warning(
                f"    El modelo devolvió {len(coerced)} verdicts pero se esperaban {total}; "
                "se completa/recorta asumiendo el mismo orden."
            )
        if len(coerced) < total:
            coerced += [0] * (total - len(coerced))
        return coerced[:total]

    # Caso general: lista de dicts con nombres de llave variables para el
    # índice y para el veredicto.
    verdict_map: dict[int, int] = {}
    for pos, v in enumerate(verdicts_raw, start=1):
        if not isinstance(v, dict):
            continue

        raw_index = next((v[k] for k in _INDEX_KEYS if k in v), None)
        try:
            resolved_index = int(raw_index) if raw_index is not None else pos
        except (TypeError, ValueError):
            resolved_index = pos

        raw_verdict = next((v[k] for k in _VERDICT_KEYS if k in v), None)
        verdict_map[resolved_index] = _coerce_verdict(raw_verdict)

    if not verdict_map:
        _LOG.warning(f"    No se pudieron interpretar los verdicts del modelo: {verdicts_raw!r}")
        return [0] * total

    return [verdict_map.get(i, 0) for i in range(1, total + 1)]


# ── Faithfulness ───────────────────────────────────────────────────────────────

def judge_faithfulness(statements: list[str], context: str) -> list[int]:
    if not statements:
        return []
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, start=1))
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un evaluador experto encargado de verificar la fidelidad "
                "(faithfulness) de afirmaciones respecto a un contexto. Para cada "
                "afirmación numerada, determina si puede inferirse DIRECTA O "
                "INDIRECTAMENTE únicamente a partir del contexto dado, sin usar "
                "conocimiento externo ni suposiciones. Responde ÚNICAMENTE con un JSON "
                "válido, con EXACTAMENTE esta estructura (un objeto por cada afirmación, "
                "en el mismo orden, index=número de la afirmación, verdict=1 o 0 como "
                "entero, nunca texto ni booleano):\n"
                '{"verdicts": [{"index": 1, "verdict": 1}, {"index": 2, "verdict": 0}]}\n'
                "verdict=1 significa que la afirmación está soportada por el contexto y "
                "verdict=0 que no lo está. No omitas ninguna afirmación."
            ),
        },
        {"role": "user", "content": f"Contexto:\n{context}\n\nAfirmaciones:\n{numbered}"},
    ]
    result = _chat_completion(messages, temperature=0.0)
    return _verdict_list(result.get("verdicts", []), len(statements))


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    statements = extract_statements(answer)
    _debug("statements extraídas de la respuesta", statements)
    if not statements:
        return 1.0
    context_text = "\n\n---\n\n".join(contexts) if contexts else ""
    _debug(f"contexts recibidos: {len(contexts)} chunk(s), {len(context_text)} chars", context_text[:300])
    verdicts = judge_faithfulness(statements, context_text)
    _debug("verdicts faithfulness", verdicts)
    if not verdicts:
        return 0.0
    return sum(verdicts) / len(verdicts)


# ── Answer relevancy ────────────────────────────────────────────────────────────

def compute_answer_relevancy(
    question: str, answer: str, n_questions: int = ANSWER_RELEVANCY_N_QUESTIONS
) -> float:
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un evaluador experto de sistemas RAG. Dada una respuesta, genera "
                f"{n_questions} preguntas distintas que esa respuesta estaría "
                "contestando (ingeniería inversa de la pregunta original, sin verla). "
                "También determina si la respuesta es evasiva, vaga o 'noncommittal' "
                "(no aporta información concreta, p.ej. 'no tengo esa información'). "
                "Responde ÚNICAMENTE con un JSON válido con la forma "
                '{"questions": ["...", "...", "..."], "is_noncommittal": true|false}.'
            ),
        },
        {"role": "user", "content": f"Respuesta:\n{answer}"},
    ]
    result = _chat_completion(messages, temperature=0.3)
    generated_questions = [
        q for q in result.get("questions", []) if isinstance(q, str) and q.strip()
    ]
    is_noncommittal = bool(result.get("is_noncommittal", False))

    if is_noncommittal or not generated_questions:
        return 0.0

    embeddings = _embed([question] + generated_questions)
    question_emb, generated_embs = embeddings[0], embeddings[1:]
    similarities = [_cosine_similarity(question_emb, emb) for emb in generated_embs]
    score = sum(similarities) / len(similarities)
    return max(0.0, min(1.0, score))


# ── Context precision ───────────────────────────────────────────────────────────

def judge_context_relevance(question: str, ground_truth: str, contexts: list[str]) -> list[int]:
    if not contexts:
        return []
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un evaluador experto de sistemas RAG. Determina si cada fragmento "
                "de contexto numerado es relevante y útil para responder la pregunta, "
                "usando la respuesta de referencia (ground truth) como guía de lo que se "
                "necesita para responder correctamente. Responde ÚNICAMENTE con un JSON "
                "válido, con EXACTAMENTE esta estructura (un objeto por cada fragmento, "
                "en el mismo orden, index=número del fragmento, verdict=1 o 0 como "
                "entero, nunca texto ni booleano):\n"
                '{"verdicts": [{"index": 1, "verdict": 1}, {"index": 2, "verdict": 0}]}\n'
                "verdict=1 significa relevante y verdict=0 no relevante. No omitas "
                "ningún fragmento."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Pregunta:\n{question}\n\nRespuesta de referencia:\n{ground_truth}\n\n"
                f"Fragmentos de contexto:\n{numbered}"
            ),
        },
    ]
    result = _chat_completion(messages, temperature=0.0)
    return _verdict_list(result.get("verdicts", []), len(contexts))


def compute_context_precision(question: str, ground_truth: str, contexts: list[str]) -> float:
    _debug(f"contexts recibidos: {len(contexts)} chunk(s)", contexts[0][:300] if contexts else "(vacío)")
    verdicts = judge_context_relevance(question, ground_truth, contexts)
    _debug("verdicts context_precision", verdicts)
    total_relevant = sum(verdicts)
    if total_relevant == 0:
        return 0.0
    running_relevant = 0
    precisions = []
    for k, v in enumerate(verdicts, start=1):
        if v == 1:
            running_relevant += 1
            precisions.append(running_relevant / k)
    return sum(precisions) / total_relevant


# ── Context recall ──────────────────────────────────────────────────────────────

def judge_claim_attribution(claims: list[str], context: str) -> list[int]:
    if not claims:
        return []
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un evaluador experto de sistemas RAG. Para cada afirmación "
                "numerada extraída de una respuesta de referencia (ground truth), "
                "determina si puede atribuirse (encontrarse o inferirse) a partir del "
                "contexto recuperado dado. Responde ÚNICAMENTE con un JSON válido, con "
                "EXACTAMENTE esta estructura (un objeto por cada afirmación, en el "
                "mismo orden, index=número de la afirmación, verdict=1 o 0 como entero, "
                "nunca texto ni booleano):\n"
                '{"verdicts": [{"index": 1, "verdict": 1}, {"index": 2, "verdict": 0}]}\n'
                "verdict=1 significa que la afirmación se puede atribuir al contexto y "
                "verdict=0 que no. No omitas ninguna afirmación."
            ),
        },
        {"role": "user", "content": f"Contexto:\n{context}\n\nAfirmaciones:\n{numbered}"},
    ]
    result = _chat_completion(messages, temperature=0.0)
    return _verdict_list(result.get("verdicts", []), len(claims))


def compute_context_recall(ground_truth: str, contexts: list[str]) -> float:
    claims = extract_statements(ground_truth)
    _debug("claims extraídas del ground_truth", claims)
    if not claims:
        return 1.0
    context_text = "\n\n---\n\n".join(contexts) if contexts else ""
    _debug(f"contexts recibidos: {len(contexts)} chunk(s), {len(context_text)} chars", context_text[:300])
    verdicts = judge_claim_attribution(claims, context_text)
    _debug("verdicts context_recall", verdicts)
    if not verdicts:
        return 0.0
    return sum(verdicts) / len(verdicts)


# ── Retrieval con el mismo pipeline de embeddings que producción ────────────────

_vector_store: Chroma | None = None


def _get_vector_store() -> Chroma:
    """Carga (una sola vez, cacheado en el módulo) el Chroma de VECTORSTORE_DIR
    con HuggingFaceEmbeddings, igual que api/main.py en el lifespan de la API."""
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name=RETRIEVAL_HF_EMBEDDING_MODEL)
        _vector_store = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    return _vector_store


def retrieve_documents_hf(question: str, k: int = RETRIEVER_K) -> list:
    """Recupera documentos contra el vectorstore de producción usando
    HuggingFaceEmbeddings, evitando el mismatch de espacio vectorial que
    tiene retrieve_documents() en src/rag_chain.py (que usa OllamaEmbeddings
    sobre un índice construido con sentence-transformers)."""
    vector_store = _get_vector_store()
    return vector_store.similarity_search(question, k=k)


def build_ragas_dataset_hf() -> list[dict]:
    """
    Genera pregunta + respuesta + contexto + ground_truth para cada item de
    EVAL_RAGAS, recuperando el contexto con retrieve_documents_hf() en vez de
    retrieve_documents() de src.rag_chain (ver nota al inicio del archivo).
    Devuelve una lista de dicts, sin depender del paquete `datasets`.
    """
    rows = []
    print("=" * 65)
    print("  GENERANDO RESPUESTAS Y CONTEXTOS PARA RAGAS (retrieval HF)...")
    print("=" * 65)

    for i, item in enumerate(EVAL_RAGAS, start=1):
        q = item["question"]
        print(f"\n[{i}/{len(EVAL_RAGAS)}] {q[:60]}...")

        t0 = time.time()
        answer = ask(q, session_id=f"ragas_eval_{i}")
        latencia = round(time.time() - t0, 2)
        print(f"  Respuesta generada en {latencia}s")

        docs = retrieve_documents_hf(q, k=RETRIEVER_K)
        context_list = [doc.page_content for doc in docs]
        print(f"  Contexto recuperado: {len(context_list)} chunk(s)")
        if not context_list:
            _LOG.warning(f"  ADVERTENCIA: no se recuperó ningún contexto para la pregunta {i}.")

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": context_list,
            "ground_truth": item["ground_truth"],
        })

    return rows


# ── Orquestación de la evaluación ────────────────────────────────────────────────

def evaluar_ragas_manual() -> dict:
    """Ejecuta la evaluación RAGAS manual (Mistral como juez) y registra en MLflow."""
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY no está configurada. Revisa tu .env / src/config.py."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset = build_ragas_dataset_hf()

    print("\n" + "=" * 65)
    print("  EJECUTANDO EVALUACIÓN RAGAS MANUAL (Mistral como juez)...")
    print("=" * 65)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores: dict[str, list[float]] = {m: [] for m in metric_names}
    detalle_por_pregunta = []

    with mlflow.start_run(run_name="evaluacion_ragas_gaia_manual"):
        mlflow.log_param("eval_framework", "mistral_direct_llm_judge")
        mlflow.log_param("eval_method", "custom_ragas_reimplementation_no_ragas_lib")
        mlflow.log_param("modelo_llm", MISTRAL_MODEL)
        mlflow.log_param("modelo_embeddings_answer_relevancy", MISTRAL_EMBED_MODEL)
        mlflow.log_param("modelo_embeddings_retrieval", RETRIEVAL_HF_EMBEDDING_MODEL)
        mlflow.log_param("total_preguntas", len(EVAL_RAGAS))
        mlflow.log_param("retriever_k", RETRIEVER_K)

        t0 = time.time()
        n = len(dataset)

        for i in range(n):
            row = dataset[i]
            question = row["question"]
            answer = row["answer"]
            contexts = row["contexts"]
            ground_truth = row["ground_truth"]

            print(f"\n[{i + 1}/{n}] {question[:60]}...")

            print("    - faithfulness...")
            faithfulness_score = compute_faithfulness(answer, contexts)

            print("    - answer_relevancy...")
            answer_relevancy_score = compute_answer_relevancy(question, answer)

            print("    - context_precision...")
            context_precision_score = compute_context_precision(question, ground_truth, contexts)

            print("    - context_recall...")
            context_recall_score = compute_context_recall(ground_truth, contexts)

            fila = {
                "faithfulness": round(faithfulness_score, 4),
                "answer_relevancy": round(answer_relevancy_score, 4),
                "context_precision": round(context_precision_score, 4),
                "context_recall": round(context_recall_score, 4),
            }
            print(
                f"      faithfulness={fila['faithfulness']:.3f}  "
                f"answer_relevancy={fila['answer_relevancy']:.3f}  "
                f"context_precision={fila['context_precision']:.3f}  "
                f"context_recall={fila['context_recall']:.3f}"
            )

            scores["faithfulness"].append(faithfulness_score)
            scores["answer_relevancy"].append(answer_relevancy_score)
            scores["context_precision"].append(context_precision_score)
            scores["context_recall"].append(context_recall_score)

            detalle_por_pregunta.append({"question": question, **fila})

        elapsed = round(time.time() - t0, 2)

        results_dict = {m: (sum(v) / len(v) if v else 0.0) for m, v in scores.items()}

        print("\n  RESULTADOS RAGAS (MANUAL):")
        for metric, score in results_dict.items():
            threshold = THRESHOLDS.get(metric, 0)
            estado = "OK" if score >= threshold else "BAJO"
            print(f"  {estado} | {metric:<25} {score:.3f}")
            mlflow.log_metric(f"ragas_{metric}", round(score, 4))

        mlflow.log_metric("ragas_eval_time_s", elapsed)

        passed = eval_gate(results_dict)
        mlflow.log_param("eval_gate_passed", str(passed))

        reporte = {
            "framework": "mistral_direct_llm_judge",
            "modelo": MISTRAL_MODEL,
            "modelo_embeddings_answer_relevancy": MISTRAL_EMBED_MODEL,
            "modelo_embeddings_retrieval": RETRIEVAL_HF_EMBEDDING_MODEL,
            "total_preguntas": len(EVAL_RAGAS),
            "metricas": {k: round(v, 4) for k, v in results_dict.items()},
            "thresholds": THRESHOLDS,
            "eval_gate_passed": passed,
            "tiempo_evaluacion_s": elapsed,
            "detalle_por_pregunta": detalle_por_pregunta,
        }

        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        reporte_path = EVALUATION_DIR / "resultados_ragas_manual.json"
        reporte_path.write_text(
            json.dumps(reporte, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(reporte_path), artifact_path="evaluation")

        print(f"\n  Reporte guardado en: {reporte_path}")
        print(f"  Metricas en MLflow : {MLFLOW_TRACKING_URI}")
        print(f"  Tiempo total       : {elapsed}s")
        print(f"  Eval gate          : {'PASSED' if passed else 'FAILED'}")

    return results_dict


# ── Multi-run: promedio ± desviación entre corridas independientes ─────────────

_MULTIRUN_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def evaluar_ragas_manual_multirun(n_runs: int = 3) -> dict:
    """
    Corre evaluar_ragas_manual() n_runs veces (cada una queda registrada en
    MLflow como una corrida independiente) y agrega mean/std por métrica.
    Recomendado para reportar resultados en la tesis: un juez LLM tiene algo
    de varianza entre corridas, así que un promedio de 2-3 corridas es más
    defendible académicamente que una sola.
    """
    print("\n" + "#" * 65)
    print(f"  EVALUACIÓN MULTI-RUN: {n_runs} corridas independientes")
    print("#" * 65)

    per_run_results = []
    for run_idx in range(1, n_runs + 1):
        print(f"\n>>> CORRIDA {run_idx}/{n_runs} <<<")
        per_run_results.append(evaluar_ragas_manual())

    agregado = {}
    for m in _MULTIRUN_METRICS:
        valores = [r[m] for r in per_run_results]
        media = statistics.mean(valores)
        desviacion = statistics.stdev(valores) if len(valores) > 1 else 0.0
        agregado[m] = {"mean": media, "std": desviacion, "valores": [round(v, 4) for v in valores]}

    print("\n" + "=" * 65)
    print(f"  RESULTADOS AGREGADOS ({n_runs} corridas)")
    print("=" * 65)
    for m in _MULTIRUN_METRICS:
        media = agregado[m]["mean"]
        desviacion = agregado[m]["std"]
        threshold = THRESHOLDS.get(m, 0)
        estado = "OK" if media >= threshold else "BAJO"
        print(f"  {estado} | {m:<25} {media:.3f} ± {desviacion:.3f}  (threshold: {threshold})")

    passed = eval_gate({m: agregado[m]["mean"] for m in _MULTIRUN_METRICS})

    reporte = {
        "framework": "mistral_direct_llm_judge",
        "n_runs": n_runs,
        "modelo": MISTRAL_MODEL,
        "modelo_embeddings_answer_relevancy": MISTRAL_EMBED_MODEL,
        "modelo_embeddings_retrieval": RETRIEVAL_HF_EMBEDDING_MODEL,
        "total_preguntas": len(EVAL_RAGAS),
        "metricas_agregadas": {
            m: {
                "mean": round(agregado[m]["mean"], 4),
                "std": round(agregado[m]["std"], 4),
                "valores_por_corrida": agregado[m]["valores"],
            }
            for m in _MULTIRUN_METRICS
        },
        "thresholds": THRESHOLDS,
        "eval_gate_passed": passed,
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    reporte_path = EVALUATION_DIR / "resultados_ragas_manual_agregado.json"
    reporte_path.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluacion_ragas_gaia_manual_agregado"):
        mlflow.log_param("n_runs", n_runs)
        mlflow.log_param("eval_framework", "mistral_direct_llm_judge_multirun")
        mlflow.log_param("modelo_llm", MISTRAL_MODEL)
        for m in _MULTIRUN_METRICS:
            mlflow.log_metric(f"ragas_{m}_mean", round(agregado[m]["mean"], 4))
            mlflow.log_metric(f"ragas_{m}_std", round(agregado[m]["std"], 4))
        mlflow.log_param("eval_gate_passed", str(passed))
        mlflow.log_artifact(str(reporte_path), artifact_path="evaluation")

    print(f"\n  Reporte agregado guardado en: {reporte_path}")
    print(f"  Run agregado en MLflow: evaluacion_ragas_gaia_manual_agregado")
    print(f"  Eval gate (sobre el promedio): {'PASSED' if passed else 'FAILED'}")

    return agregado


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _n_runs = int(os.getenv("RAGAS_MANUAL_N_RUNS", "1"))
    if _n_runs > 1:
        evaluar_ragas_manual_multirun(_n_runs)
    else:
        evaluar_ragas_manual()