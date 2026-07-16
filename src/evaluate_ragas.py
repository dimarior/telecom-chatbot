"""
src/evaluate_ragas.py
─────────────────────
Evaluación avanzada del sistema RAG con métricas RAGAS.
Mide faithfulness, answer_relevancy, context_precision y context_recall.

Métricas y targets de producción:
  - faithfulness       > 0.85  ¿La respuesta es fiel al contexto recuperado?
  - answer_relevancy   > 0.80  ¿La respuesta es relevante a la pregunta?
  - context_precision  > 0.75  ¿Los chunks recuperados son relevantes?
  - context_recall     > 0.80  ¿Se encontró toda la info necesaria?

Uso:
    python -m src.evaluate_ragas
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlflow
from datasets import Dataset

from src.config import (
    EVALUATION_DIR,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_HOST,
    RETRIEVER_K,
    VECTORSTORE_DIR,
)
from src.rag_chain import ask, retrieve_documents

_LOG = logging.getLogger("gaia.evaluate_ragas")

# ── Dataset de evaluación con ground_truth ────────────────────────────────────

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

# ── Thresholds de calidad ─────────────────────────────────────────────────────

THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.80,
}


def build_ragas_dataset() -> Dataset:
    """Construye el dataset de evaluación con preguntas, respuestas y contextos."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("=" * 65)
    print("  GENERANDO RESPUESTAS Y CONTEXTOS PARA RAGAS...")
    print("=" * 65)

    for i, item in enumerate(EVAL_RAGAS, start=1):
        q = item["question"]
        print(f"\n[{i}/{len(EVAL_RAGAS)}] {q[:60]}...")

        # Obtener respuesta del sistema RAG
        t0 = time.time()
        answer = ask(q, session_id=f"ragas_eval_{i}")
        latencia = round(time.time() - t0, 2)
        print(f"  Respuesta generada en {latencia}s")

        # Obtener contextos recuperados
        docs = retrieve_documents(q)
        context_list = [doc.page_content for doc in docs[:RETRIEVER_K]]

        questions.append(q)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


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


def evaluar_ragas() -> dict:
    """Ejecuta la evaluación RAGAS completa y registra en MLflow."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        _LOG.error("RAGAS no está instalado. Ejecuta: pip install ragas datasets")
        raise

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset = build_ragas_dataset()

    print("\n" + "=" * 65)
    print("  EJECUTANDO EVALUACIÓN RAGAS...")
    print("=" * 65)

    with mlflow.start_run(run_name="evaluacion_ragas_gaia"):
        mlflow.log_param("eval_framework", "ragas")
        mlflow.log_param("ragas_version", "0.4.3")
        mlflow.log_param("modelo_llm", MISTRAL_MODEL)
        mlflow.log_param("embedding_model", OLLAMA_EMBEDDING_MODEL)
        mlflow.log_param("total_preguntas", len(EVAL_RAGAS))
        mlflow.log_param("retriever_k", RETRIEVER_K)

        t0 = time.time()
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        elapsed = round(time.time() - t0, 2)

        results_dict = results.to_pandas().mean().to_dict()

        print("\n  RESULTADOS RAGAS:")
        for metric, score in results_dict.items():
            threshold = THRESHOLDS.get(metric, 0)
            estado = "OK" if score >= threshold else "BAJO"
            print(f"  {estado} | {metric:<25} {score:.3f}")
            mlflow.log_metric(f"ragas_{metric}", round(score, 4))

        mlflow.log_metric("ragas_eval_time_s", elapsed)

        passed = eval_gate(results_dict)
        mlflow.log_param("eval_gate_passed", str(passed))

        # Guardar reporte
        reporte = {
            "framework": "ragas",
            "modelo": MISTRAL_MODEL,
            "total_preguntas": len(EVAL_RAGAS),
            "metricas": {k: round(v, 4) for k, v in results_dict.items()},
            "thresholds": THRESHOLDS,
            "eval_gate_passed": passed,
            "tiempo_evaluacion_s": elapsed,
        }

        reporte_path = EVALUATION_DIR / "resultados_ragas.json"
        reporte_path.write_text(
            json.dumps(reporte, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(reporte_path), artifact_path="evaluation")

        print(f"\n  Reporte guardado en: {reporte_path}")
        print(f"  Metricas en MLflow : http://127.0.0.1:5002")
        print(f"  Tiempo total       : {elapsed}s")
        print(f"  Eval gate          : {'PASSED' if passed else 'FAILED'}")

    return results_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluar_ragas()
