"""
src/evaluate.py
───────────────
Evaluación del sistema RAG con métricas registradas en MLflow.
Conjunto de preguntas basadas en productos Recamier y Salon In Professional.
"""
from __future__ import annotations

import json
import time
import logging

import mlflow

from src.rag_chain import ask
from src.config import EVALUATION_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME

_LOG = logging.getLogger("recamier.evaluate")

EVAL_SET = [
    {
        "pregunta": "¿Qué productos de Recamier son recomendados para cabello teñido?",
        "keywords": ["color", "teñido", "tratamiento", "protección", "cabello"],
        "descripcion": "Consulta sobre cuidado del color",
    },
    {
        "pregunta": "¿Cómo puedo hidratar mi cabello con productos Salon In Professional?",
        "keywords": ["hidratación", "humectación", "mascarilla", "tratamiento", "seco"],
        "descripcion": "Consulta sobre hidratación capilar",
    },
    {
        "pregunta": "¿Qué hace la línea de keratina de Recamier?",
        "keywords": ["keratina", "alisado", "frizz", "brillo", "suavidad"],
        "descripcion": "Consulta sobre tratamiento de keratina",
    },
    {
        "pregunta": "¿Cuáles son los productos para el crecimiento del cabello?",
        "keywords": ["crecimiento", "caída", "fortalecimiento", "biotina", "raíz"],
        "descripcion": "Consulta sobre anticaída y crecimiento",
    },
    {
        "pregunta": "¿Qué es Salon In Professional y en qué se diferencia de Recamier?",
        "keywords": ["profesional", "salón", "peluquería", "marca", "diferencia"],
        "descripcion": "Consulta sobre diferenciación de marcas",
    },
]


def calcular_score(respuesta: str, keywords: list[str]) -> float:
    hits = sum(1 for kw in keywords if kw.lower() in respuesta.lower())
    return hits / len(keywords)


def evaluar() -> list[dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    resultados: list[dict] = []
    scores_totales: list[float] = []

    print("=" * 60)
    print("📊 EVALUACIÓN RAG — RECAMIER CHATBOT")
    print("=" * 60)

    with mlflow.start_run(run_name="evaluacion_rag_recamier"):
        mlflow.log_param("total_preguntas_eval", len(EVAL_SET))
        mlflow.log_param("modelo_llm", "mistral-small-latest")

        for i, item in enumerate(EVAL_SET, start=1):
            print(f"\n🔍 Q{i}: {item['pregunta']}")
            t0 = time.time()
            respuesta = ask(item["pregunta"], session_id=f"eval_{i}")
            latencia = round(time.time() - t0, 3)

            score = calcular_score(respuesta, item["keywords"])
            scores_totales.append(score)

            mlflow.log_metric(f"score_q{i}", score)
            mlflow.log_metric(f"latencia_q{i}_seg", latencia)

            print(f"   ✅ Score: {score:.2f} | Latencia: {latencia}s")
            print(f"   💬 Preview: {respuesta[:120]}...")

            resultados.append({
                "pregunta_num": i,
                "pregunta": item["pregunta"],
                "descripcion": item["descripcion"],
                "keywords_buscadas": item["keywords"],
                "respuesta_completa": respuesta,
                "score": score,
                "latencia_segundos": latencia,
            })

        avg_score = sum(scores_totales) / len(scores_totales)
        mlflow.log_metric("avg_score_global", avg_score)

        print("\n" + "=" * 60)
        print(f"📈 Score promedio global: {avg_score:.2f} / 1.00")
        print("=" * 60)

        reporte_path = EVALUATION_DIR / "resultados_evaluacion.json"
        reporte_path.write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(reporte_path), artifact_path="evaluation")
        print(f"\n✅ Reporte guardado en: {reporte_path}")
        print("✅ Métricas en MLflow → http://127.0.0.1:5000")

    return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluar()
