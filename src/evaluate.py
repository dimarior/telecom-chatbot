"""
src/evaluate.py
───────────────
Evaluación del sistema RAG con métricas registradas en MLflow.
Conjunto de preguntas basadas en servicio al cliente de Claro, Movistar y Tigo.
"""
from __future__ import annotations

import json
import time
import logging

import mlflow

from src.rag_chain import ask
from src.config import EVALUATION_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME

_LOG = logging.getLogger("gaia.evaluate")

EVAL_SET = [
    # ── Claro ──────────────────────────────────────────────────────────────
    {
        "pregunta": "¿Cómo reporto una falla técnica de internet con Claro?",
        "keywords": ["claro", "falla", "técnica", "soporte", "reportar"],
        "operador": "claro",
        "tipo": "soporte_tecnico",
    },
    {
        "pregunta": "¿Cuáles son los planes de internet hogar disponibles en Claro?",
        "keywords": ["claro", "plan", "internet", "hogar", "fibra"],
        "operador": "claro",
        "tipo": "planes",
    },
    {
        "pregunta": "¿Cómo me registro en Mi Claro para gestionar mi cuenta?",
        "keywords": ["mi claro", "registro", "cuenta", "autogestion", "portal"],
        "operador": "claro",
        "tipo": "autogestion",
    },
    {
        "pregunta": "¿Cómo activo el roaming internacional con Claro?",
        "keywords": ["roaming", "internacional", "activar", "claro", "viaje"],
        "operador": "claro",
        "tipo": "servicios",
    },
    {
        "pregunta": "¿Dónde puedo pagar mi factura de Claro en línea?",
        "keywords": ["pago", "factura", "claro", "línea", "pagar"],
        "operador": "claro",
        "tipo": "facturacion",
    },
    # ── Movistar ────────────────────────────────────────────────────────────
    {
        "pregunta": "¿Cómo solicito asistencia técnica para mi internet de Movistar?",
        "keywords": ["movistar", "asistencia", "técnica", "internet", "soporte"],
        "operador": "movistar",
        "tipo": "soporte_tecnico",
    },
    {
        "pregunta": "¿Qué canales de televisión incluye el plan de Movistar TV?",
        "keywords": ["movistar", "tv", "televisión", "canales", "plan"],
        "operador": "movistar",
        "tipo": "planes",
    },
    {
        "pregunta": "¿Cómo accedo al portal de autogestión de Movistar?",
        "keywords": ["movistar", "autogestion", "portal", "mi movistar", "cuenta"],
        "operador": "movistar",
        "tipo": "autogestion",
    },
    {
        "pregunta": "¿Cómo activo el buzón de voz en Movistar?",
        "keywords": ["buzon", "voz", "movistar", "activar", "mensaje"],
        "operador": "movistar",
        "tipo": "servicios",
    },
    {
        "pregunta": "¿Cómo pago mi factura de Movistar en línea?",
        "keywords": ["pago", "factura", "movistar", "línea", "pagar"],
        "operador": "movistar",
        "tipo": "facturacion",
    },
    # ── Tigo ────────────────────────────────────────────────────────────────
    {
        "pregunta": "¿Cómo recargo mi línea prepago de Tigo?",
        "keywords": ["tigo", "recarga", "prepago", "línea", "saldo"],
        "operador": "tigo",
        "tipo": "recargas",
    },
    {
        "pregunta": "¿Cuáles son los planes prepago disponibles en Tigo?",
        "keywords": ["tigo", "prepago", "plan", "paquete", "datos"],
        "operador": "tigo",
        "tipo": "planes",
    },
    {
        "pregunta": "¿Cómo contacto al soporte técnico de Tigo?",
        "keywords": ["tigo", "soporte", "contacto", "atención", "ayuda"],
        "operador": "tigo",
        "tipo": "soporte_tecnico",
    },
    # ── Generales ────────────────────────────────────────────────────────────
    {
        "pregunta": "¿Cómo hago la portabilidad numérica entre operadores en Colombia?",
        "keywords": ["portabilidad", "numero", "operador", "cambio", "CRC"],
        "operador": "general",
        "tipo": "portabilidad",
    },
    {
        "pregunta": "¿Cuáles son los canales de atención al cliente disponibles en los operadores?",
        "keywords": ["atención", "cliente", "canal", "chat", "teléfono"],
        "operador": "general",
        "tipo": "atencion_cliente",
    },
]


def calcular_score(respuesta: str, keywords: list[str]) -> float:
    """Calcula score basado en keywords esperadas en la respuesta."""
    hits = sum(1 for kw in keywords if kw.lower() in respuesta.lower())
    return round(hits / len(keywords), 3)


def evaluar() -> list[dict]:
    """Ejecuta la evaluación completa del sistema RAG y registra en MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    resultados: list[dict] = []
    scores_totales: list[float] = []
    latencias: list[float] = []

    print("=" * 65)
    print("  EVALUACION RAG — GAIA TELECOM CHATBOT")
    print("=" * 65)

    with mlflow.start_run(run_name="evaluacion_rag_gaia"):
        mlflow.log_param("total_preguntas_eval", len(EVAL_SET))
        mlflow.log_param("modelo_llm", "mistral-small-latest")
        mlflow.log_param("operadores", "claro,movistar,tigo")

        for i, item in enumerate(EVAL_SET, start=1):
            operador = item.get("operador", "general")
            tipo = item.get("tipo", "general")
            print(f"\n[Q{i:02d}] [{operador.upper()}] [{tipo}]")
            print(f"      {item['pregunta']}")

            t0 = time.time()
            respuesta = ask(item["pregunta"], session_id=f"eval_{i}")
            latencia = round(time.time() - t0, 3)

            score = calcular_score(respuesta, item["keywords"])
            scores_totales.append(score)
            latencias.append(latencia)

            mlflow.log_metric(f"score_q{i:02d}", score)
            mlflow.log_metric(f"latencia_q{i:02d}_seg", latencia)

            estado = "OK" if score >= 0.6 else "BAJO"
            print(f"      Score: {score:.2f} | Latencia: {latencia}s | {estado}")
            print(f"      Preview: {respuesta[:100]}...")

            resultados.append({
                "pregunta_num": i,
                "operador": operador,
                "tipo": tipo,
                "pregunta": item["pregunta"],
                "keywords_buscadas": item["keywords"],
                "respuesta_completa": respuesta,
                "score": score,
                "latencia_segundos": latencia,
            })

        avg_score = round(sum(scores_totales) / len(scores_totales), 3)
        avg_latencia = round(sum(latencias) / len(latencias), 3)

        mlflow.log_metric("avg_score_global", avg_score)
        mlflow.log_metric("avg_latencia_seg", avg_latencia)
        mlflow.log_metric("total_preguntas", len(EVAL_SET))

        # Score por operador
        for op in ["claro", "movistar", "tigo", "general"]:
            scores_op = [r["score"] for r in resultados if r["operador"] == op]
            if scores_op:
                avg_op = round(sum(scores_op) / len(scores_op), 3)
                mlflow.log_metric(f"avg_score_{op}", avg_op)
                print(f"\n  Score promedio {op.upper()}: {avg_op:.2f}")

        print("\n" + "=" * 65)
        print(f"  Score promedio global : {avg_score:.2f} / 1.00")
        print(f"  Latencia promedio     : {avg_latencia:.2f}s")
        print("=" * 65)

        reporte_path = EVALUATION_DIR / "resultados_evaluacion.json"
        reporte_path.write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(reporte_path), artifact_path="evaluation")
        print(f"\n  Reporte guardado en: {reporte_path}")
        print(f"  Metricas en MLflow : http://127.0.0.1:5002")

    return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluar()
