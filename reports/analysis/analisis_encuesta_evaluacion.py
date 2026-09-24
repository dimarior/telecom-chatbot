"""
Análisis descriptivo de la encuesta de evaluación del prototipo GAIA (n = 30).
Reproduce: SUS, escala Likert por dimensión y por ítem, TRPC aproximada
(ítems 17 y 18) y NPS (pregunta 21).

Uso (desde la raíz del repositorio):
    python reports/analysis/analisis_encuesta_evaluacion.py
Requiere: pandas, numpy, openpyxl
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Raíz del repositorio: reports/analysis/<script> -> subir dos niveles
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "encuestas"
EVAL_FILE = DATA_DIR / "Encuesta de Evaluación - Asistente Virtual GAIA (respuestas).xlsx"


def nps(scores: pd.Series) -> dict:
    n = len(scores)
    p = int((scores >= 9).sum())
    d = int((scores <= 6).sum())
    pas = n - p - d
    return {"n": n, "promotores": p, "pasivos": pas, "detractores": d,
            "NPS": round((p - d) / n * 100, 1)}


def analizar_evaluacion(path: Path) -> None:
    df = pd.read_excel(path)
    print(f"\n=== ENCUESTA DE EVALUACIÓN GAIA (n = {len(df)}) ===")

    # --- SUS (ítems 1-10, Brooke, 1996)
    sus_items = df.iloc[:, 1:11].astype(float)
    impares = sus_items.iloc[:, [0, 2, 4, 6, 8]] - 1
    pares = 5 - sus_items.iloc[:, [1, 3, 5, 7, 9]]
    sus = (impares.sum(axis=1) + pares.sum(axis=1)) * 2.5
    print(f"SUS: media = {sus.mean():.2f}, DE = {sus.std():.2f}, "
          f"mín = {sus.min():.1f}, máx = {sus.max():.1f}, "
          f"< 68: {(sus < 68).sum()} ({(sus < 68).mean()*100:.1f} %)")

    # --- Likert (ítems 11-20)
    lik = df.iloc[:, 11:21].astype(float)
    lik.columns = [f"item_{i}" for i in range(11, 21)]
    dimensiones = {
        "Empatía percibida (11-16)": lik.iloc[:, 0:6],
        "Resolución en primer contacto (17-18)": lik.iloc[:, 6:8],
        "Confianza (19-20)": lik.iloc[:, 8:10],
        "Likert general (11-20)": lik,
    }
    print("\nDimensiones Likert (media por participante, luego media del grupo):")
    for nombre, bloque in dimensiones.items():
        m = bloque.mean(axis=1)
        print(f"  {nombre}: media = {m.mean():.2f}, DE = {m.std():.2f}")

    print("\nMedia y DE por ítem:")
    for col in lik.columns:
        print(f"  {col}: media = {lik[col].mean():.2f}, DE = {lik[col].std():.2f}, "
              f"% de acuerdo (4-5) = {(lik[col] >= 4).mean()*100:.1f} %")

    # --- TRPC aproximada: ítems 17 y 18 simultáneamente con 4 o 5
    trpc = ((lik["item_17"] >= 4) & (lik["item_18"] >= 4))
    print(f"\nTRPC aproximada (ítems 17 y 18 >= 4): {trpc.sum()} de {len(trpc)} "
          f"= {trpc.mean()*100:.1f} %")

    # --- NPS (pregunta 21)
    r = nps(df.iloc[:, 21].astype(int))
    print(f"\nNPS GAIA (pregunta 21): {r}")


if __name__ == "__main__":
    analizar_evaluacion(EVAL_FILE)
