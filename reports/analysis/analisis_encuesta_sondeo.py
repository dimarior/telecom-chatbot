"""
Análisis descriptivo de la encuesta de sondeo diagnóstica (línea base, n = 30).
Reproduce: frecuencias y porcentajes por pregunta, medias de las preguntas
Likert (excluyendo "No aplica") y NPS de expectativas (E2).

Uso (desde la raíz del repositorio):
    python reports/analysis/analisis_encuesta_sondeo.py
Requiere: pandas, numpy, openpyxl
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Raíz del repositorio: reports/analysis/<script> -> subir dos niveles
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "encuestas"
SONDEO_FILE = DATA_DIR / "Encuesta Sondeo Chatbots y Atención al Cliente en Telecomunicaciones (respuestas).xlsx"


def nps(scores: pd.Series) -> dict:
    n = len(scores)
    p = int((scores >= 9).sum())
    d = int((scores <= 6).sum())
    pas = n - p - d
    return {"n": n, "promotores": p, "pasivos": pas, "detractores": d,
            "NPS": round((p - d) / n * 100, 1)}


def analizar_sondeo(path: Path) -> None:
    df = pd.read_excel(path)
    n = len(df)
    print(f"\n=== ENCUESTA DE SONDEO DIAGNÓSTICA (n = {n}) ===")
    multiple = ("B2", "C3")
    for col in df.columns[1:]:
        codigo = col.split(".")[0]
        print(f"\n{col}")
        if codigo in multiple:
            conteo = (df[col].dropna().astype(str).str.split(",").explode()
                      .str.strip().value_counts())
        else:
            conteo = df[col].astype(str).value_counts()
        for cat, k in conteo.items():
            print(f"  {cat}: {k} ({k/n*100:.1f} %)")

    # Medias Likert excluyendo "No aplica" (el valor numérico va al inicio: "3 = Regular")
    print("\nMedias de preguntas Likert (excluye 'No aplica'):")
    for codigo in ("B4", "C2", "C5", "D1", "D2", "D4", "E1", "E5"):
        col = [c for c in df.columns if c.startswith(codigo + ".")][0]
        valores = pd.to_numeric(df[col].astype(str).str.extract(r"^(\d)")[0], errors="coerce").dropna()
        print(f"  {codigo}: media = {valores.mean():.2f} (n = {len(valores)})")

    e2 = [c for c in df.columns if c.startswith("E2.")][0]
    print(f"\nNPS de expectativas (E2): {nps(df[e2].astype(int))}")


if __name__ == "__main__":
    analizar_sondeo(SONDEO_FILE)
