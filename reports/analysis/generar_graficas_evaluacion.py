# -*- coding: utf-8 -*-
"""
Generación de gráficas de evaluación - Prototipo GAIA
Lee el Excel de encuesta, calcula las métricas con las mismas fórmulas
que analisis_encuesta_evaluacion.py y produce los 6 archivos PNG.

Uso:
    python generar_graficas_evaluacion.py
    python generar_graficas_evaluacion.py --excel ruta/encuesta.xlsx
    python generar_graficas_evaluacion.py --excel ruta/encuesta.xlsx --salida carpeta/

Dependencias: pandas, numpy, openpyxl, matplotlib
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Parámetros RAGAS (fijos, vienen de reports/evaluation/) ──────────────────
# Ajusta estos valores si corres la evaluación de nuevo.
RAGAS = {
    "Faithfulness":      0.120,
    "Answer Relevancy":  0.889,
    "Context Precision": 0.633,
    "Context Recall":    0.294,
}
RAGAS_TARGET = {
    "Faithfulness":      0.85,
    "Answer Relevancy":  0.80,
    "Context Precision": 0.75,
    "Context Recall":    0.80,
}

# ── Colores ───────────────────────────────────────────────────────────────────
C_OSCURO   = "#1B3358"
C_MEDIO    = "#2E75B6"
C_CLARO    = "#9DC3E6"
C_UMBRAL   = "#E07B00"
C_VERDE    = "#1A7A4A"
C_ROJO     = "#C0392B"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
})


# ════════════════════════════════════════════════════════════════════════════
# CARGA Y CÁLCULO DE MÉTRICAS
# ════════════════════════════════════════════════════════════════════════════

def cargar_y_calcular(ruta_excel: pathlib.Path) -> dict:
    """
    Lee el Excel y devuelve un dict con todas las métricas calculadas.
    Estructura esperada del Excel (hoja 1, sin fila de encabezado extra):
        Columna A  : ID participante
        Ítems 1-10 : SUS  (columnas B-K, índices 1-10 en base 0)
        Ítems 11-16: Empatía percibida
        Ítems 17-18: Resolución primer contacto
        Ítems 19-20: Confianza
        Ítem 21    : NPS (escala 0-10)
    """
    df = pd.read_excel(ruta_excel, header=0)
    df.columns = (
        ["timestamp"]
        + [f"sus_{i}" for i in range(1, 11)]
        + [f"emp_{i}" for i in range(11, 17)]
        + [f"res_{i}" for i in range(17, 19)]
        + [f"conf_{i}" for i in range(19, 21)]
        + ["nps"]
        + ["abierta_22", "abierta_23", "abierta_24"]
    )

    n = len(df)

    # ── SUS ──────────────────────────────────────────────────────────────────
    # Ítems impares (1,3,5,7,9): (valor - 1)
    # Ítems pares  (2,4,6,8,10): (5 - valor)
    sus_cols_impar = ["sus_1", "sus_3", "sus_5", "sus_7", "sus_9"]
    sus_cols_par   = ["sus_2", "sus_4", "sus_6", "sus_8", "sus_10"]
    sus_impar = df[sus_cols_impar].apply(lambda x: x - 1)
    sus_par   = df[sus_cols_par].apply(lambda x: 5 - x)
    suma_sus  = sus_impar.values + sus_par.values
    scores_sus = suma_sus.sum(axis=1) * 2.5

    sus_media = float(np.mean(scores_sus))
    sus_sd    = float(np.std(scores_sus, ddof=1))
    sus_min   = float(np.min(scores_sus))
    sus_max   = float(np.max(scores_sus))
    sus_bajo  = int(np.sum(scores_sus < 68))

    # ── Likert ────────────────────────────────────────────────────────────────
    cols_emp  = [f"emp_{i}"  for i in range(11, 17)]
    cols_res  = [f"res_{i}"  for i in range(17, 19)]
    cols_conf = [f"conf_{i}" for i in range(19, 21)]

    media_emp  = float(df[cols_emp].values.mean())
    media_res  = float(df[cols_res].values.mean())
    media_conf = float(df[cols_conf].values.mean())
    media_gen  = float(
        df[cols_emp + cols_res + cols_conf].values.mean()
    )

    medias_items = {
        col: float(df[col].mean())
        for col in cols_emp + cols_res + cols_conf
    }

    # ── TRPC (ítems 17 y 18 con valor 4 o 5) ────────────────────────────────
    trpc_count = int(
        ((df["res_17"] >= 4) & (df["res_18"] >= 4)).sum()
    )
    trpc_pct = trpc_count / n * 100

    # ── NPS ───────────────────────────────────────────────────────────────────
    promotores   = int((df["nps"] >= 9).sum())
    pasivos      = int(((df["nps"] >= 7) & (df["nps"] <= 8)).sum())
    detractores  = int((df["nps"] <= 6).sum())
    nps_score    = round((promotores - detractores) / n * 100)
    pct_prom     = promotores  / n * 100
    pct_pas      = pasivos     / n * 100
    pct_det      = detractores / n * 100

    return {
        "n": n,
        "scores_sus": scores_sus.tolist(),
        "sus_media": sus_media,
        "sus_sd": sus_sd,
        "sus_min": sus_min,
        "sus_max": sus_max,
        "sus_bajo": sus_bajo,
        "media_emp": media_emp,
        "media_res": media_res,
        "media_conf": media_conf,
        "media_gen": media_gen,
        "medias_items": medias_items,
        "trpc_pct": trpc_pct,
        "promotores": promotores,
        "pasivos": pasivos,
        "detractores": detractores,
        "nps_score": nps_score,
        "pct_prom": pct_prom,
        "pct_pas": pct_pas,
        "pct_det": pct_det,
    }


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — NPS
# ════════════════════════════════════════════════════════════════════════════

def grafica_nps(m: dict, salida: pathlib.Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    categorias = [
        f"Promotores\n(NPS 9-10)",
        f"Pasivos\n(NPS 7-8)",
        f"Detractores\n(NPS 0-6)",
    ]
    conteos = [m["promotores"], m["pasivos"], m["detractores"]]
    pcts    = [m["pct_prom"],   m["pct_pas"], m["pct_det"]]
    colores = [C_OSCURO, C_MEDIO, C_CLARO]

    barras = ax.bar(categorias, conteos, color=colores,
                    width=0.5, edgecolor="white", linewidth=0.8)

    for barra, cnt, pct in zip(barras, conteos, pcts):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.3,
            f"{cnt} participantes\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="#1B2A3A",
        )

    formula = (
        f"NPS = % Promotores - % Detractores = "
        f"{m['pct_prom']:.1f}% - {m['pct_det']:.1f}% = "
        f"+{m['nps_score']} puntos"
    )
    ax.text(
        0.5, 0.94, formula,
        transform=ax.transAxes, ha="center", va="top", fontsize=11,
        color="#1B2A3A", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#D6E4F7",
                  edgecolor=C_MEDIO, linewidth=1.5),
    )

    ax.set_title(
        "Distribución Net Promoter Score (NPS)\n"
        "Promotor, Pasivo y Detractor - Prototipo GAIA",
        fontsize=14, fontweight="bold", color=C_OSCURO, pad=16,
    )
    ax.set_ylabel("Número de participantes", fontsize=12)
    ax.set_ylim(0, max(conteos) * 1.35)
    ax.tick_params(axis="x", labelsize=11)
    fig.tight_layout()
    ruta = salida / "grafica_01_nps.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {ruta}")


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — SUS por participante
# ════════════════════════════════════════════════════════════════════════════

def grafica_sus(m: dict, salida: pathlib.Path):
    scores    = m["scores_sus"]
    n         = m["n"]
    promedio  = m["sus_media"]
    umbral    = 68.0
    part      = list(range(1, n + 1))

    colores_barras = [
        C_OSCURO if s >= umbral else C_CLARO for s in scores
    ]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.55), 6))
    ax.bar(part, scores, color=colores_barras,
           edgecolor="white", linewidth=0.6)

    ax.axhline(umbral,   color=C_UMBRAL, linestyle="--",
               linewidth=1.8)
    ax.axhline(promedio, color=C_VERDE,  linestyle="-",
               linewidth=1.8)

    ax.text(n + 0.6, promedio, f"{promedio:.2f}",
            va="center", fontsize=9, color=C_VERDE, fontweight="bold")
    ax.text(n + 0.6, umbral,   f"{umbral}",
            va="center", fontsize=9, color=C_UMBRAL, fontweight="bold")

    leyenda = [
        mpatches.Patch(color=C_OSCURO, label="Cumple umbral (>=68)"),
        mpatches.Patch(color=C_CLARO,  label="Por debajo del umbral (<68)"),
        plt.Line2D([0], [0], color=C_UMBRAL, linestyle="--",
                   linewidth=1.8, label="Umbral aceptable (68)"),
        plt.Line2D([0], [0], color=C_VERDE, linestyle="-",
                   linewidth=1.8, label=f"Promedio GAIA ({promedio:.2f})"),
    ]
    ax.legend(handles=leyenda, loc="lower right",
              fontsize=9, framealpha=0.9)

    ax.set_title(
        "Distribución de Scores SUS por Participante\n"
        "Escala de Usabilidad del Sistema (SUS) - Prototipo GAIA",
        fontsize=13, fontweight="bold", color=C_OSCURO, pad=14,
    )
    ax.set_xlabel("Participante", fontsize=12)
    ax.set_ylabel("Score SUS", fontsize=12)
    ax.set_xticks(part)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    ruta = salida / "grafica_02_sus_participantes.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {ruta}")


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Media por ítem Likert
# ════════════════════════════════════════════════════════════════════════════

def grafica_likert_items(m: dict, salida: pathlib.Path):
    etiquetas_legibles = {
        "emp_11":  "Ítem 11\nEmpatía",
        "emp_12":  "Ítem 12\nEmpatía",
        "emp_13":  "Ítem 13\nEmpatía",
        "emp_14":  "Ítem 14\nEmpatía",
        "emp_15":  "Ítem 15\nEmpatía",
        "emp_16":  "Ítem 16\nEmpatía",
        "res_17":  "Ítem 17\nResolución",
        "res_18":  "Ítem 18\nResolución",
        "conf_19": "Ítem 19\nConfianza",
        "conf_20": "Ítem 20\nConfianza",
    }
    colores_items = (
        [C_OSCURO] * 6 + [C_MEDIO] * 2 + [C_CLARO] * 2
    )

    etiquetas = list(etiquetas_legibles.values())
    medias    = [m["medias_items"][k] for k in etiquetas_legibles]

    fig, ax = plt.subplots(figsize=(13, 6))
    barras = ax.bar(etiquetas, medias, color=colores_items,
                    edgecolor="white", linewidth=0.8, width=0.6)

    for barra, val in zip(barras, medias):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#1B2A3A",
        )

    ax.axhline(3.5, color=C_UMBRAL, linestyle="--",
               linewidth=1.8, label="Umbral percepción positiva (3.5)")

    leyenda = [
        mpatches.Patch(color=C_OSCURO, label="Empatía percibida (ítems 11-16)"),
        mpatches.Patch(color=C_MEDIO,  label="Resolución primer contacto (ítems 17-18)"),
        mpatches.Patch(color=C_CLARO,  label="Confianza (ítems 19-20)"),
        plt.Line2D([0], [0], color=C_UMBRAL, linestyle="--",
                   linewidth=1.8, label="Umbral percepción positiva (3.5)"),
    ]
    ax.legend(handles=leyenda, loc="lower right",
              fontsize=9, framealpha=0.9)

    ax.set_title(
        "Media por Ítem - Escala Likert\n"
        "Percepción de Empatía y Satisfacción - Prototipo GAIA",
        fontsize=13, fontweight="bold", color=C_OSCURO, pad=14,
    )
    ax.set_ylabel("Media (escala 1-5)", fontsize=12)
    ax.set_ylim(0, 5.2)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    ruta = salida / "grafica_03_likert_items.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {ruta}")


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Dimensiones Likert (barras horizontales)
# ════════════════════════════════════════════════════════════════════════════

def grafica_dimensiones(m: dict, salida: pathlib.Path):
    # Orden: Likert general arriba → Empatía abajo (invert_yaxis lo invierte)
    datos = [
        ("Likert\ngeneral",
         m["media_gen"],  "#3A7FC1", m["media_gen"] > 3.5),
        ("Confianza\n(ítems 19-20)",
         m["media_conf"], C_CLARO,  m["media_conf"] > 3.5),
        ("Resolución en\nprimer contacto\n(ítems 17-18)",
         m["media_res"],  C_MEDIO,  m["media_res"]  > 3.5),
        ("Empatía\npercibida\n(ítems 11-16)",
         m["media_emp"],  C_OSCURO, m["media_emp"]  > 3.5),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    etiquetas = [d[0] for d in datos]
    for i, (etq, val, col, cump) in enumerate(datos):
        ax.barh(i, val, color=col, height=0.55, edgecolor="white")
        estado  = "Cumple"    if cump else "No cumple"
        color_t = C_VERDE if cump else C_ROJO
        ax.text(
            val + 0.05, i,
            f"{val:.2f} - {estado}",
            va="center", fontsize=11, fontweight="bold", color=color_t,
        )

    ax.axvline(3.5, color=C_UMBRAL, linestyle="--", linewidth=1.8,
               label="Umbral percepción positiva (3.5)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_yticks(range(len(datos)))
    ax.set_yticklabels(etiquetas, fontsize=10)
    ax.set_title(
        "Resultados por Dimensión - Escala Likert\n"
        "Promedios generales - Prototipo GAIA",
        fontsize=13, fontweight="bold", color=C_OSCURO, pad=14,
    )
    ax.set_xlabel("Media (escala 1-5)", fontsize=12)
    ax.set_xlim(0, 5.2)
    ax.invert_yaxis()
    fig.tight_layout()
    ruta = salida / "grafica_04_dimensiones_likert.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {ruta}")


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 5 — Tabla resumen básica (SUS + Likert + NPS)
# ════════════════════════════════════════════════════════════════════════════

def grafica_tabla_basica(m: dict, salida: pathlib.Path):
    indicadores = [
        ("Usabilidad (SUS)",
         f"{m['sus_media']:.2f} / 100", ">68", "Bueno",
         m["sus_media"] > 68),
        ("Empatía percibida (Likert)",
         f"{m['media_emp']:.2f} / 5", ">3.5", "4 de 5",
         m["media_emp"] > 3.5),
        ("Resolución primer contacto",
         f"{m['media_res']:.2f} / 5", ">3.5", "3 de 5",
         m["media_res"] > 3.5),
        ("Confianza",
         f"{m['media_conf']:.2f} / 5", ">3.5", "4 de 5",
         m["media_conf"] > 3.5),
        ("Satisfacción (NPS)",
         f"+{m['nps_score']} puntos", ">0", "Positivo",
         m["nps_score"] > 0),
    ]
    _dibujar_tabla(
        indicadores,
        ["Indicador", "Valor obtenido", "Target", "Interpretación", "Estado"],
        "Tabla Resumen de Indicadores de Evaluación - Prototipo GAIA",
        salida / "grafica_05_tabla_resumen.png",
        figsize=(13, 4.0),
    )


# ════════════════════════════════════════════════════════════════════════════
# GRÁFICA 6 — Tabla resumen extendida (+ TRPC + RAGAS)
# ════════════════════════════════════════════════════════════════════════════

def grafica_tabla_extendida(m: dict, salida: pathlib.Path):
    ragas_interp = {
        "Faithfulness":      "Baja fidelidad al contexto",
        "Answer Relevancy":  "Alta pertinencia",
        "Context Precision": "Precisión moderada",
        "Context Recall":    "Cobertura baja",
    }
    ragas_instrumento = {
        "Faithfulness":      "RAGAS",
        "Answer Relevancy":  "RAGAS",
        "Context Precision": "RAGAS",
        "Context Recall":    "RAGAS",
    }

    indicadores = [
        ("Usabilidad",
         "SUS",
         f"{m['sus_media']:.2f} / 100",
         "> 68",
         "Sobre el promedio",
         m["sus_media"] > 68),
        ("Empatía percibida",
         "Likert (ítems 11-16)",
         f"{m['media_emp']:.2f} / 5",
         "> 3.5",
         "Positiva",
         m["media_emp"] > 3.5),
        ("Resolución en primer contacto",
         "Likert (ítems 17-18)",
         f"{m['media_res']:.2f} / 5",
         "> 3.5",
         "Por debajo del umbral",
         m["media_res"] > 3.5),
        ("Confianza",
         "Likert (ítems 19-20)",
         f"{m['media_conf']:.2f} / 5",
         "> 3.5",
         "Positiva",
         m["media_conf"] > 3.5),
        ("Satisfacción",
         "NPS (pregunta 21)",
         f"+{m['nps_score']}",
         "> 0",
         "Positiva",
         m["nps_score"] > 0),
        ("TRPC aproximada",
         "Ítems 17 y 18\n(ambos con 4 o 5)",
         f"{m['trpc_pct']:.1f} %",
         "> 70 %",
         "Por debajo del umbral",
         m["trpc_pct"] > 70),
    ] + [
        (nombre,
         ragas_instrumento[nombre],
         f"{valor:.3f}",
         f"> {RAGAS_TARGET[nombre]:.2f}",
         ragas_interp[nombre],
         valor > RAGAS_TARGET[nombre])
        for nombre, valor in RAGAS.items()
    ]

    _dibujar_tabla(
        indicadores,
        ["Indicador", "Instrumento", "Valor\nobtenido",
         "Valor de\nreferencia", "Interpretación", "Estado"],
        "Tabla Resumen de Indicadores de Evaluación - Prototipo GAIA",
        salida / "grafica_06_tabla_extendida.png",
        figsize=(15, 6.5),
        col_estado=5,
    )


# ── Helper tabla ─────────────────────────────────────────────────────────────

def _dibujar_tabla(
    indicadores, encabezados, titulo, ruta,
    figsize=(13, 4.5), col_estado=4,
):
    filas = [
        list(row[:-1]) + [("Cumple" if row[-1] else "No cumple")]
        for row in indicadores
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    tabla = ax.table(
        cellText=filas,
        colLabels=encabezados,
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    escala_y = max(1.8, 5.5 / len(filas))
    tabla.scale(1, escala_y)

    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor("#CCCCCC")
        celda.set_linewidth(0.5)
        if fila == 0:
            celda.set_facecolor(C_OSCURO)
            celda.set_text_props(color="white", fontweight="bold")
        else:
            celda.set_facecolor("#EAF1FB" if fila % 2 == 0 else "white")
            if col == col_estado:
                texto = celda.get_text().get_text()
                color_t = C_VERDE if texto == "Cumple" else C_ROJO
                celda.set_text_props(color=color_t, fontweight="bold")

    ax.set_title(titulo, fontsize=13, fontweight="bold",
                 color=C_OSCURO, pad=18)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {ruta}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Genera las gráficas de evaluación del prototipo GAIA"
    )
    parser.add_argument(
        "--excel",
        default="data/encuestas/encuesta_evaluacion.xlsx",
        help="Ruta al archivo Excel de la encuesta de evaluación",
    )
    parser.add_argument(
        "--salida",
        default="reports/analysis/figures",
        help="Carpeta donde se guardan los PNG (se crea si no existe)",
    )
    args = parser.parse_args()

    ruta_excel = pathlib.Path(args.excel)
    salida     = pathlib.Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    if not ruta_excel.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo Excel: {ruta_excel}\n"
            "Usa --excel para indicar la ruta correcta."
        )

    print(f"Leyendo: {ruta_excel}")
    m = cargar_y_calcular(ruta_excel)
    print(f"n = {m['n']} participantes | SUS = {m['sus_media']:.2f} | NPS = +{m['nps_score']}\n")

    grafica_nps(m, salida)
    grafica_sus(m, salida)
    grafica_likert_items(m, salida)
    grafica_dimensiones(m, salida)
    grafica_tabla_basica(m, salida)
    grafica_tabla_extendida(m, salida)

    print(f"\nTodas las gráficas generadas en: {salida.resolve()}")


if __name__ == "__main__":
    main()