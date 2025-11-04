# src/report_tables.py
"""
Build publication-ready tables (Markdown and LaTeX) from metrics_summary.csv.

Input
-----
results/metrics/metrics_summary.csv  # produced by src.evaluate

Outputs
-------
results/metrics/metrics_summary.md
results/metrics/metrics_summary.tex
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from config import ROOT

METRICS_DIR = ROOT / "results" / "metrics"
IN_CSV      = METRICS_DIR / "metrics_summary.csv"
OUT_MD      = METRICS_DIR / "metrics_summary.md"
OUT_TEX     = METRICS_DIR / "metrics_summary.tex"

NAME_MAP = {
    "naive": "Naïve",
    "mean": "Mean",
    "arima": "ARIMA",
}

COL_ORDER = ["Model", "RMSFE", "MAFE", "MSFE", "MSFE ratio vs Naive"]


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    # 'model' potrebbe essere indice nel CSV originale; normalizziamo
    if "model" in df.columns:
        df.rename(columns={"model": "Model"}, inplace=True)
    # Rinomina modelli per presentazione
    if "Model" in df.columns:
        df["Model"] = df["Model"].replace(NAME_MAP)
    # Ordina (se non già) per RMSFE crescente
    if "RMSFE" in df.columns:
        df = df.sort_values("RMSFE", ascending=True)
    # Rinomina MSFE_ratio_vs_naive per leggibilità
    if "MSFE_ratio_vs_naive" in df.columns:
        df.rename(columns={"MSFE_ratio_vs_naive": "MSFE ratio vs Naive"}, inplace=True)
    # Re-order columns se presenti
    present_cols = [c for c in COL_ORDER if c in df.columns]
    df = df[present_cols]
    return df


def format_numbers(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(ndigits)
    return out


def to_markdown(df: pd.DataFrame) -> str:
    # Costruiamo una semplice tabella Markdown per evitare dipendenze extra
    headers = df.columns.tolist()
    lines = []
    # header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # rows
    for _, row in df.iterrows():
        vals = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    df = load_summary()
    df_fmt = format_numbers(df, ndigits=3)

    # --- Markdown ---
    md = to_markdown(df_fmt)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")

    # --- LaTeX ---
    tex = df_fmt.to_latex(index=False, escape=True, column_format="lrrrr")
    OUT_TEX.write_text(tex, encoding="utf-8")

    print("=== Report tables ===")
    print(f"Saved Markdown: {OUT_MD}")
    print(f"Saved LaTeX:    {OUT_TEX}")
    print("\nPreview (Markdown):")
    print(md.splitlines()[0])
    print(md.splitlines()[1])
    print(md.splitlines()[2])


if __name__ == "__main__":
    main()
