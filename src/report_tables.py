# src/report_tables.py
"""
Build publication-ready tables (Markdown and LaTeX) from metrics_summary.csv
and (if available) dm_tests.csv. Includes DM vs Naive columns.

Inputs
------
results/metrics/metrics_summary.csv  # from src.evaluate
results/metrics/dm_tests.csv         # optional, from src.evaluate

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
IN_SUMMARY  = METRICS_DIR / "metrics_summary.csv"
IN_DM       = METRICS_DIR / "dm_tests.csv"           # opzionale
OUT_MD      = METRICS_DIR / "metrics_summary.md"
OUT_TEX     = METRICS_DIR / "metrics_summary.tex"

NAME_MAP = {"naive": "Naïve", "mean": "Mean", "arima": "ARIMA"}

BASE_COLS = ["Model", "RMSFE", "MAFE", "MSFE", "MSFE ratio vs Naive"]
DM_COLS   = ["DM stat vs Naïve", "DM p-value vs Naïve"]


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(IN_SUMMARY)
    # normalizza struttura
    if "model" in df.columns:
        df.rename(columns={"model": "Model"}, inplace=True)
    if "MSFE_ratio_vs_naive" in df.columns:
        df.rename(columns={"MSFE_ratio_vs_naive": "MSFE ratio vs Naive"}, inplace=True)
    if "DM_vs_naive_stat" in df.columns:
        df.rename(columns={"DM_vs_naive_stat": "DM stat vs Naïve"}, inplace=True)
    if "DM_vs_naive_pvalue" in df.columns:
        df.rename(columns={"DM_vs_naive_pvalue": "DM p-value vs Naïve"}, inplace=True)

    # Mappa nomi modello
    if "Model" in df.columns:
        df["Model"] = df["Model"].replace(NAME_MAP)

    # Ordina per RMSFE crescente (se presente)
    if "RMSFE" in df.columns:
        df = df.sort_values("RMSFE", ascending=True)

    # Reorder colonne se presenti
    present = [c for c in BASE_COLS + DM_COLS if c in df.columns]
    df = df[present]
    return df


def maybe_merge_dm(df: pd.DataFrame) -> pd.DataFrame:
    """Se le colonne DM non sono in summary, prova a caricarle da dm_tests.csv."""
    if all(col in df.columns for col in DM_COLS):
        return df  # già presenti

    if not IN_DM.exists():
        # nessun dm_tests disponibile
        # aggiungi colonne vuote se mancanti per compattezza
        for c in DM_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df

    dm = pd.read_csv(IN_DM)
    # dm ha righe per (model_i vs naive). Vogliamo allineare per Model (ARIMA/Mean)
    # prepariamo una serie per stat/pvalue
    dm["Model"] = dm["model_i"].replace(NAME_MAP)
    dm_map = dm.set_index("Model")[["DM_stat", "DM_pvalue"]]
    dm_map.rename(columns={"DM_stat": "DM stat vs Naïve",
                           "DM_pvalue": "DM p-value vs Naïve"}, inplace=True)

    out = df.copy()
    # se "Model" è indice o colonna, gestisci entrambi i casi
    if "Model" in out.columns:
        out = out.set_index("Model")
    out = out.join(dm_map, how="left")
    out = out.reset_index()

    # Ordina colonne nella sequenza desiderata
    present = [c for c in BASE_COLS + DM_COLS if c in out.columns]
    out = out[present]
    return out


def format_numbers(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
    out = df.copy()
    # arrotonda numeriche; lascia le stringhe (Model) intatte
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(ndigits)
    # migliora leggibilità: NaN DM → stringhe vuote (per output)
    for c in DM_COLS:
        if c in out.columns:
            out[c] = out[c].astype(object).where(~out[c].isna(), "")
    return out


def to_markdown(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def latex_col_format(ncols: int) -> str:
    # prima colonna (Model) allineata a sinistra, le altre a destra
    return "l" + "r" * (ncols - 1)


def main():
    df = load_summary()
    df = maybe_merge_dm(df)
    df_fmt = format_numbers(df, ndigits=3)

    # Salva Markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(to_markdown(df_fmt), encoding="utf-8")

    # Salva LaTeX
    colfmt = latex_col_format(df_fmt.shape[1])
    tex = df_fmt.to_latex(index=False, escape=True, column_format=colfmt)
    OUT_TEX.write_text(tex, encoding="utf-8")

    print("=== Report tables ===")
    print(f"Saved Markdown: {OUT_MD}")
    print(f"Saved LaTeX:    {OUT_TEX}")
    # Preview prima riga
    print("\nPreview (Markdown):")
    md = to_markdown(df_fmt).splitlines()
    for line in md[:3]:
        print(line)


if __name__ == "__main__":
    main()
