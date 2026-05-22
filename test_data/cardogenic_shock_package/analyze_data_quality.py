#!/usr/bin/env python3
"""Summarize cleanliness and missingness for SHOCK / cardiogenic shock package CSVs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent
CSVS = ["trial_deident.csv", "echo_deident.csv", "survival_deident.csv"]


def _is_sentinel_missing(val: object) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    if isinstance(val, str):
        t = val.strip()
        if not t:
            return True
        if t in {".", "NA", "NaN", "None", "?", "NULL"}:
            return True
        # SAS-style missing numeric printed as spaced dots
        if re.fullmatch(r"[\s.]+", t):
            return True
    return False


def missing_mask(series: pd.Series) -> pd.Series:
    """True where cell should be treated as missing for this audit."""
    if pd.api.types.is_numeric_dtype(series):
        na = series.isna()
        return na
    # object / string: NA + blank + SAS-like dots
    return series.map(_is_sentinel_missing)


def summarize_frame(df: pd.DataFrame, name: str) -> dict:
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        m = missing_mask(s)
        miss = int(m.sum())
        pct = round(100.0 * miss / n, 2) if n else 0.0
        nonnull = s[~m]
        nunique = int(nonnull.nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_n": miss,
                "missing_pct": pct,
                "non_missing_n": int(n - miss),
                "n_unique_approx": nunique,
            }
        )
    rows.sort(key=lambda r: -r["missing_pct"])
    return {"file": name, "n_rows": n, "n_columns": len(df.columns), "columns": rows}


def main() -> None:
    summaries = []
    lines: list[str] = []
    lines.append("=== Cardiogenic shock package (SHOCK trial CSVs) — data quality ===\n")

    for fname in CSVS:
        path = DIR / fname
        if not path.is_file():
            lines.append(f"MISSING FILE: {path}\n")
            continue
        df = pd.read_csv(path, low_memory=False)
        summ = summarize_frame(df, fname)
        summaries.append(summ)

        lines.append(f"\n--- {fname} ---")
        lines.append(f"Rows: {summ['n_rows']:,}  |  Columns: {summ['n_columns']}")
        complete_cols = sum(1 for c in summ["columns"] if c["missing_pct"] == 0)
        heavy = sum(1 for c in summ["columns"] if c["missing_pct"] >= 50)
        lines.append(
            f"Columns with 0% missing: {complete_cols}  |  "
            f"Columns with ≥50% missing: {heavy}"
        )
        lines.append("\nTop 25 columns by missing % (field name = category/variable):")
        lines.append(
            f"{'column':<36} {'missing%':>9} {'missing_n':>10} {'non-miss':>10} {'n_unique':>10}"
        )
        for c in summ["columns"][:25]:
            lines.append(
                f"{c['column'][:35]:<36} {c['missing_pct']:>9.2f} {c['missing_n']:>10,} "
                f"{c['non_missing_n']:>10,} {c['n_unique_approx']:>10,}"
            )
        if len(summ["columns"]) > 25:
            lines.append(f"  ... plus {len(summ['columns']) - 25} more columns (see JSON)")

        # Overall "cleanliness" score: mean missing % across columns
        mean_miss = (
            sum(c["missing_pct"] for c in summ["columns"]) / len(summ["columns"])
            if summ["columns"]
            else 0
        )
        lines.append(f"\nMean missing % across all columns: {mean_miss:.2f}%")
        lines.append(
            "(Many fields are intentionally blank until that form/timepoint is collected.)"
        )

    out_txt = DIR / "data_quality_report.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json = DIR / "data_quality_report.json"
    out_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(out_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
