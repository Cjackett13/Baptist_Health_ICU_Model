#!/usr/bin/env python3
"""
Verify ``infusion_residual_within_scai_12h`` computation.

Two definitions in this repo:
  A) DuckDB sidecar (EXPLORATORY / parquet): cohort-wide mean by round(current_scai_12h)
  B) Training preprocessor (MODEL): train-fold-only bin means on med_infusion_mean_12h
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.length_of_stay.config import REPO_ROOT
from models.length_of_stay.transforms import RESIDUAL_BIN_COL, RESIDUAL_OUT_COL, RESIDUAL_VALUE_COL
from models.length_of_stay.train import load_training_frame, split_by_manifest


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _duckdb_style_residual(df: pd.DataFrame) -> pd.Series:
    """Replicate DuckDB: value - AVG(value) OVER (PARTITION BY round(current_scai_12h))."""
    bins = pd.to_numeric(df[RESIDUAL_BIN_COL], errors="coerce").round()
    vals = pd.to_numeric(df[RESIDUAL_VALUE_COL], errors="coerce")
    means = vals.groupby(bins).transform("mean")
    return vals - means


def _train_fold_residual(df: pd.DataFrame, train_bins_means: dict, global_mean: float) -> pd.Series:
    bins = pd.to_numeric(df[RESIDUAL_BIN_COL], errors="coerce").round().astype("Int64")
    vals = pd.to_numeric(df[RESIDUAL_VALUE_COL], errors="coerce")
    means = bins.map(train_bins_means).astype(float).fillna(global_mean)
    return vals - means


def run_residual_audit(data_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    df = load_training_frame(data_dir)
    tr, _, te = split_by_manifest(df, data_dir)

    need = [RESIDUAL_VALUE_COL, RESIDUAL_BIN_COL]
    for c in need:
        if c not in df.columns:
            return {"error": f"missing column {c}"}

    # Train-fold statistics (same as LosTrainPreprocessor.fit)
    tr_bins = pd.to_numeric(tr[RESIDUAL_BIN_COL], errors="coerce").round()
    tr_vals = pd.to_numeric(tr[RESIDUAL_VALUE_COL], errors="coerce")
    frame = pd.DataFrame({"bin": tr_bins, "val": tr_vals}).dropna()
    global_mean = float(frame["val"].mean()) if len(frame) else 0.0
    bin_means = frame.groupby("bin")["val"].mean().to_dict()
    bin_means_serial = {str(int(k) if k == k else k): float(v) for k, v in bin_means.items()}

    duck_tr = _duckdb_style_residual(tr)
    model_tr = _train_fold_residual(tr, bin_means, global_mean)
    duck_te = _duckdb_style_residual(te)
    model_te = _train_fold_residual(te, bin_means, global_mean)

    parquet_col = te[RESIDUAL_OUT_COL] if RESIDUAL_OUT_COL in te.columns else pd.Series(dtype=float)
    has_parquet = RESIDUAL_OUT_COL in te.columns

    def _cmp(a: pd.Series, b: pd.Series) -> dict[str, float]:
        m = a.notna() & b.notna()
        if m.sum() == 0:
            return {}
        d = (a[m] - b[m]).abs()
        return {
            "n": int(m.sum()),
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "correlation": float(a[m].corr(b[m])),
        }

    report = {
        "feature": RESIDUAL_OUT_COL,
        "value_column": RESIDUAL_VALUE_COL,
        "bin_column": RESIDUAL_BIN_COL,
        "definitions": {
            "duckdb_sidecar": (
                "med_infusion_mean_12h - AVG(med_infusion_mean_12h) "
                "OVER (PARTITION BY CAST(ROUND(current_scai_12h) AS INT)) on full cohort at SQL time"
            ),
            "model_training": (
                "Same formula but bin means computed on TRAIN fold only during LosTrainPreprocessor.fit; "
                "test/val use train bin means (no test leakage)"
            ),
        },
        "train_fold_bin_means": bin_means_serial,
        "train_fold_global_mean": global_mean,
        "comparison_train_duckdb_vs_train_fold_policy": _cmp(duck_tr, model_tr),
        "comparison_test_duckdb_vs_train_fold_policy": _cmp(duck_te, model_te),
        "model_uses": "train_fold_policy (recomputed in pipeline; not DuckDB parquet column)",
    }
    if has_parquet:
        report["comparison_test_parquet_vs_train_fold_policy"] = _cmp(
            parquet_col, model_te
        )
        report["comparison_test_parquet_vs_duckdb"] = _cmp(parquet_col, duck_te)
    report["leakage_assessment"] = (
        "PASS for strict policy: bin means fit on train only. "
        "WARN: DuckDB audit column uses full-cohort means — do not use parquet residual for training; "
        "pipeline overwrites with train-fold version."
    )
    return report
