#!/usr/bin/env python3
"""Train a sklearn pipeline on 8 Flutter-aligned features + hospital mortality proxy.

Aligned with `Baptist_tester/ml_practice.py` target definition (death-like discharge codes).

Usage:
  export DIABETIC_DATA_PATH=/path/to/diabetic_data.csv
  python scripts/train_mortality_subset.py

Writes `app/artifact/mortality_pipeline.joblib` for the FastAPI service.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACK_END = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BACK_END / "app" / "artifact"
ARTIFACT_PATH = ARTIFACT_DIR / "mortality_pipeline.joblib"

FEATURE_COLUMNS = [
    "num_medications",
    "number_inpatient",
    "num_lab_procedures",
    "time_in_hospital",
    "number_diagnoses",
    "number_emergency",
    "number_outpatient",
    "age_mid",
]


def age_to_mid(value: object) -> float:
    if pd.isna(value):
        return 55.0
    text = str(value).strip()
    if text in {"", "?", "nan", "None"}:
        return 55.0
    cleaned = (
        text.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "")
    )
    if "-" not in cleaned:
        try:
            return float(cleaned)
        except ValueError:
            return 55.0
    lower, upper = cleaned.split("-", 1)
    try:
        return (float(lower) + float(upper)) / 2.0
    except ValueError:
        return 55.0


def build_target(df: pd.DataFrame) -> pd.Series:
    death_like = {11, 19, 20, 21}
    disp = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    return disp.isin(death_like).astype(int)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    for col in FEATURE_COLUMNS[:-1]:
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r} in CSV")
    if "age" not in df.columns:
        raise KeyError("Expected column 'age' (categorical) in CSV")

    out = pd.DataFrame()
    for col in FEATURE_COLUMNS[:-1]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["age_mid"] = df["age"].map(age_to_mid)

    y = build_target(df)
    mask = y.notna() & out.notna().all(axis=1)
    return out.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.environ.get("DIABETIC_DATA_PATH", ""),
        help="Path to diabetic_data.csv (or set DIABETIC_DATA_PATH)",
    )
    args = parser.parse_args()
    csv_path = args.csv_path
    if not csv_path:
        print("Provide csv_path or set DIABETIC_DATA_PATH", file=sys.stderr)
        return 1

    path = Path(csv_path).expanduser()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path).replace("?", np.nan)
    X, y = prepare_xy(df)
    if len(X) < 200:
        print(f"Too few rows after cleaning: {len(X)}", file=sys.stderr)
        return 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=0.25,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=3000,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", round(float(roc_auc_score(y_test, proba)), 4))
    print("PR-AUC :", round(float(average_precision_score(y_test, proba)), 4))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)
    print("Saved", ARTIFACT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
