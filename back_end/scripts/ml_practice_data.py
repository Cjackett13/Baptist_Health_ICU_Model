"""
Data prep matching `Baptist_tester/ml_practice.py` (Ridge logistic baseline).

Import this from export / training scripts so behavior stays aligned with ml_practice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

THEORETICAL_QUANTILE_LIMITS = {
    "number_outpatient": (-4.0, 4.0),
    "time_in_hospital": (-3.75, 3.75),
    "num_lab_procedures": (-4.0, 4.0),
    "num_medications": (-4.0, 4.0),
    "number_diagnoses": (-3.75, 3.75),
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path).replace("?", np.nan)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    death_like_codes = {11, 19, 20, 21}
    out = df.copy()
    out["hospital_mortality_proxy"] = (
        out["discharge_disposition_id"].isin(death_like_codes).astype(int)
    )
    return out


def apply_theoretical_quantile_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    for feature, (lower, upper) in THEORETICAL_QUANTILE_LIMITS.items():
        if feature not in filtered.columns:
            continue
        numeric_feature = pd.to_numeric(filtered[feature], errors="coerce")
        std = numeric_feature.std(ddof=1)
        if pd.isna(std) or std == 0:
            continue
        z_scores = (numeric_feature - numeric_feature.mean()) / std
        keep_mask = z_scores.isna() | ((z_scores >= lower) & (z_scores <= upper))
        filtered = filtered.loc[keep_mask].copy()
    return filtered


def build_features_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    leak_or_id_cols = [
        "hospital_mortality_proxy",
        "encounter_id",
        "patient_nbr",
        "discharge_disposition_id",
    ]
    X = df.drop(columns=leak_or_id_cols, errors="ignore")
    y = df["hospital_mortality_proxy"]
    missing_pct = X.isna().mean() * 100
    high_missing_cols = missing_pct[missing_pct > 80].index.tolist()
    X = X.drop(columns=high_missing_cols, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, high_missing_cols


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def make_ridge_ml_practice_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """First model in ml_practice.main() — Ridge logistic."""
    ridge = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        class_weight="balanced",
        max_iter=3000,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", ridge)])

