#!/usr/bin/env python3
"""
Train the Ridge logistic pipeline from `ml_practice`, compute SHAP, and export:

1) `mortality_shap_bundle.json` — Flutter-compatible `shap_mortality` list plus metadata.
   The FastAPI route `GET /v1/mortality/shap-bundle` returns this exact JSON.

2) `mortality_predictions_all.pkl` — joblib dict with full prediction table + SHAP eval slice.

Usage:
  pip install -r back_end/requirements.txt
  export DIABETIC_DATA_PATH=/path/to/diabetic_data.csv
  python back_end/scripts/export_shap_predictions_bundle.py

Outputs default to `back_end/app/artifact/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow `from scripts.ml_practice_data import ...` when run as file
SCRIPTS_DIR = Path(__file__).resolve().parent
BACK_END = SCRIPTS_DIR.parent
if str(BACK_END) not in sys.path:
    sys.path.insert(0, str(BACK_END))

from scripts.ml_practice_data import (  # noqa: E402
    apply_theoretical_quantile_filters,
    build_features_and_split,
    build_target,
    load_dataset,
    make_preprocessor,
    make_ridge_ml_practice_pipeline,
)

DECISION_THRESHOLD = 0.25
MODEL_VERSION = "ridge-logistic-ml-practice-v1"
JSON_NAME = "mortality_shap_bundle.json"
PKL_NAME = "mortality_predictions_all.pkl"
PIPELINE_NAME = "mortality_ml_practice_ridge_pipeline.joblib"


def _shap_values_positive_class(raw_values: np.ndarray) -> np.ndarray:
    """Reduce SHAP output to shape (n_samples, n_features) for positive class."""
    if raw_values.ndim == 3:
        # (samples, features, 2) for binary logistic
        return raw_values[:, :, 1]
    if raw_values.ndim == 2:
        return raw_values
    raise ValueError(f"Unexpected shap values ndim={raw_values.ndim}")


def _mean_abs_shap_to_flutter_shap(
    mean_signed: np.ndarray,
    feature_names: list[str],
    top_k: int = 20,
) -> list[dict]:
    """Build `shap_mortality` entries matching Flutter `ShapValue.fromJson`."""
    n = min(mean_signed.shape[0], len(feature_names))
    mean_signed = mean_signed[:n]
    feature_names = feature_names[:n]
    order = np.argsort(-np.abs(mean_signed))[:top_k]
    out: list[dict] = []
    for i in order:
        v = float(mean_signed[i])
        mag = round(abs(v), 4)
        direction = "positive" if v >= 0 else "negative"
        name = str(feature_names[i]) if i < len(feature_names) else f"feature_{i}"
        out.append({"feature": name, "value": mag, "direction": direction})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SHAP JSON + predictions PKL")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.environ.get("DIABETIC_DATA_PATH", ""),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BACK_END / "app" / "artifact",
        help="Directory for json, pkl, and ridge pipeline joblib",
    )
    parser.add_argument(
        "--shap-eval-rows",
        type=int,
        default=2500,
        help="Max test rows for SHAP (runtime/memory tradeoff)",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv_path or "").expanduser()
    if not csv_path.is_file():
        print("Set DIABETIC_DATA_PATH or pass path to diabetic_data.csv", file=sys.stderr)
        return 1

    try:
        import shap  # noqa: WPS433 — runtime optional until export
    except ImportError:
        print("Install shap: pip install shap", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_target(load_dataset(csv_path))
    df = apply_theoretical_quantile_filters(df)
    X_train, X_test, y_train, y_test, dropped_cols = build_features_and_split(df)
    preprocessor = make_preprocessor(X_train)
    pipeline = make_ridge_ml_practice_pipeline(preprocessor)
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, out_dir / PIPELINE_NAME)

    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)
    proba_all = pipeline.predict_proba(X_all)[:, 1].astype(np.float64)

    predictions_df = X_all.copy()
    predictions_df["hospital_mortality_proxy"] = y_all.values
    predictions_df["predicted_mortality_probability"] = proba_all
    predictions_df["predicted_positive_class"] = (
        proba_all >= DECISION_THRESHOLD
    ).astype(int)

    eval_n = min(args.shap_eval_rows, len(X_test))
    X_eval = X_test.iloc[:eval_n]
    y_eval = y_test.loc[X_eval.index]

    vals: np.ndarray | None = None
    names: list[str] = []
    shap_method: str

    try:
        bg_size = min(300, len(X_train))
        background = shap.sample(X_train, bg_size, random_state=42)
        explainer = shap.Explainer(pipeline.predict_proba, background)
        shap_explanation = explainer(X_eval)
        vals = _shap_values_positive_class(np.array(shap_explanation.values))
        raw_names = shap_explanation.feature_names
        if raw_names is not None and len(list(raw_names)) == vals.shape[1]:
            names = [str(x) for x in raw_names]
        else:
            names = [f"f{i}" for i in range(vals.shape[1])]
        if len(names) != vals.shape[1]:
            names = [f"f{i}" for i in range(vals.shape[1])]
        mean_signed = vals.mean(axis=0)
        shap_method = (
            "shap.Explainer(pipeline.predict_proba); summary = mean SHAP "
            "(positive class) over eval rows"
        )
    except Exception as exc:  # noqa: BLE001
        print("SHAP Explainer failed, using permutation_importance:", exc)
        from sklearn.inspection import permutation_importance

        n_pi = min(500, len(X_eval))
        X_pi = X_eval.iloc[:n_pi]
        y_pi = y_eval.iloc[:n_pi]
        pi = permutation_importance(
            pipeline,
            X_pi,
            y_pi,
            n_repeats=10,
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
        )
        mean_signed = pi.importances_mean
        names = list(X_pi.columns)
        vals = None
        shap_method = (
            "sklearn.inspection.permutation_importance (ROC-AUC) — "
            "SHAP Explainer unavailable for this pipeline"
        )

    shap_mortality = _mean_abs_shap_to_flutter_shap(mean_signed, names, top_k=20)

    json_payload = {
        "model_version": MODEL_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
        "shap_method": shap_method,
        "shap_mortality": shap_mortality,
        "n_training_rows": int(len(X_train)),
        "n_prediction_rows": int(len(predictions_df)),
        "n_shap_eval_rows": int(eval_n),
        "dropped_high_missing_columns": dropped_cols,
        "predictions_pkl_filename": PKL_NAME,
        "ridge_pipeline_joblib": PIPELINE_NAME,
    }

    json_path = out_dir / JSON_NAME
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    shap_eval: dict = {"feature_names": names, "n_rows": int(eval_n)}
    if vals is not None:
        shap_eval["values"] = vals

    pkl_payload = {
        "model_version": MODEL_VERSION,
        "predictions_df": predictions_df.reset_index(drop=True),
        "shap_eval": shap_eval,
    }
    joblib.dump(pkl_payload, out_dir / PKL_NAME)

    print("Wrote", json_path)
    print("Wrote", out_dir / PKL_NAME)
    print("Wrote", out_dir / PIPELINE_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
