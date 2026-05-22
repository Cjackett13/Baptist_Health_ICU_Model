#!/usr/bin/env python3
"""SHAP + permutation importance for LOS XGBoost regressor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from models.length_of_stay.config import BUNDLE_NAME, LABEL_EVAL, LABEL_TRAIN, REPO_ROOT
from models.length_of_stay.features import load_training_feature_spec, raw_matrix_for_pipeline
from models.length_of_stay.train import load_training_frame, split_by_manifest


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _raw_matrix(df: pd.DataFrame, included: list[str]) -> pd.DataFrame:
    return raw_matrix_for_pipeline(df, included)


def run_shap_audit(data_dir: Path, model_dir: Path, *, n_shap: int = 200) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir)
    pipe: Pipeline = joblib.load(model_dir / BUNDLE_NAME)
    prep = pipe.named_steps["prep"]
    xgb = pipe.named_steps["xgb"]
    included, _ = load_training_feature_spec(model_dir, data_dir)

    df = load_training_frame(data_dir)
    tr, _, te = split_by_manifest(df, data_dir)
    X_tr = _raw_matrix(tr, included)
    X_te = _raw_matrix(te, included)
    y_te_h = te[LABEL_EVAL].to_numpy()
    y_te_log = te[LABEL_TRAIN].to_numpy()

    X_tr_t = prep.transform(X_tr)
    X_te_t = prep.transform(X_te)
    feat_names = list(prep.feature_names_)

    pred_log = xgb.predict(X_te_t)
    pred_h = np.expm1(np.clip(pred_log, 0, None))
    mae = float(mean_absolute_error(y_te_h, pred_h))

    out: dict[str, Any] = {
        "task": "length_of_stay_regression",
        "n_test": int(len(X_te_t)),
        "mae_hours_test": mae,
        "feature_names": feat_names,
        "shap_available": False,
    }

    # Permutation importance (model-agnostic fallback)
    perm = permutation_importance(
        xgb,
        X_te_t,
        y_te_log,
        n_repeats=8,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    perm_rows = [
        {
            "feature": feat_names[i],
            "perm_mae_increase_log1p": float(-perm.importances_mean[i]),
            "perm_std": float(perm.importances_std[i]),
        }
        for i in np.argsort(-perm.importances_mean)
    ]
    out["permutation_importance"] = perm_rows[:20]

    fw = prep.get_feature_weights()
    weight_by_feat = {feat_names[i]: float(fw[i]) for i in range(len(feat_names))}
    top_weighted = [f for f, _ in sorted(weight_by_feat.items(), key=lambda x: -x[1])[:8]]
    top_perm = [r["feature"] for r in perm_rows[:8]]
    out["weights_sanity"] = {
        "top_xgb_feature_weights": top_weighted,
        "top_permutation_features": top_perm,
        "overlap_count": len(set(top_perm) & set(top_weighted)),
        "note": "feature_weights bias splits; permutation ranks impact on MAE (log1p).",
    }

    try:
        import shap

        bg_n = min(150, len(X_tr_t))
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_tr_t), size=bg_n, replace=False)
        background = X_tr_t.iloc[bg_idx]

        explainer = shap.TreeExplainer(xgb, data=background)
        n_use = min(n_shap, len(X_te_t))
        sample_idx = rng.choice(len(X_te_t), size=n_use, replace=False)
        X_sample = X_te_t.iloc[sample_idx]
        shap_values = explainer.shap_values(X_sample)
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)
        shap_rows = [
            {
                "feature": feat_names[i],
                "mean_abs_shap_log1p": float(mean_abs[i]),
                "mean_shap_log1p": float(sv[:, i].mean()),
            }
            for i in order
        ]
        out["shap_available"] = True
        out["shap_n_samples"] = int(n_use)
        out["mean_abs_shap"] = shap_rows[:25]
        # Example: median-LOS patient vs long-LOS patient (hours)
        med_idx = int(np.argmin(np.abs(y_te_h - np.median(y_te_h))))
        long_idx = int(np.argmax(y_te_h))
        for tag, idx in [("median_los_patient", med_idx), ("long_los_patient", long_idx)]:
            row = X_te_t.iloc[[idx]]
            sv1 = np.asarray(explainer.shap_values(row)).ravel()
            if len(sv1) == len(feat_names):
                base = float(explainer.expected_value)
                if isinstance(base, np.ndarray):
                    base = float(base.ravel()[0])
                contrib = [
                    {"feature": feat_names[i], "shap_log1p": float(sv1[i])}
                    for i in np.argsort(-np.abs(sv1))[:12]
                ]
                out[f"case_{tag}"] = {
                    "actual_los_hours": float(y_te_h[idx]),
                    "predicted_los_hours": float(pred_h[idx]),
                    "expected_value_log1p": base,
                    "top_contributions": contrib,
                }
    except ImportError:
        out["shap_note"] = "pip install shap for TreeExplainer; using XGBoost pred_contribs fallback"
        try:
            import xgboost as xgb_lib

            booster = xgb.get_booster()
            dmat = xgb_lib.DMatrix(
                X_te_t.to_numpy(dtype=np.float32),
                feature_names=feat_names,
            )
            contrib = booster.predict(dmat, pred_contribs=True)
            sv = contrib[:, :-1]
            mean_abs = np.abs(sv).mean(axis=0)
            order = np.argsort(-mean_abs)
            out["shap_available"] = True
            out["shap_backend"] = "xgboost_pred_contribs"
            out["mean_abs_shap"] = [
                {
                    "feature": feat_names[i],
                    "mean_abs_shap_log1p": float(mean_abs[i]),
                    "mean_shap_log1p": float(sv[:, i].mean()),
                }
                for i in order[:25]
            ]
        except Exception as exc:  # noqa: BLE001
            out["shap_fallback_error"] = str(exc)

    return out
