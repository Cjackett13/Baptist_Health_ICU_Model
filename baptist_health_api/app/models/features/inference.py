#!/usr/bin/env python3
"""
Baptist Health ICU — Vasopressor Model: Inference Entry Point

Loads the three production models and runs predictions on a feature matrix.

Models:
  binary_alert_model.pkl      — IsotonicCalibratedModel (XGBoost + isotonic)
                                 Output: alert probability + binary alert flag
  ordinal_count_model.pkl     — XGBClassifier (multi:softprob, 4-class)
                                 Output: predicted vasopressor count (0/1/2/3)
  high_severity_classifier.pkl — PlattXGB (XGBoost + Platt, count>=2 vs <2)
                                 Output: high-severity sub-score

Usage (CLI):
    python src/inference.py --split test

Usage (API):
    from src.inference import load_models, predict
    models = load_models()
    result = predict(X_features, models)
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Must be defined here for pickle.load to find these classes ─────────────────

class IsotonicCalibratedModel:
    """Binary alert model wrapper: XGBoost + isotonic calibration."""
    def __init__(self, base_model, iso_reg):
        self.base_model = base_model
        self.iso_reg    = iso_reg

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = np.clip(self.iso_reg.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


class PlattXGB:
    """OvR binary classifier wrapper: XGBoost + Platt (sigmoid) calibration."""
    def __init__(self, xgb_model, platt_lr, threshold):
        self.xgb_model = xgb_model
        self.platt_lr  = platt_lr
        self.threshold = threshold

    def predict_proba(self, X):
        raw = self.xgb_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.platt_lr.predict_proba(raw)[:, 1]
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X, decision_threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= decision_threshold).astype(int)


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
ARTS_DIR = BASE_DIR / "artifacts"
FEATS_DIR = BASE_DIR / "data" / "features"

ALERT_THRESHOLD = 0.1882   # F1-optimal on val; converges with F2/F1.5


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models() -> dict:
    """Load all three production models and the feature manifest."""
    with open(ARTS_DIR / "feature_manifest.json") as f:
        manifest = json.load(f)

    with open(ARTS_DIR / "binary_alert_model.pkl", "rb") as f:
        binary_model = pickle.load(f)

    with open(ARTS_DIR / "ordinal_count_model.pkl", "rb") as f:
        ordinal_model = pickle.load(f)

    with open(ARTS_DIR / "high_severity_classifier.pkl", "rb") as f:
        severity_model = pickle.load(f)

    with open(ARTS_DIR / "shap_display_features.json") as f:
        shap_display = json.load(f)

    return {
        "binary":        binary_model,
        "ordinal":       ordinal_model,
        "high_severity": severity_model,
        "feature_cols":  manifest["feature_columns"],
        "shap_display":  shap_display,
    }


# ── Core prediction ────────────────────────────────────────────────────────────

def predict(X: np.ndarray, models: dict, threshold: float = ALERT_THRESHOLD) -> dict:
    """
    Run all three models on a feature matrix.

    Args:
        X: (N, 360) float array, columns ordered per feature_manifest.json
        models: dict returned by load_models()
        threshold: binary alert threshold (default 0.1882)

    Returns dict with keys:
        alert_proba     — calibrated alert probability per row  (N,)
        alert_flag      — binary alert (1 = vasopressor likely needed) (N,)
        ordinal_pred    — predicted vasopressor count 0-3           (N,)
        ordinal_proba   — softmax class probabilities              (N, 4)
        high_severity   — P(count>=2) calibrated score            (N,)
    """
    alert_proba   = models["binary"].predict_proba(X)[:, 1]
    alert_flag    = (alert_proba >= threshold).astype(int)

    ordinal_proba = models["ordinal"].predict_proba(X)
    ordinal_pred  = np.argmax(ordinal_proba, axis=1)

    high_severity = models["high_severity"].predict_proba(X)[:, 1]

    return {
        "alert_proba":   alert_proba,
        "alert_flag":    alert_flag,
        "ordinal_pred":  ordinal_pred,
        "ordinal_proba": ordinal_proba,
        "high_severity": high_severity,
    }


def predict_from_df(df: pd.DataFrame, models: dict, threshold: float = ALERT_THRESHOLD) -> pd.DataFrame:
    """
    Convenience wrapper: takes a DataFrame with the production feature columns,
    returns predictions merged back with ENCOUNTER_ID and HOUR_FROM_ADMIT.
    """
    feat_cols = models["feature_cols"]
    X = df[feat_cols].astype(float).values
    preds = predict(X, models, threshold)

    out = pd.DataFrame({
        "ENCOUNTER_ID":    df["ENCOUNTER_ID"].values,
        "HOUR_FROM_ADMIT": df["HOUR_FROM_ADMIT"].values,
        "alert_proba":     preds["alert_proba"].round(4),
        "alert_flag":      preds["alert_flag"],
        "ordinal_pred":    preds["ordinal_pred"],
        "high_severity":   preds["high_severity"].round(4),
    })
    for c in range(4):
        out[f"ordinal_p{c}"] = preds["ordinal_proba"][:, c].round(4)

    return out


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run vasopressor model inference")
    parser.add_argument("--split",    choices=["train", "val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=ALERT_THRESHOLD)
    parser.add_argument("--out",      type=str,   default=None,
                        help="Output parquet path (default: artifacts/inference_<split>.parquet)")
    args = parser.parse_args()

    print("=" * 65)
    print("  Baptist Health ICU — Vasopressor Model Inference")
    print("=" * 65)

    print("\n[1/3] Loading models...")
    models = load_models()
    print(f"  Feature columns: {len(models['feature_cols'])}")
    print(f"  Alert threshold: {args.threshold}")

    feat_path = FEATS_DIR / f"{args.split}_features_v3.parquet"
    print(f"\n[2/3] Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)
    print(f"  Rows: {len(df):,}")

    print(f"\n[3/3] Running inference...")
    results = predict_from_df(df, models, threshold=args.threshold)

    n_alerts = int(results["alert_flag"].sum())
    print(f"  Alerts fired:    {n_alerts:,} / {len(results):,}  "
          f"({n_alerts/len(results)*100:.1f}%)")

    if "VASOPRESSOR_NEEDED_24H" in df.columns:
        y_true = df["VASOPRESSOR_NEEDED_24H"].values
        tp = int(((results["alert_flag"] == 1) & (y_true == 1)).sum())
        fp = int(((results["alert_flag"] == 1) & (y_true == 0)).sum())
        fn = int(((results["alert_flag"] == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        print(f"  Sensitivity:     {sens:.4f}")
        print(f"  Precision:       {prec:.4f}")
        print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}")

    out_path = args.out or str(ARTS_DIR / f"inference_{args.split}.parquet")
    results.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
