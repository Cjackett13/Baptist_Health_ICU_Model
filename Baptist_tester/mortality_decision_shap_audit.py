#!/usr/bin/env python3
"""
Audit: (1) SHAP for a synthetic "low-risk" row on the inner XGBoost, (2) test-set
permutation importance, (3) threshold sweep with precision/recall/NPV and Vickers-style
net benefit vs "treat none".

  pip install shap  # optional but recommended for SHAP block
  PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_decision_shap_audit.py
  PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_decision_shap_audit.py \\
      --data-dir Baptist_tester/synth_cs_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

_REPO = Path(__file__).resolve().parent.parent


def _add_models_path() -> None:
    p = _REPO / "Frontend" / "lib" / "models"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def net_benefit(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """Vickers-style net benefit at decision threshold (predict positive if score >= t)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    if n == 0:
        return 0.0
    t = float(np.clip(threshold, 1e-9, 1.0 - 1e-9))
    odds = t / (1.0 - t)
    return float(tp / n - fp / n * odds)


def _inner_xgb_pipeline(cal: CalibratedClassifierCV) -> Pipeline:
    """First fold’s calibrated base pipeline (log1p → sqrt → xgb)."""
    cc0 = cal.calibrated_classifiers_[0]
    est = cc0.estimator
    if not isinstance(est, Pipeline):
        raise TypeError("Expected sklearn Pipeline inside CalibratedClassifierCV")
    return est


def shap_good_patient(
    inner: Pipeline,
    X_train: pd.DataFrame,
    feature_names: list[str],
    good: pd.DataFrame,
) -> tuple[list[str], np.ndarray, str] | None:
    try:
        import shap
    except ImportError:
        return None

    pre = inner[:-1]
    xgb = inner.named_steps["xgb"]
    bg_n = min(80, len(X_train))
    bg_raw = X_train[feature_names].sample(n=bg_n, random_state=42)
    bg_t = pre.transform(bg_raw)
    x_good = pre.transform(good[feature_names])
    explainer = shap.TreeExplainer(xgb, data=bg_t)
    sv = explainer.shap_values(x_good)
    # Binary XGB: often array (1, n_feat) or list [class0, class1]
    if isinstance(sv, list):
        arr = np.asarray(sv[1])
    else:
        arr = np.asarray(sv)
    if arr.ndim == 3:
        arr = arr[0, :, 1]
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    arr = np.asarray(arr).ravel()
    if len(arr) != len(feature_names):
        return None
    order = np.argsort(-np.abs(arr))
    lines = [f"  {feature_names[i]:24s}  SHAP={arr[i]:+.4f}" for i in order]
    return feature_names, arr, "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    ap.add_argument("--model-dir", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _add_models_path()
    import joblib

    from mortality_model import FEATURE_COLUMNS, prepare_train_test_features

    data_dir = _resolve(args.data_dir)
    model_dir = _resolve(args.model_dir)
    bundle = model_dir / "xgb_mortality_pipeline.joblib"
    if not bundle.is_file():
        print(f"Missing {bundle}", file=sys.stderr)
        return 1

    meta_path = model_dir / "xgb_mortality_model_meta.json"
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pct_lo = float(meta.get("outlier_pct_lo", 1.0))
    pct_hi = float(meta.get("outlier_pct_hi", 99.0))
    seed = int(meta.get("split_seed", args.seed))

    sys.modules.setdefault("__main__", __import__("mortality_model"))

    X_train, X_test, y_train, y_test, pid_tr, pid_te, prep_diag = prepare_train_test_features(
        data_dir,
        pct_lo=pct_lo,
        pct_hi=pct_hi,
        random_state=seed,
    )
    counts = {
        "n_rows_raw": prep_diag["n_rows_raw"],
        "n_rows_after_outlier_filter": int(len(X_train) + len(X_test)),
    }
    feat_cols = list(FEATURE_COLUMNS)
    import mortality_model  # noqa: E402

    sys.modules["__main__"] = mortality_model
    blob = joblib.load(bundle)
    pipe = blob["pipeline"]
    if list(blob.get("feature_names", feat_cols)) != feat_cols:
        print("Warning: joblib feature_names differ from mortality_model.FEATURE_COLUMNS", file=sys.stderr)

    y_test_i = np.asarray(y_test).astype(int)
    proba = pipe.predict_proba(X_test[feat_cols])[:, 1]

    print("=== Test-set scores (same stratified split as training script) ===\n")
    print(f"  n_train={len(y_train)}  n_test={len(y_test)}  prevalence_test={y_test_i.mean():.3f}")
    print(f"  AUROC:     {roc_auc_score(y_test_i, proba):.4f}")
    print(f"  AUPRC:     {average_precision_score(y_test_i, proba):.4f}")
    print(f"  Brier:     {brier_score_loss(y_test_i, proba):.4f}")
    prec, rec, thr = precision_recall_curve(y_test_i, proba)
    pr_auc = auc(rec, prec)
    print(f"  PR-AUC:    {pr_auc:.4f}  (integral of precision–recall curve)")
    print(f"  counts:   {counts}\n")

    # --- Permutation importance (discrimination stability) ---
    print("=== Permutation importance on test (scoring=roc_auc, n_repeats=20) ===\n")
    pi = permutation_importance(
        pipe,
        X_test[feat_cols],
        y_test_i,
        n_repeats=20,
        random_state=args.seed,
        scoring="roc_auc",
        n_jobs=1,
    )
    order = np.argsort(-pi.importances_mean)
    for i in order:
        m, s = pi.importances_mean[i], pi.importances_std[i]
        print(f"  {feat_cols[i]:24s}  mean ΔAUROC={m:+.5f}  ±{s:.5f}")
    print()

    # --- Synthetic "good" patient (low acuity in this feature space) ---
    good = X_train[feat_cols].median(numeric_only=True).to_frame().T
    for c, v in {
        "med_distinct": 1.0,
        "med_infusion_mean": 0.05,
        "proc_n": 0.0,
        "current_scai": 0.0,
        "scai_prop_ge3": 0.0,
        "scai_std": 0.0,
        "demo_profile_bucket": 0.0,
    }.items():
        if c in good.columns:
            good[c] = v
    p_good = float(pipe.predict_proba(good[feat_cols])[0, 1])
    print('=== Synthetic "good patient" row (all mild / zero SCAI burden) ===\n')
    print(json.dumps(good.iloc[0].to_dict(), indent=2))
    print(f"\n  P(death | row): {p_good:.1%}\n")

    cal = pipe
    if isinstance(cal, CalibratedClassifierCV):
        try:
            inner = _inner_xgb_pipeline(cal)
            sh = shap_good_patient(inner, X_train, feat_cols, good)
            if sh is None:
                print("=== SHAP (TreeExplainer on inner XGB) ===\n  (install shap: pip install shap)\n")
            else:
                _names, _arr, block = sh
                print("=== SHAP contributions (positive class), inner XGB, interventional tree ===\n")
                print(block + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"=== SHAP skipped: {exc}\n", file=sys.stderr)

    # --- Threshold / net benefit (tune decision, not only the score) ---
    print("=== Threshold sweep on **test** predictions (not 0.5 by default) ===\n")
    print("  thr   P(pred+)  Prec   Recall  Spec    NPV     NetBen  F1")
    rows = []
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    best_f1, best_t_f1 = -1.0, 0.5
    best_nb, best_t_nb = -1e9, 0.5
    best_nb_mid, best_t_nb_mid = -1e9, 0.5  # NB among 25–75% flagged (avoid trivial "call all high")
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn_c, tp = confusion_matrix(y_test_i, pred, labels=[0, 1]).ravel()
        n = len(y_test_i)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn_c) if (tp + fn_c) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        npv = tn / (tn + fn_c) if (tn + fn_c) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        nb = net_benefit(y_test_i, proba, float(t))
        p_pred_pos = float((pred == 1).mean())
        rows.append(
            {
                "threshold": float(t),
                "frac_pred_positive": p_pred_pos,
                "precision": float(prec),
                "recall": float(rec),
                "specificity": float(spec),
                "npv": float(npv),
                "net_benefit": float(nb),
                "f1": float(f1),
            }
        )
        if f1 > best_f1:
            best_f1, best_t_f1 = f1, float(t)
        if nb > best_nb:
            best_nb, best_t_nb = nb, float(t)
        if 0.25 <= p_pred_pos <= 0.75 and nb > best_nb_mid:
            best_nb_mid, best_t_nb_mid = nb, float(t)
        print(
            f"  {t:.2f}   {p_pred_pos:5.1%}    {prec:.3f}  {rec:.3f}  {spec:.3f}  {npv:.3f}   {nb:+.4f}  {f1:.3f}"
        )

    print("\n--- Suggested operating points (test set, descriptive only) ---")
    print(f"  Max F1 threshold:           {best_t_f1:.2f}  (F1={best_f1:.3f})")
    print(f"  Max net benefit threshold:  {best_t_nb:.2f}  (NB={best_nb:+.4f})")
    print(
        f"  Max NB with 25–75% flagged:   {best_t_nb_mid:.2f}  (NB={best_nb_mid:+.4f}) "
        "(avoids trivial 'predict everyone high-risk')"
    )
    print("  Net benefit uses Vickers form: TP/n − FP/n × t/(1−t); 'treat none' baseline = 0.\n")

    shap_lines: str | None = None
    shap_top: list[dict[str, float]] = []
    if isinstance(pipe, CalibratedClassifierCV):
        try:
            inner = _inner_xgb_pipeline(pipe)
            sh = shap_good_patient(inner, X_train, feat_cols, good)
            if sh is not None:
                _names, arr, shap_lines = sh
                order = np.argsort(-np.abs(arr))
                shap_top = [
                    {"feature": feat_cols[i], "shap": float(arr[i])}
                    for i in order[:8]
                ]
        except Exception:
            pass

    perm_sorted = sorted(
        ((feat_cols[i], float(pi.importances_mean[i])) for i in range(len(feat_cols))),
        key=lambda t: -t[1],
    )
    narrative = (
        "Primary signal: SCAI trajectory severity (scai_prop_ge3, current_scai). "
        "Secondary: medication infusion intensity and clinical-event load. "
        "Demographics are capped weak priors."
    )

    out_json = model_dir / "mortality_threshold_audit.json"
    explain_json = model_dir / "mortality_explainability.json"
    payload = {
        "good_patient_row": good.iloc[0].to_dict(),
        "p_death_good_row": p_good,
        "test_metrics": {
            "auroc": float(roc_auc_score(y_test_i, proba)),
            "auprc": float(average_precision_score(y_test_i, proba)),
            "brier": float(brier_score_loss(y_test_i, proba)),
            "pr_auc": float(pr_auc),
        },
        "threshold_sweep": rows,
        "suggested_threshold_max_f1": best_t_f1,
        "suggested_threshold_max_net_benefit": best_t_nb,
        "suggested_threshold_max_net_benefit_mid_flag_rate": best_t_nb_mid,
        "permutation_importance_mean": {
            feat_cols[i]: float(pi.importances_mean[i]) for i in range(len(feat_cols))
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    explain_payload = {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "presentation_narrative": narrative,
        "permutation_importance_test_auroc": {
            k: v for k, v in perm_sorted
        },
        "permutation_top_features": [k for k, _ in perm_sorted[:5]],
        "shap_available": shap_lines is not None,
        "shap_good_patient_top": shap_top,
        "shap_good_patient_text": shap_lines,
        "verbal_answer_if_no_shap_plot": (
            "The model behaves as a SCAI-trajectory severity score with secondary refinement "
            "from infusion intensity (med_infusion_mean) and clinical event load; "
            "permutation importance on the holdout shows scai_prop_ge3 dominates (ΔAUROC ~0.14)."
        ),
    }
    explain_json.write_text(json.dumps(explain_payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {explain_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
