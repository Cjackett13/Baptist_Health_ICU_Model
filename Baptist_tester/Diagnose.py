#!/usr/bin/env python3
"""
Clinical ML checklist diagnostics for the **mortality XGBoost** pipeline
(``Frontend/lib/models/mortality_model.py`` + saved joblib bundle).

  cd Baptist_Health_ICU_Model   # repo root
  python3 Baptist_tester/Diagnose.py --data-dir Baptist_tester/synth_cs_data

Writes ``diagnose_mortality_report.json`` under ``--out-dir`` (default: same as data-dir).
Exit 0 if all automated *hard* checks pass; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

_REPO = Path(__file__).resolve().parent.parent


def _add_models_path() -> None:
    p = _REPO / "Frontend" / "lib" / "models"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _register_mortality_unpickle_aliases(mm: Any) -> None:
    """Bundles may pickle nested estimators as ``__main__.*``; map them onto ``mortality_model``."""
    main = sys.modules.get("__main__")
    if main is None:
        return
    for name in ("TemperatureScaledBinaryCalibrator", "AveragedBinaryCalibrators"):
        if hasattr(main, name):
            continue
        if hasattr(mm, name):
            setattr(main, name, getattr(mm, name))


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _find_inner_xgb_pipeline(final_est: Any) -> Pipeline | None:
    """Unwrap TemperatureScaledBinaryCalibrator / AveragedBinaryCalibrators / CalibratedClassifierCV."""
    from xgboost import XGBClassifier

    def from_cal(cc: CalibratedClassifierCV) -> Pipeline | None:
        try:
            cc0 = cc.calibrated_classifiers_[0]
        except Exception:
            return None
        est = getattr(cc0, "estimator", None)
        if isinstance(est, Pipeline) and "xgb" in est.named_steps:
            return est
        return None

    cur: Any = final_est
    for _ in range(12):
        if cur is None:
            return None
        if isinstance(cur, Pipeline) and "xgb" in cur.named_steps:
            x = cur.named_steps["xgb"]
            if isinstance(x, XGBClassifier):
                return cur
        if isinstance(cur, CalibratedClassifierCV):
            pl = from_cal(cur)
            if pl is not None:
                return pl
        if hasattr(cur, "calibrators"):
            cals = getattr(cur, "calibrators") or []
            if cals:
                cur = cals[0]
                continue
        if hasattr(cur, "calibrated_estimator"):
            cur = getattr(cur, "calibrated_estimator")
            continue
        break
    return None


def _label_from_parquet_docs() -> str:
    return (
        "Training label ``died`` = 1 if ``person.parquet`` / encounter-linked "
        "``DECEASED_DT_TM`` is non-null for the patient (see mortality_model / DuckDB build); "
        "0 otherwise. Binary outcome."
    )


def run_checks(
    *,
    data_dir: Path,
    model_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    _add_models_path()
    import mortality_model as mm  # noqa: PLC0415

    _register_mortality_unpickle_aliases(mm)
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir)
    out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = model_dir / "xgb_mortality_model_meta.json"
    bundle_path = model_dir / "xgb_mortality_pipeline.joblib"
    report: dict[str, Any] = {"data_dir": str(data_dir), "model_dir": str(model_dir)}

    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # --- Category 1: data & labels ---
    raw = mm.build_patient_frame(data_dir)
    report["category_1_data"] = {}

    X_train, X_test, y_train, y_test, pid_train, pid_test, prep_diag = mm.prepare_train_test_features(
        data_dir,
        pct_lo=float(meta.get("outlier_pct_lo", 2.0)),
        pct_hi=float(meta.get("outlier_pct_hi", 98.0)),
        random_state=int(meta.get("split_seed", meta.get("split_random_state", 42))),
    )

    pr_tr = float(y_train.mean())
    pr_te = float(y_test.mean())
    report["category_1_data"]["1_class_distribution"] = {
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "train_positive_rate": pr_tr,
        "test_positive_rate": pr_te,
        "delta_positive_rate_abs": float(abs(pr_tr - pr_te)),
        "consistent_stratified_split": bool(abs(pr_tr - pr_te) < 0.08),
        "note": "Stratified on label; small drift can occur after joint fence drops rows.",
    }

    n_pos_tr = int((y_train == 1).sum())
    n_neg_tr = int((y_train == 0).sum())
    ratio = float(n_neg_tr / max(n_pos_tr, 1))
    spw = float(meta.get("scale_pos_weight", float("nan")))
    report["category_1_data"]["2_class_imbalance"] = {
        "positive_rate_train": pr_tr,
        "imbalance_exists_positive_lt_20pct": bool(pr_tr < 0.20),
        "negative_over_positive_ratio_train": ratio,
        "scale_pos_weight_in_meta": spw,
        "scale_pos_weight_matches_neg_over_pos": bool(np.isclose(spw, ratio, rtol=1e-5, atol=1e-8)),
        "mitigation_documented": "XGBoost scale_pos_weight from train counts; stratified split; no SMOTE in code path.",
    }

    ov = mm.verify_patient_train_test_no_id_overlap(pid_train, pid_test)
    report["category_1_data"]["3_train_test_split"] = {
        "grain": "one_row_per_PERSON_ID",
        "patient_disjoint": ov,
        "percentile_fences_fit_on": prep_diag.get("percentile_fences_fit_on"),
        "split_order": prep_diag.get("split_and_scaling_order"),
    }

    nan_died = int(raw["died"].isna().sum())
    lab_vals = pd.unique(raw["died"].dropna())
    uniq_died = {int(v) for v in lab_vals if pd.notna(v)}
    report["category_1_data"]["4_label"] = {
        "label_column": "died",
        "nan_labels_in_raw_frame": nan_died,
        "unique_non_nan_label_values": sorted(uniq_died),
        "binary_0_1": uniq_died <= {0, 1},
        "both_classes_in_test_after_fence": bool(
            (y_test == 0).any() and (y_test == 1).any()
        ),
        "label_definition": _label_from_parquet_docs(),
    }

    # --- Category 2: features & leakage ---
    feats = list(mm.FEATURE_COLUMNS)
    report["category_2_features"] = {
        "5_feature_list": feats,
        "5_prediction_time_note": (
            "Features are encounter-aggregates / hourly SCAI summaries materialized in "
            "duckdb_patient_features — review each column vs intended prediction time; "
            "this script does not prove absence of subtle leakage (e.g. residual features using full cohort)."
        ),
        "6_leakage": {
            "died_in_feature_columns": "died" in feats,
            "prep_diag_patient_disjoint": prep_diag.get("patient_id_train_test_disjoint"),
            "duckdb_residual_warning": (
                "DuckDB residual features may use statistics over all patients in the sidecar — "
                "see run_mortality_integrity_audit / meta notes."
            ),
        },
    }

    if not bundle_path.is_file():
        report["error"] = f"missing_bundle:{bundle_path}"
        return report

    blob = joblib.load(bundle_path)
    pipe = blob["pipeline"]
    feat_saved: list[str] = list(blob.get("feature_names", feats))
    scale_audit = mm.inspect_xgb_mortality_pipeline_for_scaling(pipe)
    miss_tr = {c: float(X_train[c].isna().mean()) for c in feats}
    report["category_2_features"]["7_scaling"] = scale_audit
    report["category_2_features"]["8_missing_value_fraction_train"] = miss_tr

    inner = _find_inner_xgb_pipeline(pipe)
    gain: dict[str, float] = {}
    if inner is not None:
        xgb = inner.named_steps["xgb"]
        imp = getattr(xgb, "feature_importances_", None)
        if imp is not None and len(imp) == len(feat_saved):
            gain = {feat_saved[i]: float(imp[i]) for i in range(len(feat_saved))}
            gain = dict(sorted(gain.items(), key=lambda kv: -kv[1])[:12])
    report["category_2_features"]["9_feature_importance_gain_top"] = gain

    # --- Category 3: metrics ---
    proba = pipe.predict_proba(X_test[feat_saved])[:, 1]
    y_te = np.asarray(y_test).astype(int)
    baseline_pr = float(y_te.mean())
    auroc = float(roc_auc_score(y_te, proba))
    auprc = float(average_precision_score(y_te, proba))
    brier = float(brier_score_loss(y_te, proba))
    report["category_3_metrics"] = {
        "10_primary_metrics": {
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "note": "For rare positives, AUPRC is primary; AUROC secondary; Brier for calibration.",
        },
        "11_12_interpretation": {
            "auc_pr_over_baseline": float(auprc / max(baseline_pr, 1e-12)),
            "baseline_positive_rate": baseline_pr,
        },
        "13_brier": {
            "test_brier": brier,
            "meta_test_brier": float(meta.get("test_brier", float("nan"))),
        },
        "14_cv_stability_note": (
            "Hyperparameter search uses 4-fold CV on train; full nested std across repeated "
            "retrains is not computed here — see tune_cv_* in meta if present."
        ),
    }

    # --- Category 4: fairness (limited) ---
    report["category_4_bias"] = {
        "15_17_subgroup_note": (
            "``demo_profile_bucket`` encodes age/weight/sex bands with capped XGBoost feature weight; "
            "race is not modeled. Subgroup fairness by race requires a separate evaluation join."
        ),
    }

    # --- Category 5: sanity ---
    c5: dict[str, Any] = {}
    p1 = pipe.predict_proba(X_test[feat_saved].iloc[:3])[0, 1]
    p2 = pipe.predict_proba(X_test[feat_saved].iloc[:3])[0, 1]
    det_ok = bool(np.isclose(p1, p2, rtol=0, atol=0))

    row_nan = pd.DataFrame([{c: np.nan for c in feat_saved}])
    try:
        p_nan = float(pipe.predict_proba(row_nan)[0, 1])
        nan_ok = np.isfinite(p_nan)
    except Exception as exc:  # noqa: BLE001
        p_nan = None
        nan_ok = False
        c5["edge_nan_error"] = str(exc)

    median_row = X_test[feat_saved].median(numeric_only=True).to_dict()
    hi_row = {c: float(median_row.get(c, 0) or 0) * 10.0 + 1.0 for c in feat_saved}
    try:
        p_hi = float(pipe.predict_proba(pd.DataFrame([hi_row]))[0, 1])
        hi_ok = np.isfinite(p_hi)
    except Exception:
        p_hi = None
        hi_ok = False

    # Monotonic: bump current_scai and scai_prop_ge3 if present
    mono: dict[str, Any] = {}
    if "current_scai" in feat_saved and "scai_prop_ge3" in feat_saved:
        b = median_row.copy()
        b2 = {**{k: (v if pd.notna(v) else 0.0) for k, v in b.items()}, "current_scai": 1.0, "scai_prop_ge3": 0.5}
        b3 = {**{k: (v if pd.notna(v) else 0.0) for k, v in b.items()}, "current_scai": 3.0, "scai_prop_ge3": 0.9}
        try:
            p_lo = float(pipe.predict_proba(pd.DataFrame([b2]))[0, 1])
            p_hi2 = float(pipe.predict_proba(pd.DataFrame([b3]))[0, 1])
            mono = {
                "current_scai_1_scai_prop_0_5_prob": p_lo,
                "current_scai_3_scai_prop_0_9_prob": p_hi2,
                "higher_risk_inputs_give_higher_or_equal_prob": bool(p_hi2 >= p_lo - 1e-6),
            }
        except Exception as exc:  # noqa: BLE001
            mono = {"error": str(exc)}

    # Threshold sweep (test)
    fpr, tpr, thr = roc_curve(y_te, proba)
    rows = []
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        rows.append(
            {
                "threshold": t,
                "sensitivity": float(sens),
                "specificity": float(spec),
                "ppv": float(ppv),
                "npv": float(npv),
            }
        )

    c5.update(
        {
            "20_determinism_same_slice": {"identical_probs": det_ok},
            "21_edge_cases": {
                "all_nan_prob": p_nan,
                "all_nan_finite": nan_ok,
                "extreme_scaled_row_prob": p_hi,
                "extreme_finite": hi_ok,
            },
            "18_19_monotonicity_probe": mono,
            "22_threshold_analysis_test_sample": rows,
        }
    )
    report["category_5_sanity"] = c5

    # --- Category 6–7 ---
    majority_baseline = float(max(baseline_pr, 1.0 - baseline_pr))
    report["category_6_clinical"] = {
        "23_alert_rate": "Not computed (needs alert policy per patient-time).",
        "24_lead_time": "Not applicable at patient-summary grain without hourly scoring path here.",
        "25_baselines": {
            "test_positive_rate": baseline_pr,
            "brier_always_predict_train_death_rate": float(
                meta.get("test_brier_if_always_pred_train_death_rate", float("nan"))
            ),
            "model_brier": brier,
            "auroc_vs_random_0_5": auroc > 0.5,
        },
    }

    integrity = mm.run_mortality_integrity_audit(data_dir, model_dir, meta_override=meta if meta else None)
    report["integrity_audit_summary"] = {
        "all_critical_checks_passed": integrity.get("all_critical_checks_passed"),
        "patient_disjoint": integrity.get("patient_disjoint_report"),
        "scaling": integrity.get("scaling_leakage_and_transform_policy"),
        "skew_f_summary": integrity.get("skew_and_f_test_summary"),
    }

    report["category_7_deployment"] = {
        "27_reproducibility": {
            "split_seed_in_meta": meta.get("split_seed", meta.get("split_random_state")),
            "calibration_dual_seed": meta.get("calibration_dual_seed_xgb"),
        },
        "28_train_serve": (
            "Training uses same Pipeline as saved joblib: log1p/sqrt then XGB then calibration wrapper. "
            "Serve must use identical column order and raw feature semantics."
        ),
        "29_model_card_pointers": [
            "See xgb_mortality_model_meta.json for metrics, calibration, fences.",
            "Run: python3 Frontend/lib/models/mortality_model.py --integrity-audit",
        ],
        "30_monitoring": "Define drift metrics and retrain triggers outside this script.",
    }

    report["overall_pass"] = bool(
        report["category_1_data"]["3_train_test_split"]["patient_disjoint"].get("disjoint")
        and report["category_1_data"]["4_label"]["both_classes_in_test_after_fence"]
        and not report["category_2_features"]["6_leakage"]["died_in_feature_columns"]
        and scale_audit.get("no_standard_scaler_on_xgb_path")
        and det_ok
        and integrity.get("all_critical_checks_passed", False)
    )

    out_path = out_dir / "diagnose_mortality_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["_written"] = str(out_path)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Mortality XGBoost checklist diagnostics.")
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument("--model-dir", type=Path, default=None, help="Default: same as --data-dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: same as --data-dir")
    args = ap.parse_args()
    model_dir = args.model_dir or args.data_dir
    out_dir = args.out_dir or args.data_dir
    try:
        r = run_checks(data_dir=args.data_dir, model_dir=model_dir, out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in r.items() if k != "_written"}, indent=2, default=str))
    print(f"\nWrote: {r.get('_written')}", file=sys.stderr)
    return 0 if r.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
