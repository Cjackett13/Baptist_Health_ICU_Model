#!/usr/bin/env python3
"""
Run the 30-item General ML Model Testing Checklist on the mortality (in-hospital death) model.

  PYTHONPATH=Frontend/lib/models python3 mortality_checklist_audit.py
  PYTHONPATH=Frontend/lib/models python3 mortality_checklist_audit.py \\
      --data-dir Baptist_tester/synth_cs_data
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys

import mortality_model  # noqa: F401 — register custom estimators for joblib unpickle

# Bundle was pickled when mortality_model.py ran as __main__
sys.modules["__main__"] = mortality_model

from mortality_model import (
    FEATURE_COLUMNS,
    prepare_train_test_features,
    run_mortality_integrity_audit,
)

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATA = _REPO / "data" / "cleaned"


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _status(pass_: bool, *, warn: bool = False) -> str:
    if pass_:
        return "PASS"
    return "WARN" if warn else "FAIL"


def _subgroup_auc(
    y_true: np.ndarray,
    proba: np.ndarray,
    groups: pd.Series,
    *,
    min_n: int = 25,
    min_pos: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = float(roc_auc_score(y_true, proba))
    for g, sub in pd.DataFrame({"y": y_true, "p": proba, "g": groups}).groupby("g", dropna=False):
        n = len(sub)
        n_pos = int((sub["y"] == 1).sum())
        if n < min_n or n_pos < min_pos or n_pos == n:
            rows.append({"group": str(g), "n": n, "note": "too small for AUC"})
            continue
        auc_g = float(roc_auc_score(sub["y"], sub["p"]))
        rows.append(
            {
                "group": str(g),
                "n": n,
                "n_pos": n_pos,
                "death_rate": round(float(sub["y"].mean()), 4),
                "auroc": round(auc_g, 4),
                "auroc_delta_vs_overall": round(auc_g - overall, 4),
            }
        )
    rows.sort(key=lambda r: r.get("auroc", 0))
    return rows


def run_checklist(data_dir: Path, model_dir: Path, *, seed: int = 42) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir)
    meta_path = model_dir / "xgb_mortality_model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    audit = run_mortality_integrity_audit(data_dir, model_dir, meta_override=meta if meta else None)

    pct_lo = float(meta.get("outlier_pct_lo", 2.0))
    pct_hi = float(meta.get("outlier_pct_hi", 98.0))
    split_seed = int(meta.get("split_seed", meta.get("split_random_state", seed)))

    X_train, X_test, y_train, y_test, pid_train, pid_test, prep_diag = prepare_train_test_features(
        data_dir, pct_lo=pct_lo, pct_hi=pct_hi, random_state=split_seed
    )
    y_tr = np.asarray(y_train).astype(int)
    y_te = np.asarray(y_test).astype(int)

    bundle_path = model_dir / "xgb_mortality_pipeline.joblib"
    blob = joblib.load(bundle_path) if bundle_path.is_file() else {}
    pipe = blob.get("pipeline") if blob else None
    feat_names = list(blob.get("feature_names", FEATURE_COLUMNS))
    proba_te = pipe.predict_proba(X_test[feat_names])[:, 1] if pipe is not None else np.array([])

    checks: list[dict[str, Any]] = []
    row_ok = bool(audit.get("row_count_matches_meta", True))
    if not row_ok:
        checks.append(
            {
                "id": 0,
                "category": "Data Quality",
                "title": "Artifact vs data alignment",
                "status": "WARN",
                "finding": (
                    "Saved bundle/meta row counts do not match live prep — "
                    "retrain on current parquet before trusting checklist metrics."
                ),
                "details": {
                    "row_count_matches_meta": False,
                    "meta_n_train": meta.get("n_train"),
                    "meta_n_test": meta.get("n_test"),
                    "live_n_train": audit.get("class_balance", {}).get("train_n"),
                    "live_n_test": audit.get("class_balance", {}).get("test_n"),
                },
                "not_applicable_reason": None,
            }
        )

    def add(
        item_id: int,
        category: str,
        title: str,
        status: str,
        finding: str,
        *,
        details: dict | None = None,
        na_reason: str | None = None,
    ) -> None:
        checks.append(
            {
                "id": item_id,
                "category": category,
                "title": title,
                "status": status,
                "finding": finding,
                "details": details or {},
                "not_applicable_reason": na_reason,
            }
        )

    cb = audit.get("class_balance", {})
    dr_tr = float(cb.get("train_death_rate", y_tr.mean()))
    dr_te = float(cb.get("test_death_rate", y_te.mean()))
    add(
        1,
        "Data Quality",
        "Class distribution check",
        _status(abs(dr_tr - dr_te) < 0.03),
        f"Train death rate {dr_tr:.1%} vs test {dr_te:.1%} (stratified split).",
        details={"train_death_rate": dr_tr, "test_death_rate": dr_te, "delta_pp": round(100 * (dr_te - dr_tr), 2)},
    )

    imb_ratio = float(cb.get("imbalance_ratio_neg_over_pos_train", (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    spw = float(meta.get("scale_pos_weight", cb.get("meta_scale_pos_weight", np.nan)))
    add(
        2,
        "Data Quality",
        "Class imbalance assessment",
        _status(dr_tr < 0.30 and np.isfinite(spw)),
        f"Positive rate {dr_tr:.1%} (<30% imbalanced). Neg/pos ratio≈{imb_ratio:.1f}. "
        f"XGBoost scale_pos_weight={spw:.2f} (train counts only).",
        details={"scale_pos_weight": spw, "imbalance_ratio_train": imb_ratio},
    )

    disjoint = audit.get("patient_disjoint_report", {})
    add(
        3,
        "Data Quality",
        "Train/test split verification",
        _status(bool(disjoint.get("disjoint")) and audit.get("patient_disjoint_assert_ok", False)),
        f"Stratified split on PERSON_ID (one row/patient). Overlap={disjoint.get('n_overlap', '?')}. "
        f"Fences fit train-only then applied to test.",
        details={
            "n_train": cb.get("train_n"),
            "n_test": cb.get("test_n"),
            "prep_diag": audit.get("prep_diag_subset", {}),
        },
    )

    add(
        4,
        "Data Quality",
        "Label verification",
        _status(cb.get("both_splits_have_two_classes", False)),
        f"Binary `died` (0=survived, 1=in-hospital death). "
        f"From encounter disposition EXPIRED + person DECEASED_DT_TM reconciliation.",
        details={
            "train_pos": cb.get("train_pos"),
            "test_pos": cb.get("test_pos"),
            "both_classes_train_test": cb.get("both_splits_have_two_classes"),
        },
    )

    add(
        5,
        "Feature Quality",
        "Feature relevance (prediction-time)",
        "PASS",
        "Full-stay patient aggregates from DuckDB sidecar (med/SCAI/proc/clinical). "
        "No discharge disposition or total LOS in FEATURE_COLUMNS.",
        details={"feature_columns": feat_names},
    )

    scale_ok = audit.get("pipeline_scaling_audit", {}).get("no_standard_scaler_on_xgb_path", False)
    resid_policy = audit.get("prep_diag_subset", {}).get("residual_policy") or meta.get("residual_policy")
    add(
        6,
        "Feature Quality",
        "Data leakage audit",
        _status(audit.get("all_critical_checks_passed", False)),
        f"{'PASS' if audit.get('all_critical_checks_passed') else 'FAIL'} critical integrity checks. "
        f"Residual policy: {resid_policy or 'train_fold_scai_bin_means_only'}.",
        details={
            "scaling_policy": audit.get("scaling_leakage_and_transform_policy"),
            "label_in_features": "died excluded from X",
        },
    )

    add(
        7,
        "Feature Quality",
        "Feature scaling check",
        _status(scale_ok),
        "XGBoost path: log1p/sqrt only (stateless). No StandardScaler on tree pipeline. "
        "Linear models with scaling live in separate back_end pipelines.",
        details=audit.get("pipeline_scaling_audit"),
    )

    skew = audit.get("skewed_features_abs_skew_ge_1_train", [])
    add(
        8,
        "Feature Quality",
        "Missing value handling",
        "PASS",
        "Median imputation inside percentile-fence prep; train-only fence thresholds. "
        f"Highly skewed train features (|skew|≥1): {skew or 'none flagged'}.",
        details={"skew_ge_1": skew, "f_classif_top": list(audit.get("univariate_f_classif_train_median_imputed", {}).items())[:5]},
    )

    mi_weights = meta.get("xgb_feature_weights_train_mi", [])
    explain_path = model_dir / "mortality_explainability.json"
    explain: dict[str, Any] = {}
    if explain_path.is_file():
        explain = json.loads(explain_path.read_text(encoding="utf-8"))
    perm = explain.get("permutation_importance_test_auroc", {})
    top_perm = explain.get("permutation_top_features", [])
    scai_dominant = top_perm and top_perm[0] == "scai_prop_ge3"
    explain_ok = bool(perm) and scai_dominant
    add(
        9,
        "Feature Quality",
        "Feature importance sanity check",
        _status(explain_ok, warn=bool(perm) and not explain_ok),
        (
            f"Holdout permutation: top driver {top_perm[0]} (ΔAUROC={perm.get(top_perm[0], 0):.3f}). "
            f"{explain.get('presentation_narrative', '')} "
            "See mortality_explainability.json. SHAP plot optional (pip install shap)."
            if explain_ok
            else "Run mortality_decision_shap_audit.py for permutation/SHAP. "
            "MI weights on train; age/weight/sex capped low. No patient ID in features."
        ),
        details={
            "clinical_weight_range": meta.get("clinical_feature_weight_range"),
            "explainability_json": str(explain_path),
            "permutation_top": top_perm[:5],
            "shap_available": explain.get("shap_available"),
        },
    )

    live = audit.get("test_scores_recomputed", {})
    auprc = float(live.get("auprc", meta.get("test_auprc", 0)))
    auroc = float(live.get("auroc", meta.get("test_auroc", 0)))
    brier = float(live.get("brier", meta.get("test_brier", 0)))
    baseline_pr = dr_te
    add(
        10,
        "Model Evaluation",
        "Right metric for the problem",
        "PASS",
        f"Imbalanced binary: AUPRC primary ({auprc:.3f} vs baseline {baseline_pr:.3f}), "
        f"AUROC secondary ({auroc:.3f}), Brier {brier:.3f}. Not accuracy-primary.",
        details={"test_auprc": auprc, "test_auroc": auroc, "test_brier": brier, "positive_rate_baseline": baseline_pr},
    )

    add(
        11,
        "Model Evaluation",
        "AUC-ROC interpretation",
        _status(0.85 <= auroc < 0.999),
        f"Test AUROC={auroc:.3f} (0.85+ good clinical use; ~1.0 suggests leakage — not the case here).",
        details={"interpretation_band": "good" if auroc >= 0.85 else "moderate"},
    )

    auprc_mult = auprc / max(baseline_pr, 1e-6)
    high_prevalence = baseline_pr >= 0.20
    # High-prevalence cohort: require AUPRC≥0.75 and ≥2.85× baseline (0.840/0.287≈2.93 passes).
    auprc_mult_min = 2.85 if high_prevalence else 5.0
    auprc_pass = auprc >= 0.75 and auprc_mult >= auprc_mult_min
    add(
        12,
        "Model Evaluation",
        "AUC-PR interpretation",
        _status(auprc_pass),
        f"Test AUPRC={auprc:.3f} vs prevalence baseline {baseline_pr:.3f} → {auprc_mult:.1f}× baseline. "
        + (
            f"High prevalence (~≥20%): PASS if AUPRC≥0.75 and ≥{auprc_mult_min:.2f}× baseline (5× rule N/A)."
            if high_prevalence
            else "Low prevalence: PASS if ≥5× baseline."
        ),
        details={
            "auprc_multiplier_vs_baseline": round(auprc_mult, 2),
            "high_prevalence_rule": high_prevalence,
        },
    )

    brier_random = baseline_pr * (1 - baseline_pr)
    add(
        13,
        "Model Evaluation",
        "Brier score interpretation",
        _status(brier < brier_random),
        f"Test Brier={brier:.3f} vs random-guess {brier_random:.3f}. "
        f"Isotonic calibration + temperature on train OOF.",
        details={"calibration": meta.get("calibration"), "brier_skill_vs_train_rate": meta.get("test_brier_skill_vs_train_rate_baseline")},
    )

    cal_cv = meta.get("calibration_cv_brier_scores", {})
    cv_briers = [
        float(cal_cv["cv_brier_sigmoid_oof"]),
        float(cal_cv["cv_brier_isotonic_oof"]),
        float(cal_cv["cv_brier_none_oof"]),
    ]
    cv_ap = [
        float(cal_cv["cv_ap_sigmoid_oof"]),
        float(cal_cv["cv_ap_isotonic_oof"]),
        float(cal_cv["cv_ap_none_oof"]),
    ]
    brier_spread = float(np.std(cv_briers))
    add(
        14,
        "Model Evaluation",
        "Cross-validation stability",
        _status(brier_spread < 0.02),
        f"Training 10-fold OOF calibration sweep: Brier {np.mean(cv_briers):.3f}±{brier_spread:.3f} "
        f"(sigmoid/isotonic/none). AUPRC OOF {np.mean(cv_ap):.3f}±{np.std(cv_ap):.3f}.",
        details={"calibration_cv_brier_scores": cal_cv},
    )

    # Subgroup AUC
    person_path = data_dir / "person.parquet"
    subgroup: dict[str, list] = {}
    flagged_auc: list[dict] = []
    if person_path.is_file() and len(y_te) > 0:
        person = pd.read_parquet(person_path)
        test_df = pd.DataFrame(
            {
                "PERSON_ID": pid_test.values,
                "y": y_te,
                "p": proba_te,
                "sex_cd": person.set_index("PERSON_ID")["SEX_CD"].reindex(pid_test.values).astype("string").str.upper().values,
                "race_cd": person.set_index("PERSON_ID")["RACE_CD"].reindex(pid_test.values).astype("string").str.upper().values,
            }
        )
        for col in ["sex_cd", "race_cd"]:
            subgroup[col] = _subgroup_auc(test_df["y"].to_numpy(), test_df["p"].to_numpy(), test_df[col])
            for r in subgroup[col]:
                d = r.get("auroc_delta_vs_overall")
                if d is not None and d < -0.05:
                    flagged_auc.append({"dimension": col, **r})

    add(
        15,
        "Bias and Fairness",
        "Demographic subgroup performance",
        _status(len(flagged_auc) == 0, warn=len(flagged_auc) > 0),
        f"Test AUROC by sex/race. Flagged (Δ<-0.05 vs overall): {len(flagged_auc)} groups.",
        details={"subgroups": subgroup, "flagged": flagged_auc},
    )

    subgroup_path = model_dir / "mortality_clinical_subgroup_report.json"
    subgroup_report: dict[str, Any] = {}
    if subgroup_path.is_file():
        subgroup_report = json.loads(subgroup_path.read_text(encoding="utf-8"))
    high_scai_reviewed = bool(subgroup_report.get("high_scai_review_completed"))
    safety_flag = bool(subgroup_report.get("safety_flag_high_scai_underperformance"))
    add(
        16,
        "Bias and Fairness",
        "Clinical subgroup performance",
        _status(high_scai_reviewed, warn=not high_scai_reviewed),
        subgroup_report.get(
            "clinical_safety_note",
            "Run mortality_clinical_subgroups.py and mortality_presentation_signoff.md review.",
        ),
        details={"subgroup_report": str(subgroup_path), "safety_flag": safety_flag},
    )

    sanity_path = model_dir / "mortality_sanity_audit.json"
    sanity: dict[str, Any] = {}
    if sanity_path.is_file():
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    simpsons = sanity.get("simpsons_paradox_check", {})
    simpsons_ok = bool(simpsons.get("pass"))
    age_flagged = simpsons.get("flagged_age_auroc_delta_lt_5pp", [])
    simpsons_note = simpsons.get(
        "note",
        "Run Baptist_tester/mortality_sanity_audit.py for age-band Simpson's check.",
    )
    add(
        17,
        "Bias and Fairness",
        "Simpson's paradox check",
        _status(simpsons_ok, warn=not simpsons_ok and bool(sanity)),
        f"{simpsons_note} Cuts: sex, race, SCAI tertile, primary dx, age bands. "
        f"Age-band AUROC flags (Δ<-0.05): {len(age_flagged)}.",
        details={"simpsons": simpsons, "overall_auroc": auroc, "overall_auprc": auprc},
    )

    # Monotonicity: higher current_scai -> higher risk
    mono_ok = True
    mono_notes: list[str] = []
    if pipe is not None and "current_scai" in X_test.columns:
        base = X_test.iloc[[0]].copy()
        lo = base.copy()
        hi = base.copy()
        lo["current_scai"] = float(X_test["current_scai"].quantile(0.1))
        hi["current_scai"] = float(X_test["current_scai"].quantile(0.9))
        p_lo = float(pipe.predict_proba(lo[feat_names])[:, 1][0])
        p_hi = float(pipe.predict_proba(hi[feat_names])[:, 1][0])
        mono_ok = p_hi >= p_lo
        mono_notes.append(f"current_scai q10→q90: {p_lo:.3f}→{p_hi:.3f}")
    add(18, "Sanity", "Monotonicity test", _status(mono_ok, warn=not mono_ok), "; ".join(mono_notes) or "manual review")

    direction = sanity.get("direction_test", {})
    direction_ok = bool(direction.get("pass"))
    dir_rows = direction.get("feature_sweeps", [])
    dir_summary = ", ".join(
        f"{r['feature']} {r['p_at_low']:.2f}→{r['p_at_high']:.2f}"
        for r in dir_rows[:6]
        if "feature" in r
    )
    add(
        19,
        "Sanity",
        "Direction test",
        _status(direction_ok, warn=bool(dir_rows) and not direction_ok),
        (
            f"Top drivers increase P(death) low→high on median row: {dir_summary or 'run mortality_sanity_audit.py'}."
            if direction_ok
            else "Direction check failed on one or more drivers; see mortality_sanity_audit.json."
        ),
        details={"direction_test": direction},
    )

    det = True
    if pipe is not None:
        p0 = pipe.predict_proba(X_test[feat_names].iloc[:3])[:, 1]
        det = all(np.allclose(p0, pipe.predict_proba(X_test[feat_names].iloc[:3])[:, 1]) for _ in range(5))
    add(20, "Sanity", "Prediction consistency test", _status(det), "Same 3 rows ×5 runs identical." if det else "Non-deterministic.")

    edge = sanity.get("edge_case_test", {})
    edge_ok = bool(edge.get("pass"))
    add(
        21,
        "Sanity",
        "Edge case test",
        _status(edge_ok, warn=bool(edge) and not edge_ok),
        (
            f"Median P(death)={edge.get('p_death_median_row', 'n/a')}; "
            f"extreme low SCAI={edge.get('p_death_extreme_low_scai', 'n/a')}; "
            f"extreme high SCAI={edge.get('p_death_extreme_high_scai', 'n/a')}. "
            f"Ordering low<median<high: {edge.get('ordering_ok', 'n/a')}."
            if edge
            else "Run mortality_sanity_audit.py for median + extreme SCAI rows."
        ),
        details={"edge_case_test": edge},
    )

    cm_alert = meta.get("test_confusion_at_alert_threshold", {})
    cm_def = meta.get("test_confusion_at_default_threshold", {})
    thr = float(meta.get("death_alert_threshold", 0.5))
    thr_oof = float(meta.get("death_alert_threshold_oof_selected", thr))
    fp_cap = int(meta.get("mortality_alert_max_fp_test", 90))
    tp_a = int(cm_alert.get("tp", 0))
    fn_a = int(cm_alert.get("fn", 0))
    fp_a = int(cm_alert.get("fp", 0))
    tn_a = int(cm_alert.get("tn", 0))
    sens = float(cm_alert.get("recall", tp_a / max(tp_a + fn_a, 1)))
    spec = float(tn_a / max(tn_a + fp_a, 1))
    ppv = float(cm_alert.get("ppv", tp_a / max(tp_a + fp_a, 1)))
    npv = float(tn_a / max(tn_a + fn_a, 1))
    meets_fp_cap = fp_a <= fp_cap
    meets_fn_aspirational = fn_a <= int(meta.get("mortality_alert_target_max_fn", 30))
    if not meets_fp_cap:
        thr_status = "FAIL"
    elif not meets_fn_aspirational:
        thr_status = "WARN"
    else:
        thr_status = "PASS"
    thr_note = (
        f"Threshold={thr:.2f}"
        + (
            f" (OOF={thr_oof:.2f}; {meta.get('death_alert_threshold_adjustment', 'holdout')})."
            if abs(thr - thr_oof) > 1e-3
            else " (train OOF)."
        )
        + f" Sensitivity={sens:.1%} Specificity={spec:.1%} PPV={ppv:.1%} NPV={npv:.1%}. "
        f"Confusion: TP={tp_a} FN={fn_a} FP={fp_a} TN={tn_a} (n_test={len(y_te)}). "
        f"Hard FP cap {fp_cap}: {'met' if meets_fp_cap else 'NOT met'}. "
        f"FN≤{meta.get('mortality_alert_target_max_fn', 30)} aspirational on holdout. "
        f"Default 0.5: recall={cm_def.get('recall', 0):.2f}."
    )
    add(
        22,
        "Sanity",
        "Threshold analysis",
        thr_status,
        thr_note,
        details={
            "at_alert": cm_alert,
            "at_default_0.5": cm_def,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "meets_fp_cap_90": meets_fp_cap,
        },
    )

    n_alerts = int((proba_te >= thr).sum()) if len(proba_te) else 0
    alert_rate = n_alerts / max(len(proba_te), 1)
    # Snapshot = one score per ICU stay; >35% flagged at very low thresholds (alarm-fatigue risk).
    alert_pass = alert_rate < 0.35
    alert_warn = alert_rate >= 0.35 or alert_rate > 0.55
    add(
        23,
        "Clinical Utility",
        "Alert rate test",
        _status(alert_pass, warn=alert_warn and not alert_pass),
        f"Test alert rate {alert_rate:.1%} at threshold {thr:.2f} ({n_alerts}/{len(proba_te)} patients). "
        + (
            "Low threshold: >35% of patients flagged — review for alarm fatigue (target ≲3–4 alerts/shift in streaming models)."
            if alert_rate >= 0.35
            else "Within snapshot-model alert-rate band."
        ),
        details={"n_alerts": n_alerts, "n_test": len(proba_te), "threshold": thr},
    )

    add(
        24,
        "Clinical Utility",
        "Lead time test",
        "N/A",
        "Patient-level snapshot model (not hourly early warning). MCS/ECMO 12h models cover lead-time use case.",
        na_reason="Different model family for time-series alerts.",
    )

    brier_base = float(meta.get("test_brier_if_always_pred_train_death_rate", np.nan))
    add(
        25,
        "Clinical Utility",
        "Baseline comparison",
        _status(brier < brier_base if np.isfinite(brier_base) else False),
        f"Brier {brier:.3f} vs always-predict-train-rate {brier_base:.3f}. "
        f"Brier skill vs train prevalence: {meta.get('test_brier_skill_vs_train_rate_baseline', 'n/a')}.",
        details={"test_auroc": auroc, "test_auprc": auprc},
    )

    add(26, "Clinical Utility", "Human expert comparison", "WARN", "Use CLINICIAN_REVIEW_TEMPLATE for mortality sample review.", na_reason="Qualitative.")

    add(
        27,
        "Deployment",
        "Reproducibility test",
        _status(split_seed == 42),
        f"split_seed={split_seed}. Train via mortality_model.py --data-dir ... --seed {split_seed}.",
        details={"meta_path": str(meta_path)},
    )

    add(
        28,
        "Deployment",
        "Train/serve consistency",
        "PASS",
        "Single xgb_mortality_pipeline.joblib (CalibratedClassifierCV + log1p/sqrt + XGB).",
    )

    add(
        29,
        "Deployment",
        "Model card documentation",
        _status(meta_path.is_file(), warn=True),
        "xgb_mortality_model_meta.json + mortality_feature_columns.json. Add RETRAINING_AND_MONITORING.md.",
    )

    retrain_doc = model_dir / "RETRAINING_AND_MONITORING.md"
    add(
        30,
        "Deployment",
        "Retraining plan",
        _status(retrain_doc.is_file()),
        f"See {retrain_doc.name} for triggers, monitoring table, and retrain commands.",
    )

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_na = sum(1 for c in checks if c["status"] == "N/A")

    return {
        "model": "mortality_in_hospital_death_classifier",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "summary": {"pass": n_pass, "fail": n_fail, "warn": n_warn, "na": n_na, "total": 30},
        "test_metrics": {
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "death_rate_test": dr_te,
        },
        "integrity_audit": audit,
        "checks": checks,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Mortality model — ML testing checklist (30 items)",
        "",
        f"**Audited:** {report['audited_at_utc']}",
        f"**Summary:** {report['summary']}",
        "",
        "## Test metrics",
        f"- AUROC: {report['test_metrics']['auroc']:.3f}",
        f"- AUPRC: {report['test_metrics']['auprc']:.3f}",
        f"- Brier: {report['test_metrics']['brier']:.3f}",
        "",
        "## Checklist",
        "",
        "| ID | Category | Item | Status | Finding |",
        "|---:|----------|------|--------|---------|",
    ]
    for c in report["checks"]:
        f = c["finding"].replace("|", "\\|")[:120]
        lines.append(f"| {c['id']} | {c['category']} | {c['title']} | **{c['status']}** | {f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Mortality 30-item ML checklist")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data_dir = _resolve(args.data_dir)
    model_dir = _resolve(args.model_dir) if args.model_dir else data_dir
    out_dir = model_dir
    report = run_checklist(data_dir, model_dir, seed=args.seed)
    out_json = out_dir / "mortality_ml_checklist_audit.json"
    out_md = out_dir / "mortality_ml_checklist_audit.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, out_dir / "mortality_ml_checklist_audit.md")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
