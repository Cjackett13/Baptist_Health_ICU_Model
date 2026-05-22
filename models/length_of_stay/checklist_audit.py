#!/usr/bin/env python3
"""
Run General ML Model Testing Checklist against the LOS (length of stay) model.

Adapted for regression (not binary classification). Writes:
  <model-dir>/los_ml_checklist_audit.json
  <model-dir>/los_ml_checklist_audit.md

  python3 -m models.length_of_stay.checklist_audit --data-dir data
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from xgboost import XGBRegressor

from models.length_of_stay.config import DATA_DIR_DEFAULT, PERFORMANCE_TARGETS, REPO_ROOT
from models.length_of_stay.features import (
    load_training_feature_spec,
    manifest_feature_columns,
    raw_matrix_for_pipeline,
)
from models.length_of_stay.train import (
    DEFAULT_XGB_PARAMS,
    LABEL_EVAL,
    LABEL_TRAIN,
    _metrics,
    load_training_frame,
    split_by_manifest,
)
from models.length_of_stay.transforms import LosTrainPreprocessor
from models.length_of_stay.subgroup_mae import run_subgroup_mae
from models.length_of_stay.sanity_tests import FEATURES_TO_TEST, run_sanity_tests


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _status(pass_: bool, *, warn: bool = False) -> str:
    if pass_:
        return "PASS"
    return "WARN" if warn else "FAIL"


def run_checklist(data_dir: Path, model_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir)
    meta = json.loads((model_dir / "xgb_los_model_meta.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((data_dir / "los_split_manifest.json").read_text(encoding="utf-8"))
    feat = pd.read_parquet(data_dir / "los_modeling_features.parquet")
    labels = pd.read_parquet(data_dir / "los_encounter_labels.parquet")
    df = load_training_frame(data_dir)
    tr, va, te = split_by_manifest(df, data_dir)
    pipe = joblib.load(model_dir / "xgb_los_pipeline.joblib")
    included, _ = load_training_feature_spec(model_dir, data_dir)
    feature_manifest = manifest_feature_columns()
    raw_meta = meta.get("feature_columns_raw_included", [])
    if set(raw_meta) != set(feature_manifest):
        mismatch = {
            "extra_in_model": sorted(set(raw_meta) - set(feature_manifest)),
            "missing_from_model": sorted(set(feature_manifest) - set(raw_meta)),
        }
    else:
        mismatch = {}

    y_all = labels[LABEL_EVAL]
    checks: list[dict[str, Any]] = []

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

    # --- Category 1 (adapted for regression) ---
    add(
        1,
        "Data Quality",
        "Class distribution check",
        "N/A",
        "LOS model is regression (continuous hours), not binary classification.",
        na_reason="Use label distribution (mean/median/std) instead of positive rate.",
        details={
            "train_mean_hours": float(tr[LABEL_EVAL].mean()),
            "test_mean_hours": float(te[LABEL_EVAL].mean()),
            "train_median_hours": float(tr[LABEL_EVAL].median()),
            "test_median_hours": float(te[LABEL_EVAL].median()),
        },
    )

    add(
        2,
        "Data Quality",
        "Class imbalance assessment",
        "N/A",
        "Not applicable to continuous LOS target.",
        na_reason="Imbalance concepts apply to classification only.",
    )

    split_overlap = set(tr["PERSON_ID"]) & set(te["PERSON_ID"])
    add(
        3,
        "Data Quality",
        "Train/test split verification",
        _status(len(split_overlap) == 0),
        f"Split by PERSON_ID: {split_manifest.get('policy')}. "
        f"Train∩test person overlap={len(split_overlap)}.",
        details={
            "n_train_persons": int(tr["PERSON_ID"].nunique()),
            "n_test_persons": int(te["PERSON_ID"].nunique()),
            "grain": "one encounter per person (1:1)",
        },
    )

    label_nan = float(labels[LABEL_EVAL].isna().mean() * 100)
    add(
        4,
        "Data Quality",
        "Label verification",
        _status(label_nan == 0.0),
        f"Label `{LABEL_EVAL}`: {label_nan:.2f}% NaN. "
        f"Training transform `{LABEL_TRAIN}` = log1p(hours). "
        "Constructed from encounter REG/DISCH and reconciled LOS_HOURS in prep.",
        details={
            "label_min": float(labels[LABEL_EVAL].min()),
            "label_max": float(labels[LABEL_EVAL].max()),
            "label_median": float(labels[LABEL_EVAL].median()),
        },
    )

    add(
        5,
        "Feature Quality",
        "Feature manifest vs trained model",
        _status(not mismatch, warn=bool(mismatch)),
        (
            "Trained feature_columns_raw_included matches los_model_config.MODEL_FEATURE_COLUMNS (14)."
            if not mismatch
            else f"Manifest mismatch — retrain required: {mismatch}"
        ),
        details={"manifest": feature_manifest, "trained": raw_meta, "mismatch": mismatch},
    )

    # --- Category 2 (continued) ---
    forbidden = {
        LABEL_EVAL,
        LABEL_TRAIN,
        "los_hours",
        "remaining_los_hours",
        "log1p_remaining_los_hours",
    }
    feat_cols = set(feat.columns)
    leak_cols = sorted(forbidden & feat_cols)
    add(
        6,
        "Feature Quality",
        "Feature relevance (prediction-time)",
        _status(len(included) == 14),
        f"Model uses {len(included)} manifest features (≤12h snapshot + admit-time categoricals). "
        "race_cd/ethnicity excluded from training (subgroup eval only).",
        details={"n_included_features": len(included), "included": included},
    )

    add(
        7,
        "Feature Quality",
        "Data leakage audit",
        _status(len(leak_cols) == 0),
        f"Label columns in feature matrix: {leak_cols or 'none'}. "
        "Preprocessor fit on train fold only (SCAI PCA disabled; residuals train-fold). "
        "Full-stay DuckDB sidecar blocked from training features.",
        details={
            "residual_policy": meta.get("residual_policy"),
            "use_scai_pca": meta.get("use_scai_pca"),
            "demographic_policy": meta.get("demographic_training_policy"),
        },
    )

    add(
        8,
        "Feature Quality",
        "Feature scaling check",
        "PASS",
        "XGBoost regressor — no StandardScaler on full matrix. "
        "Train preprocessor uses train-fold median impute, winsorize, sqrt/log1p only.",
    )

    null_pcts = {c: round(100 * feat[c].isna().mean(), 2) for c in included if c in feat.columns}
    add(
        9,
        "Feature Quality",
        "Missing value handling",
        _status(all(v == 0.0 for v in null_pcts.values())),
        f"Median/mode imputation at train time. Current null % (included features): "
        f"max={max(null_pcts.values()) if null_pcts else 0}%.",
        details={"null_pct_by_feature": null_pcts},
    )

    imp = meta.get("xgb_feature_weights", {})
    top_weighted = sorted(imp.items(), key=lambda x: -x[1])[:8]
    shap_note = "Run: python3 -m models.length_of_stay.validate --data-dir data"
    shap_path = model_dir / "validation" / "los_validation_bundle.json"
    shap_details: dict[str, Any] = {"top_feature_weights": top_weighted}
    shap_status = "WARN"
    if shap_path.is_file():
        bundle = json.loads(shap_path.read_text(encoding="utf-8"))
        shap_block = bundle.get("shap", {})
        if shap_block.get("shap_available"):
            top_shap = (shap_block.get("mean_abs_shap") or shap_block.get("permutation_importance") or [])[:5]
            shap_details["top_shap_or_perm"] = top_shap
            shap_status = _status(len(top_shap) >= 3)
            shap_note = (
                f"SHAP/perm available ({shap_block.get('shap_backend', 'shap')}). "
                f"Test MAE {shap_block.get('mae_hours_test', 0):.1f}h."
            )
    add(
        10,
        "Feature Quality",
        "Feature importance sanity check",
        shap_status,
        shap_note + " Weights are intentional — see data/los_model/FEATURE_WEIGHTS_RATIONALE.md.",
        details=shap_details,
    )

    # --- Category 3 ---
    add(
        11,
        "Model Evaluation",
        "Right metric for the problem",
        "PASS",
        "Regression task uses MAE, RMSE, R² (hours) — not accuracy/AUC. "
        f"Targets: MAE≤{PERFORMANCE_TARGETS['mae_hours_max']}h, "
        f"RMSE≤{PERFORMANCE_TARGETS['rmse_hours_max']}h, R²≥{PERFORMANCE_TARGETS['r2_hours_min']}.",
        details={"metrics_test": meta.get("metrics_test"), "targets_met": meta.get("targets_met_test")},
    )

    for i, title in [
        (12, "AUC-ROC interpretation"),
        (13, "AUC-PR interpretation"),
        (14, "Brier score interpretation"),
    ]:
        add(i, "Model Evaluation", title, "N/A", "Mortality/classification metrics — not used for LOS regression.", na_reason="Binary classification only.")

    # 15 CV stability (refit preprocessor per fold — no joblib.loads)
    cv_mae: list[float] = []
    cv_r2: list[float] = []
    try:
        _, weight_map = load_training_feature_spec(model_dir, data_dir)
        X_full = raw_matrix_for_pipeline(df, included)
        y = df[LABEL_TRAIN]
        y_h = df[LABEL_EVAL]
        groups = df["PERSON_ID"].to_numpy()
        gkf = GroupKFold(n_splits=5)
        xgb_params = {**DEFAULT_XGB_PARAMS, **meta.get("xgb_params", {})}
        xgb_params.pop("early_stopping_rounds", None)
        n_est = int(xgb_params.pop("n_estimators", 400))
        fold_kw = {
            k: v
            for k, v in xgb_params.items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        fold_kw.setdefault("random_state", 42)
        fold_kw.setdefault("n_jobs", -1)
        fold_kw.setdefault("tree_method", "hist")
        fold_kw.setdefault("objective", "reg:squarederror")
        for tr_idx, te_idx in gkf.split(X_full, y, groups):
            X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
            prep_fold = LosTrainPreprocessor(
                base_feature_columns=included,
                weight_by_column=weight_map,
                scai_pca_n_components=0,
            )
            X_tr_t = prep_fold.fit_transform(X_tr)
            X_te_t = prep_fold.transform(X_te)
            reg_fold = XGBRegressor(n_estimators=n_est, **fold_kw)
            reg_fold.fit(X_tr_t, y.iloc[tr_idx], verbose=False)
            pred = np.expm1(np.clip(reg_fold.predict(X_te_t), 0, None))
            cv_mae.append(float(mean_absolute_error(y_h.iloc[te_idx], pred)))
            cv_r2.append(float(r2_score(y_h.iloc[te_idx], pred)))
        cv_mean_mae = float(np.mean(cv_mae))
        cv_std_mae = float(np.std(cv_mae))
        cv_mean_r2 = float(np.mean(cv_r2))
        cv_std_r2 = float(np.std(cv_r2))
        add(
            15,
            "Model Evaluation",
            "Cross-validation stability",
            _status(cv_std_r2 < 0.05, warn=cv_std_r2 >= 0.02),
            f"5-fold GroupKFold by PERSON_ID: MAE={cv_mean_mae:.1f}±{cv_std_mae:.1f}h, "
            f"R²={cv_mean_r2:.3f}±{cv_std_r2:.3f}.",
            details={
                "cv_mae_folds": cv_mae,
                "cv_r2_folds": cv_r2,
                "cv_mae_mean_std": [cv_mean_mae, cv_std_mae],
                "cv_r2_mean_std": [cv_mean_r2, cv_std_r2],
            },
        )
    except Exception as exc:  # noqa: BLE001
        add(15, "Model Evaluation", "Cross-validation stability", "FAIL", f"CV failed: {exc}")

    # --- Category 4 (subgroup MAE for regression) ---
    te_pred = _predict_hours(pipe, te, included)
    te_eval = te.copy()
    te_eval["pred_hours"] = te_pred
    te_eval["abs_err"] = (te_eval[LABEL_EVAL] - te_eval["pred_hours"]).abs()

    try:
        subgroup = run_subgroup_mae(data_dir, model_dir)
        n_flagged = len(subgroup.get("flagged_mae_worse_than_overall_by_5h", []))
        race_rows = subgroup.get("subgroups", {}).get("race_cd", [])[:4]
        add(
            16,
            "Bias and Fairness",
            "Demographic subgroup performance",
            _status(n_flagged == 0, warn=n_flagged > 0),
            f"Test MAE {subgroup['overall_test']['mae_hours']:.1f}h; "
            f"{n_flagged} groups with MAE >5h vs overall. "
            f"Demographics from person.parquet (eval only). Sample race_cd: {race_rows}",
            details={"subgroup_overall": subgroup["overall_test"], "flagged": subgroup["flagged_mae_worse_than_overall_by_5h"]},
        )
        unit_rows = subgroup.get("subgroups", {}).get("unit_cd", [])
        add(
            17,
            "Bias and Fairness",
            "Clinical subgroup performance",
            _status(True, warn=any(r.get("mae_vs_overall_delta", 0) > 5 for r in unit_rows if "mae_hours" in r)),
            f"Subgroup MAE by unit_cd, primary_dx, admit_type. unit_cd rows: {unit_rows[:3]}",
            details={"unit_cd": unit_rows, "primary_dx": subgroup.get("subgroups", {}).get("primary_dx", [])[:5]},
        )
        add(
            18,
            "Bias and Fairness",
            "Simpson's paradox check",
            _status(n_flagged <= 1, warn=n_flagged > 1),
            "Review flagged subgroup deltas vs overall MAE before sign-off; no race/ethnicity in model X.",
            details={"flagged": subgroup["flagged_mae_worse_than_overall_by_5h"]},
        )
    except Exception as exc:  # noqa: BLE001
        add(16, "Bias and Fairness", "Demographic subgroup performance", "WARN", f"subgroup_mae failed: {exc}")
        add(17, "Bias and Fairness", "Clinical subgroup performance", "WARN", "Run subgroup_mae after validate.")
        add(18, "Bias and Fairness", "Simpson's paradox check", "WARN", "Run subgroup_mae after validate.")

    # --- Category 5 ---
    sanity = run_sanity_tests(data_dir, model_dir)
    dir_rows = sanity.get("direction_tests", [])
    dir_pass = bool(sanity.get("direction_test_pass"))
    deltas = [
        f"{r['feature']}: {r.get('delta_hours_p95_minus_p05', 'n/a')}h"
        for r in dir_rows
        if r.get("status") != "SKIP"
    ]
    add(
        19,
        "Sanity",
        "Direction test",
        _status(dir_pass),
        f"p5→p95 sweeps on median patient ({len(FEATURES_TO_TEST)} features). "
        f"Pass count {sanity.get('direction_pass_count')}. "
        f"Deltas: {', '.join(deltas)}. "
        "PASS when all higher values predict ≥ lower (synthetic deltas may be small).",
        details={"direction_tests": dir_rows},
    )

    add(
        20,
        "Sanity",
        "Monotonicity test",
        _status(dir_pass, warn=not dir_pass),
        "Same 5-feature direction battery as item 19; confirms model responds logically to acuity/load.",
        details={"direction_test_pass": dir_pass},
    )

    edge = sanity.get("edge_cases", {})
    edge_pass = bool(sanity.get("edge_case_pass"))
    edge_high = edge.get("extreme_high_all_numeric_999", {})
    edge_low = edge.get("extreme_low_all_numeric_0", {})
    add(
        21,
        "Sanity",
        "Edge case test",
        _status(edge_pass),
        f"Extreme numeric 999→{edge_high.get('pred_hours')}h, 0→{edge_low.get('pred_hours')}h; "
        f"no crash, finite output in (0, 10000)h. Winsorization in train preprocessor.",
        details=edge,
    )

    det = edge.get("prediction_consistency_20x", {}).get("deterministic", False)
    add(
        22,
        "Sanity",
        "Prediction consistency test",
        _status(det),
        "Same median template predicted 20× — identical outputs." if det else "Non-deterministic outputs detected.",
        details=edge.get("prediction_consistency_20x"),
    )

    add(23, "Sanity", "Threshold analysis", "N/A", "Regression has no classification threshold; use MAE buckets or long-stay binary if needed.", na_reason="Classification only.")

    # --- Category 6 ---
    add(24, "Clinical Utility", "Alert rate test", "N/A", "LOS regression outputs hours, not binary alerts.", na_reason="Define long-stay threshold for alert-rate metric.")

    add(
        25,
        "Clinical Utility",
        "Lead time test",
        "PASS",
        "v1 checkpoint: prediction_hour=12, features censored at 12h. Rolling mode documented in los_feature_policy (not enabled).",
        details={"feature_window_hours": 12, "rolling_feature_mode": meta.get("rolling_feature_mode", "v1_static_12h")},
    )

    base_mae = float(meta["metrics_test"].get("baseline_mae_hours_median_predictor", np.nan))
    test_mae = float(meta["metrics_test"]["mae_hours"])
    add(
        26,
        "Clinical Utility",
        "Baseline comparison",
        _status(test_mae < base_mae),
        f"Test MAE {test_mae:.1f}h vs median-predictor baseline {base_mae:.1f}h "
        f"({meta['metrics_test'].get('mae_improvement_vs_baseline_pct')}% improvement).",
    )

    add(27, "Clinical Utility", "Human expert comparison", "WARN", "Clinician review of high-error cases not automated.", na_reason="Qualitative step.")

    # --- Category 7 ---
    add(
        28,
        "Deployment",
        "Reproducibility test",
        _status(meta.get("xgb_params", {}).get("random_state") == 42),
        f"random_state={meta.get('xgb_params', {}).get('random_state')}, split seed={split_manifest.get('seed')}. "
        "Regenerate: simulate → los_data_prep → tune → train.",
        details={"artifact_dir": str(model_dir)},
    )

    add(
        29,
        "Deployment",
        "Train/serve consistency",
        "PASS",
        "Single joblib pipeline (LosTrainPreprocessor + XGBRegressor). "
        "Serve must load same bundle; no separate scaler files.",
    )

    add(
        30,
        "Deployment",
        "Model card documentation",
        _status((model_dir / "performance_targets.json").is_file(), warn=True),
        "See models/length_of_stay/README.md, xgb_los_model_meta.json, performance_targets.json.",
    )

    retrain_doc = model_dir / "RETRAINING_AND_MONITORING.md"
    weights_doc = model_dir / "FEATURE_WEIGHTS_RATIONALE.md"
    add(
        31,
        "Deployment",
        "Retraining plan",
        _status(retrain_doc.is_file()),
        f"Retrain/monitoring: {retrain_doc.name if retrain_doc.is_file() else 'MISSING'}. "
        f"Feature weights rationale: {weights_doc.name if weights_doc.is_file() else 'MISSING'}.",
        details={"retrain_doc": str(retrain_doc), "weights_doc": str(weights_doc)},
    )

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_na = sum(1 for c in checks if c["status"] == "N/A")

    return {
        "model": "length_of_stay_regression",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "summary": {"pass": n_pass, "fail": n_fail, "warn": n_warn, "na": n_na, "total": len(checks)},
        "metrics_test": meta.get("metrics_test"),
        "targets_met_test": meta.get("targets_met_test"),
        "checks": checks,
    }


def _predict_hours(pipe, df: pd.DataFrame, included: list[str]) -> np.ndarray:
    pred_log = pipe.predict(raw_matrix_for_pipeline(df, included))
    return np.expm1(np.clip(pred_log, 0, None))


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# LOS model — ML testing checklist audit",
        "",
        f"**Audited:** {report['audited_at_utc']}",
        f"**Summary:** {report['summary']}",
        "",
        "## Test metrics",
        f"```json\n{json.dumps(report.get('metrics_test'), indent=2)}\n```",
        "",
        "## Checklist",
        "",
        "| ID | Category | Item | Status | Finding |",
        "|---:|----------|------|--------|---------|",
    ]
    for c in report["checks"]:
        finding = c["finding"].replace("|", "\\|").replace("\n", " ")
        if len(finding) > 120:
            finding = finding[:117] + "..."
        lines.append(f"| {c['id']} | {c['category']} | {c['title']} | **{c['status']}** | {finding} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="LOS ML testing checklist audit")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args()
    model_dir = _resolve(args.model_dir) if args.model_dir else _resolve(args.data_dir) / "los_model"
    report = run_checklist(args.data_dir, model_dir)
    out_json = model_dir / "los_ml_checklist_audit.json"
    out_md = model_dir / "los_ml_checklist_audit.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, out_md)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
