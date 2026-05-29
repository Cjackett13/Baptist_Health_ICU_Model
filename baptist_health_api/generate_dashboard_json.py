#!/usr/bin/env python3
"""
generate_dashboard_json.py

Reads raw parquets from data_dir, runs all 6 models, computes SHAP values,
derives ground truth labels, computes per-model metrics, and writes a single
dashboard_results.json consumed by the Flutter app.

Usage:
    cd Baptist_Health_ICU_Model/baptist_health_api
    python3 generate_dashboard_json.py \
        --data-dir /path/to/parquet/dir \
        --model-dir model_files \
        --output ../Frontend/assets/dashboard_results.json
"""

from __future__ import annotations

import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

# ── Path setup so we can import from the API package ─────────────────────────
API_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(API_DIR))

# Pickle custom class resolution — must happen before any pkl is loaded
from app.models.wrapper_classes import IsotonicCalibratedModel, PlattXGB  # noqa: F401
sys.modules["__main__"].IsotonicCalibratedModel = IsotonicCalibratedModel
sys.modules["__main__"].PlattXGB = PlattXGB

import mcs_escalation_inference as mcs_inf  # noqa: E402
import ecmo_escalation_inference as ecmo_inf  # noqa: E402
import mortality_transforms  # noqa: F401, E402
import models.length_of_stay.transforms  # noqa: F401, E402

from model_development import (  # noqa: E402
    load_tables,
    build_hourly_wide,
    build_window_features,
    build_treatment_state,
)
from mortality_duckdb_features import materialize_duckdb_patient_features  # noqa: E402
from los_duckdb_features import materialize_duckdb_los_features  # noqa: E402
from app.models.features.scai_pipeline import build_features as scai_build_features  # noqa: E402
from app.models.features.vasopressor_pipeline import build_features as vaso_build_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_SCAI_ABC_THRESHOLDS = {0: 0.15, 1: 0.15, 2: 0.15}  # from scai_deterioration_model
_SCAI_D_THRESHOLD    = 0.14                           # bundle threshold is None; use tuned default
_SCAI_STAGE_LABEL    = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
_VASOPRESSOR_THRESHOLD     = 0.19    # from binary_alert_model bundle["threshold"]
_HIGH_SEVERITY_THRESHOLD   = 0.01   # from high_severity_classifier bundle["threshold"]
_MCS_DEFAULT_THRESHOLD  = 0.10
_ECMO_DEFAULT_THRESHOLD = 0.10

# ── Vital sign metadata: code → (display_title, units, (crit_low, low, high, crit_high)) ──
# None means no threshold in that direction.
_VITAL_META: dict[str, tuple] = {
    "HR":           ("Heart Rate",           "bpm",      (40.0,  50.0,  120.0, 150.0)),
    "SBP":          ("Systolic BP",          "mmHg",     (70.0,  80.0,  180.0, 220.0)),
    "DBP":          ("Diastolic BP",         "mmHg",     (40.0,  50.0,  110.0, 130.0)),
    "MAP":          ("Mean Arterial P.",     "mmHg",     (50.0,  60.0,  110.0, 130.0)),
    "RR":           ("Resp Rate",            "br/min",   (8.0,   10.0,  28.0,  40.0)),
    "SPO2":         ("O2 Saturation",        "%",        (80.0,  88.0,  None,  None)),
    "TEMP":         ("Temperature",          "C",        (34.0,  35.5,  38.5,  40.0)),
    "URINE_OUT_HR": ("Urine Output/hr",      "mL/hr",    (5.0,   20.0,  200.0, 400.0)),
    "CVP":          ("CVP",                  "mmHg",     (1.0,   3.0,   18.0,  25.0)),
    "CO":           ("Cardiac Output",       "L/min",    (2.0,   3.0,   8.0,   10.0)),
    "CI":           ("Cardiac Index",        "L/min/m2", (1.5,   2.0,   5.0,   6.0)),
    "LACTATE":      ("Lactate",              "mmol/L",   (None,  None,  2.0,   4.0)),
    "CREATININE":   ("Creatinine",           "mg/dL",    (None,  None,  1.5,   3.0)),
    "BUN":          ("BUN",                  "mg/dL",    (None,  None,  40.0,  80.0)),
    "NT_PROBNP":    ("NT-proBNP",            "pg/mL",    (None,  None,  2000.0,5000.0)),
    "TROPONIN_I":   ("Troponin I",           "ng/mL",    (None,  None,  0.5,   2.0)),
    "PH":           ("Arterial pH",          "",         (7.25,  7.35,  7.45,  7.55)),
    "PCO2":         ("pCO2",                 "mmHg",     (25.0,  35.0,  45.0,  55.0)),
    "HCO3":         ("Bicarbonate",          "mEq/L",    (15.0,  22.0,  26.0,  32.0)),
    "WBC":          ("WBC",                  "K/uL",     (None,  4.0,   11.0,  25.0)),
    "HGB":          ("Hemoglobin",           "g/dL",     (6.0,   8.0,   16.0,  18.0)),
    "PLT":          ("Platelets",            "K/uL",     (30.0,  50.0,  400.0, 600.0)),
    "INR":          ("INR",                  "",         (None,  None,  1.5,   3.5)),
    "PAS":          ("Pulm. Art. Systolic",  "mmHg",     (None,  None,  40.0,  60.0)),
    "PAD":          ("Pulm. Art. Diastolic", "mmHg",     (None,  None,  20.0,  30.0)),
    "SVO2":         ("SvO2",                 "%",        (50.0,  60.0,  80.0,  85.0)),
    "VIS":          ("Vasopr. Intensity",    "",         (None,  None,  20.0,  40.0)),
}

_VITAL_CODES_ORDERED = [
    "HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP", "URINE_OUT_HR",
    "CVP", "CO", "CI", "LACTATE", "CREATININE", "BUN", "NT_PROBNP", "TROPONIN_I",
    "PH", "PCO2", "HCO3", "WBC", "HGB", "PLT", "INR", "PAS", "PAD", "SVO2", "VIS",
]


def _vital_normalcy(code: str, value: float) -> str:
    meta = _VITAL_META.get(code)
    if meta is None:
        return "NORMAL"
    _, _, (crit_low, low, high, crit_high) = meta
    if crit_high is not None and value >= crit_high:
        return "CRITICAL_HIGH"
    if crit_low is not None and value <= crit_low:
        return "CRITICAL_LOW"
    if high is not None and value >= high:
        return "HIGH"
    if low is not None and value <= low:
        return "LOW"
    return "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_xgb(model):
    """Iteratively strip calibration/ensemble wrappers to reach the raw XGBClassifier."""
    from xgboost import XGBClassifier, XGBRegressor
    for _ in range(10):
        if isinstance(model, (XGBClassifier, XGBRegressor)):
            return model
        if hasattr(model, "base_model"):           # IsotonicCalibratedModel
            model = model.base_model
        elif hasattr(model, "xgb_model"):          # PlattXGB
            model = model.xgb_model
        elif hasattr(model, "calibrated_estimator"):  # TemperatureScaledBinaryCalibrator
            model = model.calibrated_estimator
        elif hasattr(model, "calibrators"):        # AveragedBinaryCalibrators
            model = model.calibrators[0]
        elif hasattr(model, "models"):             # EnsembleXGBClassifier
            model = model.models[0]
        elif hasattr(model, "calibrated_classifiers_"):  # CalibratedClassifierCV — use fitted copy
            model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            model = model.estimator
        elif hasattr(model, "base_estimator"):
            model = model.base_estimator
        elif hasattr(model, "base"):
            model = model.base
        elif hasattr(model, "steps"):              # sklearn Pipeline
            model = model.steps[-1][1]
        else:
            break
    return model


_shap_cache: dict = {}

def _shap_top5(model_key: str, xgb_model, X: pd.DataFrame) -> list[dict]:
    if model_key not in _shap_cache:
        _shap_cache[model_key] = shap.TreeExplainer(xgb_model)
    explainer = _shap_cache[model_key]
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        raw = raw[1]
    results = []
    for i in range(len(X)):
        row = raw[i]
        pairs = sorted(zip(X.columns.tolist(), row), key=lambda t: abs(t[1]), reverse=True)[:5]
        results.append([
            {
                "feature": f,
                "value": round(float(v), 4),
                "is_positive": float(v) >= 0,
                "direction": "positive" if float(v) >= 0 else "negative",
                "description": f.replace("_", " ").title(),
            }
            for f, v in pairs
        ])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(model_dir: Path) -> dict:
    log.info("Loading models from %s", model_dir)

    def _pkl(path: Path):
        if not path.exists():
            log.warning("Missing: %s", path)
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _jbl(path: Path):
        if not path.exists():
            log.warning("Missing: %s", path)
            return None
        return joblib.load(path)

    m = {
        "scai_abc":             _pkl(model_dir / "scai"       / "scai_deterioration_model.pkl"),
        "scai_d":               _pkl(model_dir / "scai"       / "scai_stage_d_model.pkl"),
        "vasopressor_binary":   _pkl(model_dir / "vasopressor" / "binary_alert_model.pkl"),
        "vasopressor_count":    _pkl(model_dir / "vasopressor" / "ordinal_count_model.pkl"),
        "vasopressor_severity": _pkl(model_dir / "vasopressor" / "high_severity_classifier.pkl"),
        "mcs":                  _pkl(model_dir / "johnathan"   / "mcs_seed42.bundle.pkl"),
        "va_ecmo":              _pkl(model_dir / "johnathan"   / "ecmo_seed42.bundle.pkl"),
        "mortality":            _pkl(model_dir / "lilly"       / "mortality_pipeline.pkl"),
        "los":                  _jbl(model_dir / "lilly"       / "xgb_los_pipeline.joblib"),
    }

    # Vasopressor preprocessing artifacts
    vaso_files = {
        "knn_imputer": (model_dir / "vasopressor" / "knn_imputer.pkl",            "pkl"),
        "iqr_bounds":  (model_dir / "vasopressor" / "iqr_bounds.json",            "json"),
        "log_features":(model_dir / "vasopressor" / "log_transform_features.json","json"),
        "feature_manifest":(model_dir / "vasopressor" / "feature_manifest.json",  "json"),
    }
    if all(p.exists() for p, _ in vaso_files.values()):
        prep: dict = {}
        for key, (path, fmt) in vaso_files.items():
            if fmt == "pkl":
                with open(path, "rb") as f:
                    prep[key] = pickle.load(f)
            else:
                with open(path) as f:
                    prep[key] = json.load(f)
        prep["feature_cols"] = prep["feature_manifest"]["feature_columns"]
        m["vasopressor_prep"] = prep
    else:
        m["vasopressor_prep"] = None
        log.warning("Vasopressor prep artifacts incomplete — vasopressor model will be skipped")

    loaded  = [k for k, v in m.items() if v is not None]
    missing = [k for k, v in m.items() if v is None]
    log.info("Loaded: %s", loaded)
    if missing:
        log.warning("Missing / skipped: %s", missing)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth derivation
# ─────────────────────────────────────────────────────────────────────────────

def derive_scai_labels(scai: pd.DataFrame, snapshot_hours: pd.Series) -> pd.Series:
    """1 if SCAI stage worsens ≥1 within 6h of the snapshot hour, else 0."""
    labels = {}
    for enc_id, grp in scai.groupby("ENCOUNTER_ID"):
        grp = grp.sort_values("HOUR_FROM_ADMIT")
        snap_h = snapshot_hours.get(enc_id, grp["HOUR_FROM_ADMIT"].max())
        snap_stage = grp.loc[grp["HOUR_FROM_ADMIT"] <= snap_h, "SCAI_STAGE_NUM"].iloc[-1] if not grp.empty else 0
        future = grp[grp["HOUR_FROM_ADMIT"].between(snap_h + 1, snap_h + 6)]
        labels[enc_id] = int((future["SCAI_STAGE_NUM"] > snap_stage).any()) if not future.empty else 0
    return pd.Series(labels, name="label_scai")


def derive_vasopressor_labels(meds: pd.DataFrame, encounter: pd.DataFrame, snapshot_hours: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Returns (label_vaso_needed, label_vaso_count) for each encounter."""
    VASOPRESSOR_CODES = {
        "NOREPINEPHRINE", "VASOPRESSIN", "EPINEPHRINE", "DOPAMINE",
        "PHENYLEPHRINE", "DOBUTAMINE", "MILRINONE",
    }
    enc_reg = encounter.set_index("ENCOUNTER_ID")["REG_DT_TM"].to_dict()
    needed, count = {}, {}

    vaso_meds = meds[meds["MEDICATION_CD"].str.upper().isin(VASOPRESSOR_CODES)].copy()
    vaso_meds["ADMIN_START_DT_TM"] = pd.to_datetime(vaso_meds["ADMIN_START_DT_TM"])

    for enc_id, grp in vaso_meds.groupby("ENCOUNTER_ID"):
        admit_dt = pd.to_datetime(enc_reg.get(enc_id))
        if pd.isna(admit_dt):
            needed[enc_id] = 0
            count[enc_id] = 0
            continue
        snap_h = int(snapshot_hours.get(enc_id, 0))
        window_start = admit_dt + pd.Timedelta(hours=snap_h)
        window_end   = admit_dt + pd.Timedelta(hours=snap_h + 24)
        new_vasops = grp[grp["ADMIN_START_DT_TM"].between(window_start, window_end)]
        n = new_vasops["MEDICATION_CD"].nunique()
        needed[enc_id] = int(n > 0)
        count[enc_id]  = min(n, 3)

    all_encs = encounter["ENCOUNTER_ID"].tolist()
    return (
        pd.Series({e: needed.get(e, 0) for e in all_encs}, name="label_vaso_needed"),
        pd.Series({e: count.get(e, 0) for e in all_encs},  name="label_vaso_count"),
    )


def derive_mcs_labels(proc: pd.DataFrame, snapshot_hours: pd.Series, encounter: pd.DataFrame) -> pd.Series:
    """1 if patient goes on any MCS device within 12h of snapshot, else 0."""
    enc_reg = encounter.set_index("ENCOUNTER_ID")["REG_DT_TM"].to_dict()
    mcs_procs = proc[proc["PROC_CATEGORY_CD"] == "MCS"].copy()
    mcs_procs["PROC_START_DT_TM"] = pd.to_datetime(mcs_procs["PROC_START_DT_TM"])
    labels = {}
    for enc_id, grp in mcs_procs.groupby("ENCOUNTER_ID"):
        admit_dt = pd.to_datetime(enc_reg.get(enc_id))
        if pd.isna(admit_dt):
            labels[enc_id] = 0
            continue
        snap_h = int(snapshot_hours.get(enc_id, 0))
        window_end = admit_dt + pd.Timedelta(hours=snap_h + 12)
        window_start = admit_dt + pd.Timedelta(hours=snap_h)
        labels[enc_id] = int(grp["PROC_START_DT_TM"].between(window_start, window_end).any())
    all_encs = encounter["ENCOUNTER_ID"].tolist()
    return pd.Series({e: labels.get(e, 0) for e in all_encs}, name="label_mcs")


def derive_ecmo_labels(proc: pd.DataFrame, snapshot_hours: pd.Series, encounter: pd.DataFrame) -> pd.Series:
    """1 if patient goes on VA-ECMO within 12h of snapshot, else 0."""
    enc_reg = encounter.set_index("ENCOUNTER_ID")["REG_DT_TM"].to_dict()
    ecmo_procs = proc[
        (proc["PROC_CATEGORY_CD"] == "MCS") & (proc["NOMENCLATURE_CD"] == "VA_ECMO")
    ].copy()
    ecmo_procs["PROC_START_DT_TM"] = pd.to_datetime(ecmo_procs["PROC_START_DT_TM"])
    labels = {}
    for enc_id, grp in ecmo_procs.groupby("ENCOUNTER_ID"):
        admit_dt = pd.to_datetime(enc_reg.get(enc_id))
        if pd.isna(admit_dt):
            labels[enc_id] = 0
            continue
        snap_h = int(snapshot_hours.get(enc_id, 0))
        window_start = admit_dt + pd.Timedelta(hours=snap_h)
        window_end   = admit_dt + pd.Timedelta(hours=snap_h + 12)
        labels[enc_id] = int(grp["PROC_START_DT_TM"].between(window_start, window_end).any())
    all_encs = encounter["ENCOUNTER_ID"].tolist()
    return pd.Series({e: labels.get(e, 0) for e in all_encs}, name="label_ecmo")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _binary_metrics(y_true, y_prob, threshold=0.5) -> dict:
    from sklearn.metrics import roc_curve
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)
    auc = sens_80 = sens_90 = None
    try:
        auc = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        spec = 1 - fpr
        idx80 = np.where(spec >= 0.80)[0]
        idx90 = np.where(spec >= 0.90)[0]
        sens_80 = round(float(tpr[idx80[-1]]), 4) if len(idx80) else None
        sens_90 = round(float(tpr[idx90[-1]]), 4) if len(idx90) else None
    except Exception:
        pass
    return {
        "auc":                    round(auc, 4) if auc is not None else None,
        "accuracy":               round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":                     round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity_at_80_spec": sens_80,
        "sensitivity_at_90_spec": sens_90,
    }


def _regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def _scai_test_metrics(models: dict, test_path: Path) -> dict:
    """Evaluate both SCAI models on test_frame.parquet (pre-built features + labels)."""
    df     = pd.read_parquet(test_path)
    y_all  = df["Y_PROGRESSION_6H"].values
    stages = df["CURRENT_STAGE_NUM"].values

    all_probs = np.full(len(df), np.nan)

    for bundle_key, mask_fn in [
        ("scai_abc", lambda s: s <= 2),
        ("scai_d",   lambda s: s == 3),
    ]:
        bundle = models.get(bundle_key)
        if bundle is None:
            continue
        fcols = bundle["feature_cols"]
        idx   = np.where(mask_fn(stages))[0]
        if len(idx) == 0:
            continue
        sub = df.iloc[idx].copy()
        for c in [c for c in fcols if c not in sub.columns]:
            sub[c] = 0.0
        X     = sub[fcols].astype(np.float32)
        raw_p = bundle["xgboost_model"].predict_proba(X)[:, 1]
        cal_p = bundle["platt_scaler"].predict_proba(raw_p.reshape(-1, 1))[:, 1]
        all_probs[idx] = cal_p

    valid = ~np.isnan(all_probs)
    combined = _binary_metrics(y_all[valid], all_probs[valid])

    # Per-model breakdown
    result = dict(combined)
    for bundle_key, mask_fn in [
        ("scai_abc", lambda s: s <= 2),
        ("scai_d",   lambda s: s == 3),
    ]:
        mask = mask_fn(stages)
        sub_probs = all_probs[mask]
        sub_y     = y_all[mask]
        sub_valid = ~np.isnan(sub_probs)
        if sub_valid.sum() > 0:
            m = _binary_metrics(sub_y[sub_valid], sub_probs[sub_valid])
            result[f"{bundle_key}_auc"] = m["auc"]
            result[f"{bundle_key}_sensitivity_at_80_spec"] = m["sensitivity_at_80_spec"]
            result[f"{bundle_key}_sensitivity_at_90_spec"] = m["sensitivity_at_90_spec"]
    return result


def _vasopressor_test_metrics(models: dict, test_path: Path) -> dict:
    """Evaluate all vasopressor models on test_features_v3.parquet."""
    df = pd.read_parquet(test_path)
    y_binary = df["VASOPRESSOR_NEEDED_24H"].values
    y_count  = df["VASOPRESSOR_COUNT_24H"].values

    vaso_bin = models["vasopressor_binary"]
    vaso_cnt = models["vasopressor_count"]
    vaso_sev = models["vasopressor_severity"]

    # Binary alert
    bin_cols = vaso_bin["feature_cols"]
    for c in [c for c in bin_cols if c not in df.columns]:
        df[c] = 0.0
    X_bin    = df[bin_cols].astype(np.float32)
    raw      = vaso_bin["model"].predict_proba(X_bin)[:, 1]
    probs_bin = vaso_bin["platt"].predict_proba(raw.reshape(-1, 1))[:, 1]

    # Ordinal count
    cnt_cols = vaso_cnt["feature_cols"]
    for c in [c for c in cnt_cols if c not in df.columns]:
        df[c] = 0.0
    X_cnt     = df[cnt_cols].astype(np.float32)
    cnt_raw   = np.argmax(vaso_cnt["model"].predict_proba(X_cnt), axis=1)
    label_map = vaso_cnt.get("label_map", {0: 1, 1: 2, 2: 3})
    cnt_preds = np.array([label_map.get(int(c), int(c)) for c in cnt_raw])

    # High severity (same features as binary)
    probs_sev = vaso_sev["model"].predict_proba(X_bin)[:, 1]

    bin_m = _binary_metrics(y_binary, probs_bin, threshold=float(vaso_bin["threshold"]))
    sev_m = _binary_metrics(y_binary, probs_sev, threshold=float(vaso_sev["threshold"]))

    return {
        "alert_auc":                      bin_m["auc"],
        "alert_accuracy":                 bin_m["accuracy"],
        "alert_f1":                       bin_m["f1"],
        "alert_sensitivity_at_80_spec":   bin_m["sensitivity_at_80_spec"],
        "alert_sensitivity_at_90_spec":   bin_m["sensitivity_at_90_spec"],
        "high_severity_auc":              sev_m["auc"],
        "high_severity_sensitivity_at_80_spec": sev_m["sensitivity_at_80_spec"],
        "count_accuracy":                 round(float(accuracy_score(y_count, cnt_preds)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: Path, model_dir: Path, output_path: Path) -> None:
    # ── 1. Load raw tables ────────────────────────────────────────────────────
    log.info("Loading raw parquets from %s", data_dir)
    tbls = load_tables(str(data_dir))
    encounter = tbls["encounter"]
    person    = tbls["person"]
    scai      = tbls["scai_stage_hourly"]
    meds      = tbls["medication_admin"]
    proc      = tbls["procedure_event"]
    ce        = tbls["clinical_event"]

    # ── 2. Load models ────────────────────────────────────────────────────────
    models = load_models(model_dir)

    # ── 3. Build patient demographics ─────────────────────────────────────────
    log.info("Building patient demographics")
    now = pd.Timestamp.now()
    person_aug = person.copy()
    person_aug["age"] = ((now - pd.to_datetime(person_aug["BIRTH_DT_TM"])).dt.days / 365.25).astype(int)
    # Generate realistic-looking fake names deterministically from PERSON_ID.
    # The raw parquet stores synthetic "FirstN LastN" strings — replace them
    # with believable names so the demo dashboard reads naturally.
    _FIRST = [
        "James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
        "William","Barbara","David","Susan","Richard","Jessica","Joseph","Sarah",
        "Thomas","Karen","Charles","Lisa","Christopher","Nancy","Daniel","Betty",
        "Matthew","Margaret","Anthony","Sandra","Mark","Ashley","Donald","Dorothy",
        "Steven","Kimberly","Paul","Emily","Andrew","Donna","Joshua","Michelle",
        "Kenneth","Carol","Kevin","Amanda","Brian","Melissa","George","Deborah",
        "Timothy","Stephanie","Ronald","Rebecca","Edward","Sharon","Jason","Laura",
        "Jeffrey","Cynthia","Ryan","Kathleen","Jacob","Amy","Gary","Angela",
        "Nicholas","Shirley","Eric","Anna","Jonathan","Brenda","Stephen","Pamela",
        "Larry","Emma","Justin","Nicole","Scott","Helen","Brandon","Samantha",
    ]
    _LAST = [
        "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
        "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
        "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson","White",
        "Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker","Young",
        "Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores","Green",
        "Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts",
        "Phillips","Evans","Turner","Torres","Parker","Collins","Edwards","Stewart",
        "Flores","Morris","Nguyen","Murphy","Rivera","Cook","Rogers","Morgan","Peterson",
        "Cooper","Reed","Bailey","Bell","Gomez","Kelly","Howard","Ward","Cox","Diaz",
    ]
    person_aug["name"] = person_aug["PERSON_ID"].apply(
        lambda pid: f"{_FIRST[int(pid) % len(_FIRST)]} {_LAST[(int(pid) // len(_FIRST)) % len(_LAST)]}"
    )
    person_aug["died"] = person_aug["DECEASED_DT_TM"].notna().astype(int)

    demo = encounter.merge(person_aug, on="PERSON_ID", how="left")
    demo["los_days"] = (demo["LOS_HOURS"] / 24).round(2)

    # ── 4. Determine snapshot hours per encounter ─────────────────────────────
    scai_sorted   = scai.sort_values("HOUR_FROM_ADMIT")
    last_hour     = scai_sorted.groupby("ENCOUNTER_ID")["HOUR_FROM_ADMIT"].last()
    first_hour    = scai_sorted.groupby("ENCOUNTER_ID")["HOUR_FROM_ADMIT"].first()

    # label_snap_hour: 24h before last observation, floored at first hour.
    # Ensures the 24h outcome window (vasopressor) and 6h window (SCAI) always exist.
    label_snap_hour = (last_hour - 24).clip(lower=first_hour).astype(int)
    label_snap_hours = label_snap_hour.to_dict()
    snapshot_hours   = last_hour.to_dict()  # still used for feature extraction

    # ── 5. Build hourly wide feature frame (shared by SCAI, vasopressor, MCS, ECMO) ──
    log.info("Building hourly wide feature frame")
    wide = build_hourly_wide(ce, scai)
    feats_4h = build_window_features(wide, lookback=4)

    # Snapshot rows: one row per encounter at label_snap_hour.
    # Using the same hour for both features and labels ensures the model sees
    # data up to hour H and predicts outcomes in [H, H+horizon] — matching training.
    snap_df = (
        feats_4h.merge(
            label_snap_hour.rename("snap_hour").reset_index(),
            on="ENCOUNTER_ID",
        )
        .query("HOUR_FROM_ADMIT == snap_hour")
        .drop(columns=["snap_hour"])
        .reset_index(drop=True)
    )

    # SCAI_STAGE_NUM is dropped during the wide pivot — merge it back in
    snap_df = snap_df.merge(
        scai[["ENCOUNTER_ID", "HOUR_FROM_ADMIT", "SCAI_STAGE_NUM"]],
        on=["ENCOUNTER_ID", "HOUR_FROM_ADMIT"],
        how="left",
    )

    enc_ids_snap = snap_df["ENCOUNTER_ID"].tolist()

    # ── 5b. Extract vitals from snapshot rows ─────────────────────────────────
    log.info("Extracting vitals from snapshot rows")
    enc_vitals: dict[int, list] = {}
    for _, row in snap_df.iterrows():
        enc_id = int(row["ENCOUNTER_ID"])
        vitals_list = []
        for code in _VITAL_CODES_ORDERED:
            col = f"{code}_now"
            if col not in snap_df.columns:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            meta = _VITAL_META.get(code)
            if meta is None:
                continue
            title, units, _ = meta
            val_f = float(val)
            vitals_list.append({
                "code":    code,
                "title":   title,
                "value":   round(val_f, 2),
                "units":   units,
                "normalcy": _vital_normalcy(code, val_f),
            })
        enc_vitals[enc_id] = vitals_list

    # ── 5c. Extract active medications at snapshot hour ───────────────────────
    log.info("Extracting active medications at snapshot hour")
    _VASOPRESSOR_CODES = {
        "NOREPINEPHRINE", "VASOPRESSIN", "EPINEPHRINE", "DOPAMINE",
        "PHENYLEPHRINE", "DOBUTAMINE", "MILRINONE",
    }
    enc_reg_dt = encounter.set_index("ENCOUNTER_ID")["REG_DT_TM"]
    meds_dt = meds.copy()
    meds_dt["ADMIN_START_DT_TM"] = pd.to_datetime(meds_dt["ADMIN_START_DT_TM"])
    meds_dt["ADMIN_END_DT_TM"]   = pd.to_datetime(meds_dt["ADMIN_END_DT_TM"])

    enc_medications: dict[int, list] = {}
    for enc_id_m, grp in meds_dt.groupby("ENCOUNTER_ID"):
        admit_dt = pd.to_datetime(enc_reg_dt.get(enc_id_m))
        if pd.isna(admit_dt):
            enc_medications[int(enc_id_m)] = []
            continue
        snap_h  = int(snapshot_hours.get(enc_id_m, 0))
        snap_dt = admit_dt + pd.Timedelta(hours=snap_h)
        active  = grp[
            (grp["ADMIN_START_DT_TM"] <= snap_dt) &
            (grp["ADMIN_END_DT_TM"]   >= snap_dt)
        ]
        med_list = []
        for _, row in active.iterrows():
            code  = str(row["MEDICATION_CD"]).upper()
            rate  = row.get("INFUSION_RATE")
            units = str(row.get("RATE_UNIT_CD", "")) if pd.notna(row.get("RATE_UNIT_CD")) else ""
            dosage = f"{round(float(rate), 4)} {units}".strip() if pd.notna(rate) else ""
            med_list.append({
                "name":          str(row["MEDICATION_TXT"]),
                "code":          code,
                "dosage":        dosage,
                "route":         str(row.get("ROUTE_CD", "IV")),
                "is_vasopressor": code in _VASOPRESSOR_CODES,
                "start":         row["ADMIN_START_DT_TM"].isoformat(),
            })
        enc_medications[int(enc_id_m)] = med_list

    # ── 6. Derive ground truth labels ─────────────────────────────────────────
    log.info("Deriving ground truth labels")
    label_snap_series = pd.Series(label_snap_hours)  # 24h before last hour per encounter

    label_scai  = derive_scai_labels(scai, label_snap_series)
    label_mcs   = derive_mcs_labels(proc, label_snap_series, encounter)
    label_ecmo  = derive_ecmo_labels(proc, label_snap_series, encounter)
    label_vaso_needed, label_vaso_count = derive_vasopressor_labels(meds, encounter, label_snap_series)

    # Mortality: encounter-level — join via PERSON_ID (sidecar has no ENCOUNTER_ID)
    died_by_person = person_aug.set_index("PERSON_ID")["died"]
    label_mortality_s = (
        encounter[["ENCOUNTER_ID", "PERSON_ID"]]
        .assign(label=lambda df: df["PERSON_ID"].map(died_by_person).fillna(0).astype(int))
        .set_index("ENCOUNTER_ID")["label"]
        .rename("label_mort")
    )
    label_los_s = (encounter.set_index("ENCOUNTER_ID")["LOS_HOURS"] / 24).rename("label_los")

    # ── 7. Run SCAI models (ABC and D routed by current stage) ────────────────
    scai_preds = []
    if models.get("scai_abc") is not None or models.get("scai_d") is not None:
        log.info("Running SCAI models on %d patients", len(snap_df))

        # Build features for all patients — no VIS normalisation for new models
        scai_feat_df = scai_build_features(snap_df, scai)
        nan_cols = [c for c in scai_feat_df.columns if "delta" in c.lower() or "slope" in c.lower()]
        scai_feat_df[nan_cols] = scai_feat_df[nan_cols].fillna(0)

        n_patients = len(scai_feat_df)
        all_probs = np.full(n_patients, 0.0)
        all_shap  = [[] for _ in range(n_patients)]

        stage_col = scai_feat_df["SCAI_STAGE_NUM"].values if "SCAI_STAGE_NUM" in scai_feat_df.columns \
                    else np.ones(n_patients, dtype=int)

        for bundle_key, stage_mask_fn in [
            ("scai_abc", lambda s: s <= 2),
            ("scai_d",   lambda s: s == 3),
        ]:
            bundle = models.get(bundle_key)
            if bundle is None:
                continue
            fcols = bundle["feature_cols"]
            idx   = np.where(stage_mask_fn(stage_col))[0]
            if len(idx) == 0:
                continue

            # Ensure all required columns exist (fill unknown OHE/demographic cols with 0)
            missing = [c for c in fcols if c not in scai_feat_df.columns]
            if missing:
                scai_feat_df = pd.concat(
                    [scai_feat_df, pd.DataFrame(0.0, index=scai_feat_df.index, columns=missing)],
                    axis=1,
                )

            X = scai_feat_df.iloc[idx][fcols].astype(np.float32)
            raw_p = bundle["xgboost_model"].predict_proba(X)[:, 1]
            cal_p = bundle["platt_scaler"].predict_proba(raw_p.reshape(-1, 1))[:, 1]
            all_probs[idx] = cal_p

            shap_rows = _shap_top5(bundle_key, bundle["xgboost_model"], X)
            for local_i, orig_i in enumerate(idx):
                all_shap[orig_i] = shap_rows[local_i]

        enc_ids = scai_feat_df["ENCOUNTER_ID"].tolist()
        for i, enc_id in enumerate(enc_ids):
            stage_num = int(stage_col[i])
            threshold = _SCAI_D_THRESHOLD if stage_num == 3 \
                        else _SCAI_ABC_THRESHOLDS.get(stage_num, 0.15)
            scai_preds.append({
                "encounter_id":  int(enc_id),
                "prediction":    round(float(all_probs[i]), 4),
                "alert_flag":    int(all_probs[i] >= threshold),
                "current_stage": _SCAI_STAGE_LABEL.get(stage_num, "B"),
                "label":         int(label_scai.get(enc_id, 0)),
                "shap_values":   all_shap[i],
            })

    # ── 8. Run vasopressor model ──────────────────────────────────────────────
    vaso_preds = []
    if models["vasopressor_binary"] is not None and models["vasopressor_prep"] is not None:
        log.info("Running vasopressor model on %d patients", len(snap_df))
        prep = models["vasopressor_prep"]

        # Vasopressor pipeline needs the full time series (LOCF requires multiple
        # rows per encounter). Build it from wide + SCAI_STAGE_NUM, then filter
        # to the snapshot row after feature engineering.
        wide_with_scai = wide.merge(
            scai[["ENCOUNTER_ID", "HOUR_FROM_ADMIT", "SCAI_STAGE_NUM"]],
            on=["ENCOUNTER_ID", "HOUR_FROM_ADMIT"],
            how="left",
        )
        vaso_all = vaso_build_features(
            df             = wide_with_scai,
            imputer_state  = prep["knn_imputer"],
            iqr_bounds     = prep["iqr_bounds"],
            log_features   = prep["log_features"],
            medications_df = meds,
        )
        # Filter to snapshot hour per encounter
        vaso_feat_df = (
            vaso_all.merge(
                last_hour.rename("snap_hour").reset_index(),
                on="ENCOUNTER_ID",
            )
            .query("HOUR_FROM_ADMIT == snap_hour")
            .drop(columns=["snap_hour"])
            .reset_index(drop=True)
        ) if "HOUR_FROM_ADMIT" in vaso_all.columns else vaso_all

        feat_cols = prep["feature_cols"]
        missing = [c for c in feat_cols if c not in vaso_feat_df.columns]
        for c in missing:
            vaso_feat_df[c] = 0.0
        vaso_X = vaso_feat_df[feat_cols].astype(np.float32)

        # binary_alert: dict with "model" (XGBClassifier) + "platt" (LogisticRegression)
        vaso_bin = models["vasopressor_binary"]
        vaso_raw   = vaso_bin["model"].predict_proba(vaso_X)[:, 1]
        vaso_probs = vaso_bin["platt"].predict_proba(vaso_raw.reshape(-1, 1))[:, 1]

        # ordinal_count: dict with "model" + "label_map" {0:1, 1:2, 2:3}
        vaso_cnt = models["vasopressor_count"]
        count_probs_m = vaso_cnt["model"].predict_proba(vaso_X)
        count_raw     = np.argmax(count_probs_m, axis=1)
        label_map     = vaso_cnt.get("label_map", {0: 1, 1: 2, 2: 3})
        count_preds   = np.array([label_map.get(int(c), int(c)) for c in count_raw])

        # high_severity: dict with "model" (PlattXGB) — call predict_proba directly
        vaso_sev       = models["vasopressor_severity"]
        high_sev_probs = vaso_sev["model"].predict_proba(vaso_X)[:, 1]

        shap_vaso = _shap_top5("vasopressor", vaso_bin["model"], vaso_X)

        for i, enc_id in enumerate(vaso_feat_df["ENCOUNTER_ID"].tolist()):
            vaso_preds.append({
                "encounter_id":  int(enc_id),
                "alert_proba":   round(float(vaso_probs[i]), 4),
                "alert_flag":    int(vaso_probs[i] >= _VASOPRESSOR_THRESHOLD),
                "ordinal_pred":  int(count_preds[i]),
                "ordinal_proba": [round(float(p), 4) for p in count_probs_m[i]],
                "high_severity": round(float(high_sev_probs[i]), 4),
                "label_needed":  int(label_vaso_needed.get(enc_id, 0)),
                "label_count":   int(label_vaso_count.get(enc_id, 0)),
                "shap_values":   shap_vaso[i],
            })

    # ── 9. Run MCS model ──────────────────────────────────────────────────────
    mcs_preds = []
    if models["mcs"] is not None:
        log.info("Building MCS prediction frame")
        mcs_bundle   = models["mcs"]
        mcs_model    = mcs_bundle["model"]
        mcs_threshold = mcs_bundle.get("default_threshold", _MCS_DEFAULT_THRESHOLD)
        mcs_frame = mcs_inf.build_mcs_prediction_frame(tbls, inference_mode=True)

        # One row per encounter: last pre-MCS hour
        mcs_snap = (
            mcs_frame.sort_values("HOUR_FROM_ADMIT")
            .groupby("ENCOUNTER_ID", as_index=False)
            .last()
        )
        mcs_feature_cols = mcs_bundle["feature_cols"]
        mcs_cat_feats    = mcs_bundle.get("cat_feats", [])
        missing = [c for c in mcs_feature_cols if c not in mcs_snap.columns]
        for c in missing:
            mcs_snap[c] = 0.0
        mcs_X = mcs_snap[mcs_feature_cols].copy()
        for col in mcs_cat_feats:
            mcs_X[col] = mcs_X[col].astype("category")
        num_cols = [c for c in mcs_feature_cols if c not in mcs_cat_feats]
        mcs_X[num_cols] = mcs_X[num_cols].astype(np.float32)
        mcs_probs = mcs_model.predict_proba(mcs_X)[:, 1]
        shap_mcs = _shap_top5("mcs", _unwrap_xgb(mcs_model), mcs_X)

        for i, enc_id in enumerate(mcs_snap["ENCOUNTER_ID"].tolist()):
            mcs_preds.append({
                "encounter_id": int(enc_id),
                "prediction":   round(float(mcs_probs[i]), 4),
                "alert_flag":   int(mcs_probs[i] >= mcs_threshold),
                "label":        int(label_mcs.get(enc_id, 0)),
                "shap_values":  shap_mcs[i],
            })

    # ── 10. Run ECMO model ────────────────────────────────────────────────────
    ecmo_preds = []
    if models["va_ecmo"] is not None:
        log.info("Building ECMO prediction frame")
        ecmo_bundle   = models["va_ecmo"]
        ecmo_model    = ecmo_bundle["model"]
        ecmo_threshold = ecmo_bundle.get("default_threshold", _ECMO_DEFAULT_THRESHOLD)
        ecmo_frame = ecmo_inf.build_ecmo_prediction_frame(tbls, inference_mode=True)

        ecmo_snap = (
            ecmo_frame.sort_values("HOUR_FROM_ADMIT")
            .groupby("ENCOUNTER_ID", as_index=False)
            .last()
        )
        ecmo_feature_cols = ecmo_bundle["feature_cols"]
        ecmo_cat_feats    = ecmo_bundle.get("cat_feats", [])
        missing = [c for c in ecmo_feature_cols if c not in ecmo_snap.columns]
        for c in missing:
            ecmo_snap[c] = 0.0
        ecmo_X = ecmo_snap[ecmo_feature_cols].copy()
        for col in ecmo_cat_feats:
            ecmo_X[col] = ecmo_X[col].astype("category")
        num_cols = [c for c in ecmo_feature_cols if c not in ecmo_cat_feats]
        ecmo_X[num_cols] = ecmo_X[num_cols].astype(np.float32)
        ecmo_probs = ecmo_model.predict_proba(ecmo_X)[:, 1]
        shap_ecmo = _shap_top5("va_ecmo", _unwrap_xgb(ecmo_model), ecmo_X)

        for i, enc_id in enumerate(ecmo_snap["ENCOUNTER_ID"].tolist()):
            ecmo_preds.append({
                "encounter_id": int(enc_id),
                "prediction":   round(float(ecmo_probs[i]), 4),
                "alert_flag":   int(ecmo_probs[i] >= ecmo_threshold),
                "label":        int(label_ecmo.get(enc_id, 0)),
                "shap_values":  shap_ecmo[i],
            })

    # ── 11. Run mortality model ───────────────────────────────────────────────
    mort_preds = []
    if models["mortality"] is not None:
        log.info("Materializing mortality features via DuckDB")
        mort_path = materialize_duckdb_patient_features(data_dir)
        mort_df   = pd.read_parquet(mort_path)

        mort_bundle = models["mortality"]
        mort_model  = mort_bundle["pipeline"] if isinstance(mort_bundle, dict) else mort_bundle
        mort_fcols  = (
            mort_bundle.get("feature_names") if isinstance(mort_bundle, dict) else None
        )
        if mort_fcols is None:
            meta_path = model_dir / "lilly" / "xgb_mortality_model_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    mort_fcols = json.load(f).get("feature_columns")

        if mort_fcols:
            missing = [c for c in mort_fcols if c not in mort_df.columns]
            for c in missing:
                mort_df[c] = 0.0
            mort_X = mort_df[mort_fcols]
        else:
            mort_X = mort_df.select_dtypes(include=[np.number])

        # Coerce any object columns (e.g. bucket values stored as Python objects) to numeric
        obj_cols = mort_X.select_dtypes(include="object").columns.tolist()
        for col in obj_cols:
            mort_X = mort_X.copy()
            mort_X[col] = pd.to_numeric(mort_X[col], errors="coerce").fillna(0)

        mort_probs = mort_model.predict_proba(mort_X)[:, 1]
        shap_mort  = _shap_top5("mortality", _unwrap_xgb(mort_model), mort_X if isinstance(mort_X, pd.DataFrame) else pd.DataFrame(mort_X))

        # mort_df uses PERSON_ID — join to encounter to get ENCOUNTER_ID
        person_to_enc = encounter.set_index("PERSON_ID")["ENCOUNTER_ID"].to_dict()
        for i, person_id in enumerate(mort_df["PERSON_ID"].tolist()):
            enc_id = person_to_enc.get(int(person_id), int(person_id))
            mort_preds.append({
                "encounter_id": int(enc_id),
                "prediction":   round(float(mort_probs[i]), 4),
                "label":        int(label_mortality_s.get(enc_id, 0)),
                "shap_values":  shap_mort[i],
            })

    # ── 12. Run LOS model ─────────────────────────────────────────────────────
    los_preds = []
    if models["los"] is not None:
        log.info("Materializing LOS features via DuckDB")
        los_path = materialize_duckdb_los_features(data_dir)
        los_df   = pd.read_parquet(los_path)
        los_model = models["los"]

        meta_path = model_dir / "lilly" / "xgb_los_model_meta.json"
        los_fcols = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                # raw input features the pipeline transformer expects (before one-hot encoding)
                los_fcols = meta.get("feature_columns_raw_included") or meta.get("feature_columns")

        los_X = los_df[los_fcols] if los_fcols else los_df.select_dtypes(include=[np.number])
        los_preds_raw = np.expm1(los_model.predict(los_X)) / 24  # log1p → hours → days

        # SHAP for LOS regressor — unwrap to raw XGBRegressor
        try:
            los_xgb  = _unwrap_xgb(los_model)
            shap_los = _shap_top5("los", los_xgb, los_X if isinstance(los_X, pd.DataFrame) else pd.DataFrame(los_X))
        except Exception as e:
            log.warning("LOS SHAP failed: %s", e)
            shap_los = [[] for _ in range(len(los_X))]

        # los_df uses ENCOUNTER_ID directly
        enc_col = "ENCOUNTER_ID" if "ENCOUNTER_ID" in los_df.columns else los_df.index
        for i, enc_id in enumerate(los_df[enc_col] if "ENCOUNTER_ID" in los_df.columns else los_df.index):
            los_preds.append({
                "encounter_id": int(enc_id),
                "prediction":   round(float(los_preds_raw[i]), 2),
                "label":        round(float(label_los_s.get(enc_id, 0.0)), 2),
                "shap_values":  shap_los[i],
            })

    # ── 13. Compute per-model metrics ─────────────────────────────────────────
    # SCAI + vasopressor: use pre-built test parquets so metrics match training AUC.
    # MCS / ECMO / mortality / LOS: derived from raw-parquet predictions (no test parquet yet).
    log.info("Computing metrics")

    test_dir       = Path(__file__).parent / "data"
    scai_test_path = test_dir / "test_frame.parquet"
    vaso_test_path = test_dir / "test_features_v3.parquet"

    if scai_test_path.exists():
        log.info("SCAI metrics from test_frame.parquet")
        scai_metrics = _scai_test_metrics(models, scai_test_path)
    else:
        log.warning("test_frame.parquet not found — SCAI metrics from raw-parquet predictions")
        scai_metrics = _binary_metrics(
            [p["label"] for p in scai_preds],
            [p["prediction"] for p in scai_preds],
        ) if scai_preds else {}

    if vaso_test_path.exists():
        log.info("Vasopressor metrics from test_features_v3.parquet")
        vaso_metrics = _vasopressor_test_metrics(models, vaso_test_path)
    else:
        log.warning("test_features_v3.parquet not found — vasopressor metrics from raw-parquet predictions")
        vaso_metrics = {
            "alert_auc": _binary_metrics(
                [p["label_needed"] for p in vaso_preds],
                [p["alert_proba"]  for p in vaso_preds],
            )["auc"] if vaso_preds else None,
        }

    def _metrics_from_preds(preds, pred_key="prediction", label_key="label"):
        return _binary_metrics(
            [p[label_key] for p in preds],
            [p[pred_key]  for p in preds],
        )

    metrics = {
        "scai":        scai_metrics,
        "vasopressor": vaso_metrics,
        "mcs":         _metrics_from_preds(mcs_preds)  if mcs_preds  else {},
        "ecmo":        _metrics_from_preds(ecmo_preds) if ecmo_preds else {},
        "mortality":   _metrics_from_preds(mort_preds) if mort_preds else {},
        "los":         _regression_metrics(
                           [p["label"] for p in los_preds],
                           [p["prediction"] for p in los_preds],
                       ) if los_preds else {},
    }

    # ── 14. Build patient list ────────────────────────────────────────────────
    log.info("Assembling patient list")
    patients = []
    for _, row in demo.iterrows():
        enc_id = int(row["ENCOUNTER_ID"])
        patients.append({
            "encounter_id":       enc_id,
            "person_id":          int(row["PERSON_ID"]),
            "name":               str(row.get("name", "")),
            "age":                int(row.get("age", 0)),
            "sex":                str(row.get("SEX_CD", "")),
            "race":               str(row.get("RACE_CD", "")),
            "ethnicity":          str(row.get("ETHNICITY_CD", "")),
            "los_hours":          round(float(row.get("LOS_HOURS", 0)), 1),
            "los_days":           round(float(row.get("los_days", 0)), 2),
            "discharge_disposition": str(row.get("DISCH_DISPOSITION_CD", "")),
            "died":               int(row.get("died", 0)),
            "admit_dt":           str(row["REG_DT_TM"]) if pd.notna(row.get("REG_DT_TM")) else None,
            "discharge_dt":       str(row["DISCH_DT_TM"]) if pd.notna(row.get("DISCH_DT_TM")) else None,
            "vitals":       enc_vitals.get(enc_id, []),
            "medications":  enc_medications.get(enc_id, []),
        })

    # ── 15. Assemble final JSON ───────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_patients":   len(patients),
        "patients":     patients,
        "models": {
            "scai": {
                "description": "Probability SCAI shock stage worsens ≥1 level within 6 hours",
                "output_type": "probability",
                "metrics":     metrics["scai"],
                "predictions": scai_preds,
            },
            "vasopressor": {
                "description": "Vasopressor escalation: alert probability, predicted count, high-severity risk",
                "output_type": "composite",
                "metrics":     metrics["vasopressor"],
                "predictions": vaso_preds,
            },
            "mcs": {
                "description": "Probability patient requires MCS within 12 hours",
                "output_type": "probability",
                "metrics":     metrics["mcs"],
                "predictions": mcs_preds,
            },
            "ecmo": {
                "description": "Probability patient requires VA-ECMO within 12 hours",
                "output_type": "probability",
                "metrics":     metrics["ecmo"],
                "predictions": ecmo_preds,
            },
            "mortality": {
                "description": "Probability of in-hospital death",
                "output_type": "probability",
                "metrics":     metrics["mortality"],
                "predictions": mort_preds,
            },
            "los": {
                "description": "Predicted length of hospital stay (days)",
                "output_type": "numeric",
                "metrics":     metrics["los"],
                "predictions": los_preds,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Wrote %s  (%d patients, 6 models)", output_path, len(patients))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate dashboard_results.json")
    parser.add_argument(
        "--data-dir",
        default="/Users/christiejackett/Downloads/Diabetes_dataset/cardiogenic_shock_package/data",
        help="Directory containing raw .parquet files",
    )
    parser.add_argument(
        "--model-dir",
        default=str(API_DIR / "model_files"),
        help="Directory containing model_files/ subfolders",
    )
    parser.add_argument(
        "--output",
        default=str(API_DIR.parent / "Frontend" / "assets" / "dashboard_results.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()
    run(
        data_dir    = Path(args.data_dir),
        model_dir   = Path(args.model_dir),
        output_path = Path(args.output),
    )


if __name__ == "__main__":
    main()
