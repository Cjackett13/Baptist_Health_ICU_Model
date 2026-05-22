"""
model_development.py
--------------------
Feature engineering + model training for cardiogenic shock progression.

Reads the parquet tables produced by simulate_cardiogenic_shock_data.py and:
  1. Builds an hourly prediction frame (one row per encounter-hour from H+4 onward).
  2. Computes time-windowed features (current value, mean/min/max/slope over 4h
     lookback) for every relevant vital, lab, and hemodynamic.
  3. Constructs the binary label Y_PROGRESSION_6H: SCAI stage worsens by ≥1
     within the next 6 hours.
  4. Splits PATIENT-WISE (no row leakage) into train/val/test (60/20/20).
  5. Trains two models:
       (a) Logistic regression with median-impute + scaler  (transparent baseline)
       (b) HistGradientBoosting with optional random search minimizing validation Brier
           subject to AUROC/AUPRC staying within 0.01 of the default HGB, then optional
           CV isotonic/sigmoid calibration and logit-temperature refinement (rank-preserving).
  6. Performs threshold-free model selection on validation AUPRC.
  7. Saves the fitted pipeline + feature schema + split manifests for testing.

Usage:
    python model_development.py --data-dir ./data --out-dir ./artifacts --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from icu_model_calibration import maybe_refine_logit_temperature

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# Codes that contribute features (numeric)
VITAL_CODES = ["HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP", "URINE_OUT_HR"]
HEMO_CODES  = ["CVP", "PAS", "PAD", "PAM", "PCWP", "CO", "CI", "SVO2", "SVR",
               "CPO", "PAPI"]
LAB_CODES   = ["LACTATE", "CREATININE", "BUN", "NT_PROBNP", "TROPONIN_I",
               "PH", "PCO2", "HCO3", "AST", "ALT", "WBC", "HGB", "PLT", "INR"]
CALC_CODES  = ["VIS"]

NUMERIC_CODES = VITAL_CODES + HEMO_CODES + LAB_CODES + CALC_CODES

# Categorical / static features
STATIC_NUMERIC = ["AGE", "WEIGHT_KG", "BSA_M2"]
STATIC_CAT     = ["SEX_CD", "RACE_CD", "ETHNICITY_CD", "ADMIT_TYPE_CD",
                  "ADMIT_SRC_CD", "PRINCIPAL_DX_CATEGORY"]
TIME_FEATURES  = ["HOUR_FROM_ADMIT", "CURRENT_STAGE_NUM",
                  "ON_MCS", "ON_INTUBATION", "ON_CRRT",
                  "N_VASOPRESSORS_ACTIVE", "N_INOTROPES_ACTIVE"]

LOOKBACK_HOURS = 4
HORIZON_HOURS  = 6  # label window
MIN_HOUR       = 4  # start of valid prediction times

# DX category coarse-grouping
def dx_category(icd: str) -> str:
    if not isinstance(icd, str): return "OTHER"
    if icd.startswith("I21"): return "ACUTE_MI"
    if icd.startswith("I50"): return "ADHF"
    if icd.startswith("I46") or icd.startswith("I49"): return "ARRHYTHMIA_ARREST"
    if icd.startswith("I40") or icd.startswith("I42"): return "MYOCARDITIS_CMP"
    if icd.startswith("I71") or icd.startswith("I35"): return "AORTIC_VALVE"
    if icd.startswith("Z95"): return "POST_CARDIOTOMY"
    return "OTHER"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    names = ["person","encounter","diagnosis","clinical_event",
             "medication_admin","procedure_event","scai_stage_hourly"]
    return {n: pd.read_parquet(os.path.join(data_dir, f"{n}.parquet")) for n in names}


# ---------------------------------------------------------------------------
# Long → wide hourly feature frame
# ---------------------------------------------------------------------------

def build_hourly_wide(ce: pd.DataFrame, scai: pd.DataFrame) -> pd.DataFrame:
    """Pivot CLINICAL_EVENT to wide hourly frame keyed on (ENCOUNTER_ID, HOUR_FROM_ADMIT).
    The SCAI_STAGE_HOURLY table defines the canonical hour grid."""
    # Filter to numeric codes of interest
    ce_f = ce[ce.EVENT_CD.isin(NUMERIC_CODES)].copy()

    # Attach HOUR_FROM_ADMIT via scai grid join on EVENT_DT_TM
    ce_f = ce_f.merge(
        scai[["ENCOUNTER_ID","EVENT_DT_TM","HOUR_FROM_ADMIT","SCAI_STAGE_CD","SCAI_STAGE_NUM"]],
        left_on=["ENCOUNTER_ID","EVENT_END_DT_TM"],
        right_on=["ENCOUNTER_ID","EVENT_DT_TM"],
        how="inner",
    )
    # If multiple values within same hour for same code (e.g., labs + monitor),
    # keep most recent. Otherwise we’d have duplicates.
    ce_f = (ce_f.sort_values("EVENT_END_DT_TM")
                .groupby(["ENCOUNTER_ID","HOUR_FROM_ADMIT","EVENT_CD"], as_index=False)
                .last())

    wide = ce_f.pivot_table(
        index=["ENCOUNTER_ID","HOUR_FROM_ADMIT"],
        columns="EVENT_CD",
        values="RESULT_VAL",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None

    # Ensure all numeric codes are present as columns
    for c in NUMERIC_CODES:
        if c not in wide.columns:
            wide[c] = np.nan

    return wide


# ---------------------------------------------------------------------------
# Treatment state per (encounter, hour) — MCS / intubation / CRRT / pressor counts
# ---------------------------------------------------------------------------

def build_treatment_state(
    enc: pd.DataFrame,
    med: pd.DataFrame,
    proc: pd.DataFrame,
    scai: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one row per (encounter, hour) indicating concurrent treatments."""
    hours = scai[["ENCOUNTER_ID","HOUR_FROM_ADMIT","EVENT_DT_TM"]].copy()

    # MCS active flag
    mcs_proc = proc[proc.PROC_CATEGORY_CD == "MCS"][
        ["ENCOUNTER_ID","PROC_START_DT_TM","PROC_END_DT_TM"]
    ].rename(columns={"PROC_START_DT_TM":"START","PROC_END_DT_TM":"END"})
    hours = _flag_interval_active(hours, mcs_proc, "ON_MCS")

    # Intubation
    intub = proc[proc.PROC_CATEGORY_CD == "INTUBATION"][
        ["ENCOUNTER_ID","PROC_START_DT_TM","PROC_END_DT_TM"]
    ].rename(columns={"PROC_START_DT_TM":"START","PROC_END_DT_TM":"END"})
    hours = _flag_interval_active(hours, intub, "ON_INTUBATION")

    # CRRT
    crrt = proc[proc.PROC_CATEGORY_CD == "CRRT"][
        ["ENCOUNTER_ID","PROC_START_DT_TM","PROC_END_DT_TM"]
    ].rename(columns={"PROC_START_DT_TM":"START","PROC_END_DT_TM":"END"})
    hours = _flag_interval_active(hours, crrt, "ON_CRRT")

    # Active pressors / inotropes count
    PRESSORS = {"NOREPINEPHRINE","EPINEPHRINE","VASOPRESSIN","PHENYLEPHRINE","DOPAMINE"}
    INOTROPES = {"DOBUTAMINE","MILRINONE"}

    med_p = med[med.MEDICATION_CD.isin(PRESSORS)][
        ["ENCOUNTER_ID","ADMIN_START_DT_TM","ADMIN_END_DT_TM","MEDICATION_CD"]
    ].rename(columns={"ADMIN_START_DT_TM":"START","ADMIN_END_DT_TM":"END"})
    med_i = med[med.MEDICATION_CD.isin(INOTROPES)][
        ["ENCOUNTER_ID","ADMIN_START_DT_TM","ADMIN_END_DT_TM","MEDICATION_CD"]
    ].rename(columns={"ADMIN_START_DT_TM":"START","ADMIN_END_DT_TM":"END"})

    hours = _count_active_intervals(hours, med_p, "N_VASOPRESSORS_ACTIVE")
    hours = _count_active_intervals(hours, med_i, "N_INOTROPES_ACTIVE")

    return hours.drop(columns=["EVENT_DT_TM"])


def _flag_interval_active(hours: pd.DataFrame, intervals: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mark hours where any interval is active for the encounter."""
    if intervals.empty:
        hours[col] = 0
        return hours
    merged = hours.merge(intervals, on="ENCOUNTER_ID", how="left")
    active = (merged.EVENT_DT_TM >= merged.START) & \
             ((merged.END.isna()) | (merged.EVENT_DT_TM <= merged.END))
    merged["_act"] = active.astype(int)
    flag = (merged.groupby(["ENCOUNTER_ID","HOUR_FROM_ADMIT"])["_act"]
                  .max().reset_index().rename(columns={"_act": col}))
    return hours.merge(flag, on=["ENCOUNTER_ID","HOUR_FROM_ADMIT"], how="left")\
                .fillna({col: 0})


def _count_active_intervals(hours: pd.DataFrame, intervals: pd.DataFrame, col: str) -> pd.DataFrame:
    """Count distinct active intervals per (encounter, hour)."""
    if intervals.empty:
        hours[col] = 0
        return hours
    merged = hours.merge(intervals, on="ENCOUNTER_ID", how="left")
    active = (merged.EVENT_DT_TM >= merged.START) & \
             ((merged.END.isna()) | (merged.EVENT_DT_TM <= merged.END))
    merged["_act"] = active.astype(int)
    cnt = (merged.groupby(["ENCOUNTER_ID","HOUR_FROM_ADMIT"])["_act"]
                 .sum().reset_index().rename(columns={"_act": col}))
    return hours.merge(cnt, on=["ENCOUNTER_ID","HOUR_FROM_ADMIT"], how="left")\
                .fillna({col: 0})


# ---------------------------------------------------------------------------
# Time-windowed features (last 4h, last-observation-carried-forward)
# ---------------------------------------------------------------------------

def build_window_features(wide: pd.DataFrame, lookback: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """For each numeric code, compute: <code>_now, _mean<lb>h, _min<lb>h,
    _max<lb>h, _slope<lb>h, _delta<lb>h. Operates on LOCF-imputed series.

    LOCF is performed within encounter only (no cross-patient leakage).
    Built via a single pd.concat to avoid frame-fragmentation overhead.
    """
    wide = wide.sort_values(["ENCOUNTER_ID","HOUR_FROM_ADMIT"]).copy()

    # Forward-fill within each encounter
    locf = wide.copy()
    locf[NUMERIC_CODES] = locf.groupby("ENCOUNTER_ID")[NUMERIC_CODES].ffill()

    grouped = locf.groupby("ENCOUNTER_ID")
    rolled_mean = grouped[NUMERIC_CODES].rolling(window=lookback, min_periods=1).mean().reset_index(level=0, drop=True)
    rolled_min  = grouped[NUMERIC_CODES].rolling(window=lookback, min_periods=1).min().reset_index(level=0, drop=True)
    rolled_max  = grouped[NUMERIC_CODES].rolling(window=lookback, min_periods=1).max().reset_index(level=0, drop=True)
    prev        = grouped[NUMERIC_CODES].shift(lookback)

    parts = [locf[["ENCOUNTER_ID","HOUR_FROM_ADMIT"]].reset_index(drop=True)]
    parts.append(locf[NUMERIC_CODES].rename(columns={c: f"{c}_now" for c in NUMERIC_CODES}).reset_index(drop=True))
    parts.append(rolled_mean.rename(columns={c: f"{c}_mean{lookback}h" for c in NUMERIC_CODES}).reset_index(drop=True))
    parts.append(rolled_min .rename(columns={c: f"{c}_min{lookback}h"  for c in NUMERIC_CODES}).reset_index(drop=True))
    parts.append(rolled_max .rename(columns={c: f"{c}_max{lookback}h"  for c in NUMERIC_CODES}).reset_index(drop=True))

    delta = (locf[NUMERIC_CODES].reset_index(drop=True) - prev[NUMERIC_CODES].reset_index(drop=True))
    parts.append(delta.rename(columns={c: f"{c}_delta{lookback}h" for c in NUMERIC_CODES}))
    parts.append((delta / lookback).rename(columns={c: f"{c}_slope{lookback}h" for c in NUMERIC_CODES}))

    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_labels(scai: pd.DataFrame, horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    """Y = 1 if SCAI_STAGE_NUM increases by >=1 in the next `horizon` hours,
    relative to the current row's stage. Excludes rows where current stage == 4."""
    scai = scai.sort_values(["ENCOUNTER_ID","HOUR_FROM_ADMIT"]).copy()
    g = scai.groupby("ENCOUNTER_ID")["SCAI_STAGE_NUM"]
    # Future max stage within next `horizon` hours (exclude current)
    future_max = (
        g.transform(lambda s: s.shift(-1).rolling(window=horizon, min_periods=1).max())
    )
    scai["FUTURE_MAX_STAGE_NUM"] = future_max
    scai["Y_PROGRESSION_6H"] = (
        (scai.FUTURE_MAX_STAGE_NUM > scai.SCAI_STAGE_NUM).fillna(False).astype(int)
    )
    return scai


# ---------------------------------------------------------------------------
# Master prediction frame
# ---------------------------------------------------------------------------

def build_prediction_frame(tbls: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the modeling matrix: one row per encounter-hour prediction time."""
    print("   pivoting clinical_event …")
    wide = build_hourly_wide(tbls["clinical_event"], tbls["scai_stage_hourly"])

    print("   computing window features …")
    feats = build_window_features(wide)

    print("   building treatment state …")
    tx = build_treatment_state(
        tbls["encounter"], tbls["medication_admin"],
        tbls["procedure_event"], tbls["scai_stage_hourly"]
    )

    print("   building labels …")
    labeled = build_labels(tbls["scai_stage_hourly"])

    # Merge everything
    df = (labeled[["ENCOUNTER_ID","HOUR_FROM_ADMIT","EVENT_DT_TM",
                   "SCAI_STAGE_NUM","Y_PROGRESSION_6H"]]
          .rename(columns={"SCAI_STAGE_NUM": "CURRENT_STAGE_NUM"})
          .merge(feats, on=["ENCOUNTER_ID","HOUR_FROM_ADMIT"], how="left")
          .merge(tx, on=["ENCOUNTER_ID","HOUR_FROM_ADMIT"], how="left"))

    # Attach static encounter/person features
    enc = tbls["encounter"]
    person = tbls["person"]
    diag = tbls["diagnosis"]

    principal = (diag[diag.DIAG_PRIORITY == 1]
                 [["ENCOUNTER_ID","NOMENCLATURE_CD"]]
                 .rename(columns={"NOMENCLATURE_CD":"PRINCIPAL_DX_CODE"}))
    principal["PRINCIPAL_DX_CATEGORY"] = principal["PRINCIPAL_DX_CODE"].map(dx_category)

    static = (enc[["ENCOUNTER_ID","PERSON_ID","ADMIT_TYPE_CD","ADMIT_SRC_CD",
                   "REG_DT_TM","DISCH_DT_TM"]]
              .merge(person[["PERSON_ID","BIRTH_DT_TM","SEX_CD","RACE_CD","ETHNICITY_CD"]],
                     on="PERSON_ID", how="left")
              .merge(principal[["ENCOUNTER_ID","PRINCIPAL_DX_CATEGORY","PRINCIPAL_DX_CODE"]],
                     on="ENCOUNTER_ID", how="left"))
    static["AGE"] = ((static.REG_DT_TM - static.BIRTH_DT_TM).dt.days / 365.25).astype(int)
    # Weight/BSA aren't in tables (could pull from a separate observation),
    # so impute population values per sex.
    static["WEIGHT_KG"] = np.where(static.SEX_CD == "M", 82.0, 70.0)
    static["BSA_M2"]    = np.where(static.SEX_CD == "M", 1.97, 1.74)

    df = df.merge(static[["ENCOUNTER_ID","PERSON_ID","AGE","WEIGHT_KG","BSA_M2",
                          "SEX_CD","RACE_CD","ETHNICITY_CD","ADMIT_TYPE_CD",
                          "ADMIT_SRC_CD","PRINCIPAL_DX_CATEGORY"]],
                  on="ENCOUNTER_ID", how="left")

    # Exclusion: drop rows where current stage is E (no further progression possible)
    df = df[df.CURRENT_STAGE_NUM < 4].copy()
    # Drop rows where hour < MIN_HOUR (need lookback to be meaningful)
    df = df[df.HOUR_FROM_ADMIT >= MIN_HOUR].copy()
    # Drop rows whose label is NaN (end of stay with no horizon)
    df = df.dropna(subset=["Y_PROGRESSION_6H"]).copy()
    df["Y_PROGRESSION_6H"] = df["Y_PROGRESSION_6H"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Split — patient-wise, no leakage
# ---------------------------------------------------------------------------

def patient_split(df: pd.DataFrame, seed: int = 42
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) indices into df, partitioned by PERSON_ID."""
    groups = df["PERSON_ID"].values
    # Train/temp 60/40
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))
    # Val/test 50/50 of temp -> 20/20 overall
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(df.iloc[temp_idx], groups=groups[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Feature schema for the modeling pipeline
# ---------------------------------------------------------------------------

def feature_schema(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_feature_cols, categorical_feature_cols) present in df."""
    num_window = []
    for c in NUMERIC_CODES:
        num_window.extend([
            f"{c}_now", f"{c}_mean{LOOKBACK_HOURS}h",
            f"{c}_min{LOOKBACK_HOURS}h", f"{c}_max{LOOKBACK_HOURS}h",
            f"{c}_delta{LOOKBACK_HOURS}h", f"{c}_slope{LOOKBACK_HOURS}h",
        ])
    num_feats = num_window + STATIC_NUMERIC + TIME_FEATURES
    cat_feats = STATIC_CAT
    # Keep only those actually in df
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = [c for c in cat_feats if c in df.columns]
    return num_feats, cat_feats


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def make_logreg_pipeline(num_feats: List[str], cat_feats: List[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_feats),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", _get_onehot()),
            ]), cat_feats),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=0.5, solver="lbfgs")),
    ])


def _default_hgb_kwargs(random_state: int = 42) -> Dict[str, Any]:
    return dict(
        max_iter=400,
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
    )


def make_hgb_pipeline(
    num_feats: List[str],
    cat_feats: List[str],
    clf_kwargs: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Pipeline:
    """HistGradientBoosting handles NaN natively; categorical via OneHot."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_feats),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", _get_onehot()),
            ]), cat_feats),
        ],
        remainder="drop",
    )
    kw = _default_hgb_kwargs(random_state)
    if clf_kwargs:
        kw.update(clf_kwargs)
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(**kw)),
    ])


def _sample_hgb_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Random hyperparameters biased toward smoother scores (often lower Brier)."""
    max_depth_roll = rng.integers(0, 10)
    if max_depth_roll == 0:
        max_depth: Optional[int] = None
    else:
        max_depth = int(rng.choice([4, 5, 6, 7, 8, 9, 10, 12]))
    return dict(
        max_iter=int(rng.integers(280, 750)),
        learning_rate=float(10 ** rng.uniform(np.log10(0.018), np.log10(0.11))),
        max_depth=max_depth,
        max_leaf_nodes=int(rng.integers(24, 96)),
        min_samples_leaf=int(rng.integers(18, 120)),
        l2_regularization=float(10 ** rng.uniform(-1.2, 1.15)),
        max_bins=int(rng.choice([128, 160, 192, 224, 255])),
        validation_fraction=float(rng.uniform(0.08, 0.18)),
        n_iter_no_change=int(rng.integers(14, 38)),
        class_weight=rng.choice([None, "balanced"]),
    )


def tune_hgb_constrained_min_brier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    num_feats: List[str],
    cat_feats: List[str],
    *,
    seed: int,
    n_iter: int,
    margin: float = 0.01,
) -> Tuple[Pipeline, Dict[str, Any], List[Dict[str, Any]]]:
    """Random search on train; pick lowest validation Brier without dropping AUROC/AUPRC
    vs the default HGB by more than ``margin``."""
    rng = np.random.default_rng(seed)
    base = make_hgb_pipeline(num_feats, cat_feats, random_state=seed)
    base.fit(X_train, y_train)
    floor = evaluate(base, X_val, y_val)
    best_pipe = base
    best_metrics = floor
    best_tag = "default_hgb"
    trials: List[Dict[str, Any]] = [{"tag": best_tag, "params": None, **floor}]

    for i in range(n_iter):
        rs = int(rng.integers(0, 2**31 - 1))
        params = _sample_hgb_params(rng)
        pipe = make_hgb_pipeline(num_feats, cat_feats, clf_kwargs=params, random_state=rs)
        pipe.fit(X_train, y_train)
        m = evaluate(pipe, X_val, y_val)
        trials.append({"tag": f"trial_{i}", "params": params, **m})
        ok = (
            m["auroc"] >= floor["auroc"] - margin
            and m["auprc"] >= floor["auprc"] - margin
        )
        if ok and m["brier"] < best_metrics["brier"]:
            best_metrics = m
            best_pipe = pipe
            best_tag = f"trial_{i}"

    meta = dict(
        selected=best_tag,
        floor_metrics=floor,
        best_metrics=best_metrics,
        margin=margin,
    )
    return best_pipe, meta, trials


def maybe_calibrate_hgb(
    fitted_hgb_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    cv: int = 3,
    margin: float = 0.01,
    floor_auroc: float,
    floor_auprc: float,
) -> Tuple[Any, Dict[str, float]]:
    """Try isotonic and sigmoid CV calibration on train; keep the option with lowest
    validation Brier that stays within ``margin`` of the default HGB AUROC/AUPRC."""
    raw = evaluate(fitted_hgb_pipeline, X_val, y_val)
    best: Optional[Tuple[Any, Dict[str, float], str]] = None

    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(
            clone(fitted_hgb_pipeline),
            method=method,
            cv=cv,
            ensemble=True,
        )
        cal.fit(X_train, y_train)
        m = evaluate(cal, X_val, y_val)
        ok = (
            m["auroc"] >= floor_auroc - margin
            and m["auprc"] >= floor_auprc - margin
            and m["brier"] < raw["brier"] - 1e-6
        )
        if not ok:
            continue
        cand = (cal, m, method)
        if best is None or m["brier"] < best[1]["brier"]:
            best = cand

    if best is not None:
        cal, m, method = best
        return cal, dict(
            used_calibration=True,
            calibration_method=method,
            **m,
            brier_before_cal=raw["brier"],
        )
    return fitted_hgb_pipeline, dict(
        used_calibration=False,
        **raw,
        brier_before_cal=raw["brier"],
    )


def _get_onehot():
    """sklearn renamed `sparse` -> `sparse_output` in 1.2; handle both."""
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def evaluate(model, X, y) -> Dict[str, float]:
    p = model.predict_proba(X)[:, 1]
    return dict(
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        base_rate=float(np.mean(y)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir",  default="./artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tune-iter",
        type=int,
        default=48,
        help="Random HGB trials after the default baseline (0 skips search).",
    )
    ap.add_argument(
        "--disc-margin",
        type=float,
        default=0.01,
        help="Max allowed drop vs default HGB for AUROC and AUPRC when tuning/calibrating.",
    )
    ap.add_argument("--no-tune", action="store_true", help="Train default HGB only.")
    ap.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip CalibratedClassifierCV on the tuned HGB.",
    )
    ap.add_argument(
        "--calib-cv",
        type=int,
        default=3,
        help="CV folds for CalibratedClassifierCV (train-only).",
    )
    ap.add_argument(
        "--no-temperature",
        action="store_true",
        help="Skip validation-tuned logit temperature (rank-preserving Brier polish).",
    )
    args = ap.parse_args()

    print("[1/6] Loading tables …")
    tbls = load_tables(args.data_dir)
    for n, df in tbls.items():
        print(f"   {n:24s} rows={len(df):>9,}")

    print("[2/6] Building prediction frame …")
    df = build_prediction_frame(tbls)
    print(f"   prediction frame rows = {len(df):,}")
    print(f"   class balance Y=1     = {df.Y_PROGRESSION_6H.mean():.4f}")

    print("[3/6] Patient-wise train/val/test split …")
    tr, va, te = patient_split(df, seed=args.seed)
    print(f"   train rows = {len(tr):,} (patients={df.iloc[tr].PERSON_ID.nunique():,})")
    print(f"   val   rows = {len(va):,} (patients={df.iloc[va].PERSON_ID.nunique():,})")
    print(f"   test  rows = {len(te):,} (patients={df.iloc[te].PERSON_ID.nunique():,})")

    num_feats, cat_feats = feature_schema(df)
    print(f"   numeric features = {len(num_feats)}, categorical = {len(cat_feats)}")

    X = df[num_feats + cat_feats]
    y = df["Y_PROGRESSION_6H"].values

    print("[4/6] Training logistic regression baseline …")
    lr = make_logreg_pipeline(num_feats, cat_feats)
    lr.fit(X.iloc[tr], y[tr])
    lr_val = evaluate(lr, X.iloc[va], y[va])
    print(f"   LR val:  AUROC={lr_val['auroc']:.3f}  AUPRC={lr_val['auprc']:.3f}  "
          f"Brier={lr_val['brier']:.3f}  base={lr_val['base_rate']:.3f}")

    print("[5/6] Training HistGradientBoosting (tune + optional calibration) …")
    hgb_tune_meta: Dict[str, Any] = {}
    hgb_cal_meta: Dict[str, Any] = {}
    floor_m: Dict[str, float]

    if args.no_tune:
        hgb = make_hgb_pipeline(num_feats, cat_feats, random_state=args.seed)
        hgb.fit(X.iloc[tr], y[tr])
        floor_m = evaluate(hgb, X.iloc[va], y[va])
        hgb_tune_meta = {"skipped": True, "floor_metrics": floor_m}
    else:
        n_iter = max(0, int(args.tune_iter))
        hgb_inner, hgb_tune_meta, _trials = tune_hgb_constrained_min_brier(
            X.iloc[tr],
            y[tr],
            X.iloc[va],
            y[va],
            num_feats,
            cat_feats,
            seed=args.seed,
            n_iter=n_iter,
            margin=float(args.disc_margin),
        )
        hgb = hgb_inner
        floor_m = hgb_tune_meta["floor_metrics"]

    hgb_val = evaluate(hgb, X.iloc[va], y[va])
    print(
        f"   HGB val (pre-cal): AUROC={hgb_val['auroc']:.4f}  "
        f"AUPRC={hgb_val['auprc']:.4f}  Brier={hgb_val['brier']:.4f}"
    )
    if not args.no_tune and hgb_tune_meta and not hgb_tune_meta.get("skipped"):
        fm = floor_m
        print(
            f"   tune: selected={hgb_tune_meta.get('selected')}  "
            f"floor AUROC/AUPRC/Brier="
            f"{fm['auroc']:.4f}/{fm['auprc']:.4f}/{fm['brier']:.4f}"
        )

    if args.no_calibrate:
        hgb_cal_meta = {"used_calibration": False, **hgb_val}
    else:
        hgb, hgb_cal_meta = maybe_calibrate_hgb(
            hgb,
            X.iloc[tr],
            y[tr],
            X.iloc[va],
            y[va],
            cv=max(2, int(args.calib_cv)),
            margin=float(args.disc_margin),
            floor_auroc=floor_m["auroc"],
            floor_auprc=floor_m["auprc"],
        )

    hgb_temp_meta: Dict[str, Any] = {}
    if not args.no_temperature:
        hgb, hgb_temp_meta = maybe_refine_logit_temperature(hgb, X.iloc[va], y[va])

    hgb_val = evaluate(hgb, X.iloc[va], y[va])
    print(
        f"   HGB val (final):  AUROC={hgb_val['auroc']:.4f}  "
        f"AUPRC={hgb_val['auprc']:.4f}  Brier={hgb_val['brier']:.4f}  "
        f"calib={hgb_cal_meta.get('used_calibration', False)}"
        f"{(' (' + str(hgb_cal_meta.get('calibration_method')) + ')') if hgb_cal_meta.get('calibration_method') else ''}  "
        f"temp={hgb_temp_meta.get('used_temperature', False)}"
        f"{(' T=' + str(round(hgb_temp_meta.get('temperature', 1.0), 4))) if hgb_temp_meta.get('used_temperature') else ''}"
    )

    # Pick best by val AUPRC
    best_name = "hgb" if hgb_val["auprc"] >= lr_val["auprc"] else "lr"
    best_model = hgb if best_name == "hgb" else lr
    print(f"   Model selected on validation AUPRC: {best_name.upper()}")

    print("[6/6] Persisting artifacts …")
    artifact = dict(
        feature_schema=dict(numeric=num_feats, categorical=cat_feats),
        config=dict(
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            min_hour=MIN_HOUR,
            seed=args.seed,
            best_model=best_name,
        ),
        val_metrics=dict(lr=lr_val, hgb=hgb_val),
        hgb_tuning=hgb_tune_meta,
        hgb_calibration=hgb_cal_meta,
        hgb_temperature=hgb_temp_meta,
        hgb_default_val_floor=floor_m,
    )
    with open(os.path.join(args.out_dir, "logreg.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(args.out_dir, "hgb.pkl"), "wb") as f:
        pickle.dump(hgb, f)
    with open(os.path.join(args.out_dir, "training_manifest.json"), "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    # Persist the split indices (so testing script uses the same test set)
    df_for_test = df.iloc[te].copy()
    df_for_test.to_parquet(os.path.join(args.out_dir, "test_frame.parquet"), index=False)
    df.iloc[va].to_parquet(os.path.join(args.out_dir, "val_frame.parquet"), index=False)

    print("\nDone.")
    print(f"   artifacts written to {args.out_dir}/")
    print( "   files: logreg.pkl, hgb.pkl, training_manifest.json,")
    print( "          val_frame.parquet, test_frame.parquet")


if __name__ == "__main__":
    main()
