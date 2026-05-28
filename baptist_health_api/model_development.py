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
       (b) Gradient boosted trees (sklearn HistGradientBoosting, native NaN handling)
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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


def load_tables_for_encounters(
    data_dir: str,
    encounter_ids: set | frozenset,
) -> Dict[str, pd.DataFrame]:
    """Load parquet with row-group filters so huge tables are not fully read into RAM."""
    encs = [int(x) if isinstance(x, (np.integer,)) else x for x in encounter_ids]
    enc_filter = [("ENCOUNTER_ID", "in", encs)]
    data_dir = os.path.abspath(data_dir)

    encounter = pd.read_parquet(
        os.path.join(data_dir, "encounter.parquet"), filters=enc_filter
    )
    person_ids = encounter["PERSON_ID"].dropna().unique().tolist()
    person_filter = [("PERSON_ID", "in", person_ids)]

    out: Dict[str, pd.DataFrame] = {
        "encounter": encounter,
        "person": pd.read_parquet(
            os.path.join(data_dir, "person.parquet"), filters=person_filter
        ),
        "diagnosis": pd.read_parquet(
            os.path.join(data_dir, "diagnosis.parquet"), filters=enc_filter
        ),
        "clinical_event": pd.read_parquet(
            os.path.join(data_dir, "clinical_event.parquet"), filters=enc_filter
        ),
        "medication_admin": pd.read_parquet(
            os.path.join(data_dir, "medication_admin.parquet"), filters=enc_filter
        ),
        "procedure_event": pd.read_parquet(
            os.path.join(data_dir, "procedure_event.parquet"), filters=enc_filter
        ),
        "scai_stage_hourly": pd.read_parquet(
            os.path.join(data_dir, "scai_stage_hourly.parquet"), filters=enc_filter
        ),
    }
    return out


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


def make_hgb_pipeline(num_feats: List[str], cat_feats: List[str]) -> Pipeline:
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
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0,
            class_weight="balanced", random_state=42,
        )),
    ])


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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    print("[5/6] Training HistGradientBoosting …")
    hgb = make_hgb_pipeline(num_feats, cat_feats)
    hgb.fit(X.iloc[tr], y[tr])
    hgb_val = evaluate(hgb, X.iloc[va], y[va])
    print(f"   HGB val: AUROC={hgb_val['auroc']:.3f}  AUPRC={hgb_val['auprc']:.3f}  "
          f"Brier={hgb_val['brier']:.3f}")

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
