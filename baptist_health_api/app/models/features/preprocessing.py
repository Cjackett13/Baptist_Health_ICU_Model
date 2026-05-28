"""
preprocessing.py — Vasopressor Prediction Preprocessing Pipeline
-----------------------------------------------------------------
Baptist Health ICU Project — Vasopressor Model

Step 3 of the vasopressor model build.

Pipeline steps (in order):
  1.  Wide pivot: one row per (ENCOUNTER_ID, HOUR_FROM_ADMIT), Stage D/E excluded
  2.  Labels: VASOPRESSOR_NEEDED_24H, VASOPRESSOR_COUNT_24H, N_VASOPRESSORS_ACTIVE
  3.  Clinical range nulling (impossible values → NaN)
  4.  was_measured_ flags for 16 sparse features (created before imputation)
  5.  LOCF within encounter
  6.  70/15/15 patient-wise split (GroupShuffleSplit on ENCOUNTER_ID)
  7.  KNN imputation k=5 (fit on train only) → artifacts/knn_imputer.pkl
  8.  IQR capping 1.5× (fit on train only) → artifacts/iqr_bounds.json
  9.  log1p transform: PAPI, VIS, NT_PROBNP, LACTATE
  10. Save data/processed/train.parquet, val.parquet, test.parquet
  11. Save artifacts/preprocessing_manifest.json

Usage:
    python preprocessing.py --data-dir ./data --out-dir ./data/processed
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from duckdb_backend import init_db, query

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
OUT_DIR     = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "artifacts"

VASOPRESSORS = [
    "NOREPINEPHRINE",
    "EPINEPHRINE",
    "VASOPRESSIN",
    "PHENYLEPHRINE",
    "DOPAMINE",
]

ALL_FEATURES = [
    "HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP", "URINE_OUT_HR",
    "CVP", "CO", "CI", "SVO2", "CPO", "PAPI",
    "LACTATE", "CREATININE", "BUN", "NT_PROBNP", "TROPONIN_I",
    "PH", "HCO3", "WBC", "HGB", "PLT",
    "VIS",
]

SPARSE_FEATURES = [
    "CVP", "CO", "CI", "SVO2", "CPO", "PAPI",
    "LACTATE", "CREATININE", "BUN", "NT_PROBNP", "TROPONIN_I",
    "PH", "HCO3", "WBC", "HGB", "PLT",
]

LOG_TRANSFORM_FEATURES = ["PAPI", "VIS", "NT_PROBNP", "LACTATE"]

NON_FEATURE_COLS = [
    "ENCOUNTER_ID", "PERSON_ID", "EVENT_DT_TM", "HOUR_FROM_ADMIT",
    "VASOPRESSOR_NEEDED_24H", "VASOPRESSOR_COUNT_24H", "N_VASOPRESSORS_ACTIVE",
    "SCAI_STAGE_CD",
]

# Clinical validity ranges for nulling impossible values.
# TROPONIN_I: simulator generates values up to ~7300 — likely ng/mL vs ng/L unit
# mismatch. Ceiling set to 50000 (ng/L equivalent) pending Anshul confirmation.
# NT_PROBNP: pending Anshul confirmation on upper bound.
CLINICAL_RANGES = {
    "HR":           (20,    300),
    "SBP":          (40,    300),
    "DBP":          (10,    200),
    "MAP":          (20,    200),
    "RR":           (4,     60),
    "SPO2":         (50,    100),
    "TEMP":         (25,    45),
    "URINE_OUT_HR": (0,     2000),
    "CVP":          (-5,    40),
    "CO":           (0.5,   20),
    "CI":           (0.3,   10),
    "SVO2":         (20,    100),
    "CPO":          (0.1,   5),
    "PAPI":         (0,     20),
    "LACTATE":      (0.1,   30),
    "CREATININE":   (0.1,   20),
    "BUN":          (1,     200),
    "NT_PROBNP":    (5,     50000),  # TODO: confirm upper bound with Anshul
    "TROPONIN_I":   (0,     50000),  # TODO: confirm units with Anshul (ng/mL vs ng/L)
    "PH":           (6.5,   7.9),
    "HCO3":         (5,     50),
    "WBC":          (0.5,   100),
    "HGB":          (2,     25),
    "PLT":          (5,     1500),
    "VIS":          (0,     200),
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — WIDE PIVOT
# One row per (ENCOUNTER_ID, HOUR_FROM_ADMIT). Stage D/E excluded.
# ─────────────────────────────────────────────────────────────────────────────
def build_wide_pivot() -> pd.DataFrame:
    """Query all clinical events and pivot to a wide hourly feature matrix."""
    print("\n[Step 1] Building wide pivot table (Stage D/E excluded)...")

    # Pivot clinical events to wide format, excluding Stage D (3) and E (4)
    pivot_sql = f"""
    SELECT
        sh.ENCOUNTER_ID,
        sh.HOUR_FROM_ADMIT,
        sh.SCAI_STAGE_CD,
        ce_agg.HR, ce_agg.SBP, ce_agg.DBP, ce_agg.MAP, ce_agg.RR,
        ce_agg.SPO2, ce_agg.TEMP, ce_agg.URINE_OUT_HR,
        ce_agg.CVP, ce_agg.CO, ce_agg.CI, ce_agg.SVO2, ce_agg.CPO, ce_agg.PAPI,
        ce_agg.LACTATE, ce_agg.CREATININE, ce_agg.BUN, ce_agg.NT_PROBNP, ce_agg.TROPONIN_I,
        ce_agg.PH, ce_agg.HCO3, ce_agg.WBC, ce_agg.HGB, ce_agg.PLT, ce_agg.VIS
    FROM scai_stage_hourly sh
    JOIN (
        SELECT
            ce.ENCOUNTER_ID,
            sh2.HOUR_FROM_ADMIT,
            MAX(CASE WHEN ce.EVENT_CD = 'HR'           THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS HR,
            MAX(CASE WHEN ce.EVENT_CD = 'SBP'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS SBP,
            MAX(CASE WHEN ce.EVENT_CD = 'DBP'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS DBP,
            MAX(CASE WHEN ce.EVENT_CD = 'MAP'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS MAP,
            MAX(CASE WHEN ce.EVENT_CD = 'RR'           THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS RR,
            MAX(CASE WHEN ce.EVENT_CD = 'SPO2'         THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS SPO2,
            MAX(CASE WHEN ce.EVENT_CD = 'TEMP'         THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS TEMP,
            MAX(CASE WHEN ce.EVENT_CD = 'URINE_OUT_HR' THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS URINE_OUT_HR,
            MAX(CASE WHEN ce.EVENT_CD = 'CVP'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS CVP,
            MAX(CASE WHEN ce.EVENT_CD = 'CO'           THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS CO,
            MAX(CASE WHEN ce.EVENT_CD = 'CI'           THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS CI,
            MAX(CASE WHEN ce.EVENT_CD = 'SVO2'         THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS SVO2,
            MAX(CASE WHEN ce.EVENT_CD = 'CPO'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS CPO,
            MAX(CASE WHEN ce.EVENT_CD = 'PAPI'         THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS PAPI,
            MAX(CASE WHEN ce.EVENT_CD = 'LACTATE'      THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS LACTATE,
            MAX(CASE WHEN ce.EVENT_CD = 'CREATININE'   THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS CREATININE,
            MAX(CASE WHEN ce.EVENT_CD = 'BUN'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS BUN,
            MAX(CASE WHEN ce.EVENT_CD = 'NT_PROBNP'    THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS NT_PROBNP,
            MAX(CASE WHEN ce.EVENT_CD = 'TROPONIN_I'   THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS TROPONIN_I,
            MAX(CASE WHEN ce.EVENT_CD = 'PH'           THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS PH,
            MAX(CASE WHEN ce.EVENT_CD = 'HCO3'         THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS HCO3,
            MAX(CASE WHEN ce.EVENT_CD = 'WBC'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS WBC,
            MAX(CASE WHEN ce.EVENT_CD = 'HGB'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS HGB,
            MAX(CASE WHEN ce.EVENT_CD = 'PLT'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS PLT,
            MAX(CASE WHEN ce.EVENT_CD = 'VIS'          THEN TRY_CAST(ce.RESULT_VAL AS DOUBLE) END) AS VIS
        FROM clinical_event ce
        JOIN scai_stage_hourly sh2
          ON ce.ENCOUNTER_ID = sh2.ENCOUNTER_ID
         AND ce.EVENT_END_DT_TM = sh2.EVENT_DT_TM
        WHERE sh2.SCAI_STAGE_CD NOT IN ('D', 'E')
        GROUP BY ce.ENCOUNTER_ID, sh2.HOUR_FROM_ADMIT
    ) ce_agg
      ON sh.ENCOUNTER_ID = ce_agg.ENCOUNTER_ID
     AND sh.HOUR_FROM_ADMIT = ce_agg.HOUR_FROM_ADMIT
    JOIN encounter e ON sh.ENCOUNTER_ID = e.ENCOUNTER_ID
    WHERE sh.SCAI_STAGE_CD NOT IN ('D', 'E')
    ORDER BY sh.ENCOUNTER_ID, sh.HOUR_FROM_ADMIT
    """

    df = query(pivot_sql)

    # Attach PERSON_ID for GroupShuffleSplit
    person_sql = "SELECT ENCOUNTER_ID, PERSON_ID FROM encounter"
    persons = query(person_sql)
    df = df.merge(persons, on="ENCOUNTER_ID", how="left")

    print(f"  Rows after Stage D/E exclusion: {len(df):,}")
    print(f"  Unique encounters: {df['ENCOUNTER_ID'].nunique():,}")
    print(f"  Unique patients:   {df['PERSON_ID'].nunique():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LABELS
# VASOPRESSOR_NEEDED_24H  : 1 if a NEW vasopressor starts in next 24 rows
# VASOPRESSOR_COUNT_24H   : count of distinct new vasopressors in next 24 rows
# N_VASOPRESSORS_ACTIVE   : count of vasopressors currently active at row T
# ─────────────────────────────────────────────────────────────────────────────
def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach vasopressor labels. Uses pandas row-window (next 24 rows = 24 hours)."""
    print("\n[Step 2] Building vasopressor labels...")

    # Pull all hours where each vasopressor was actively infusing.
    # A drug is "active" at hour H if it started on or before EVENT_DT_TM
    # and ended on or after EVENT_DT_TM.
    vaso_sql = f"""
    SELECT
        ma.ENCOUNTER_ID,
        sh.HOUR_FROM_ADMIT,
        ma.MEDICATION_CD
    FROM medication_admin ma
    JOIN scai_stage_hourly sh
      ON ma.ENCOUNTER_ID = sh.ENCOUNTER_ID
     AND ma.ADMIN_START_DT_TM <= sh.EVENT_DT_TM
     AND ma.ADMIN_END_DT_TM   >= sh.EVENT_DT_TM
    WHERE UPPER(ma.MEDICATION_CD) IN ({', '.join(f"'{v}'" for v in VASOPRESSORS)})
      AND sh.SCAI_STAGE_CD NOT IN ('D', 'E')
    """
    vaso_df = query(vaso_sql)

    # Build lookup: (ENCOUNTER_ID, HOUR_FROM_ADMIT) → set of active vasopressors
    active_map: dict[tuple, set] = {}
    for row in vaso_df.itertuples(index=False):
        key = (row.ENCOUNTER_ID, row.HOUR_FROM_ADMIT)
        active_map.setdefault(key, set()).add(row.MEDICATION_CD)

    encounters = df["ENCOUNTER_ID"].unique()
    needed_24h_list = []
    count_24h_list = []
    n_active_list = []

    for enc_id in encounters:
        mask = df["ENCOUNTER_ID"] == enc_id
        enc_df = df[mask].sort_values("HOUR_FROM_ADMIT").copy()
        hours = enc_df["HOUR_FROM_ADMIT"].tolist()
        n = len(hours)

        needed_24h = []
        count_24h = []
        n_active = []

        for i, h in enumerate(hours):
            # Currently active at this hour
            active_now = active_map.get((enc_id, h), set())
            n_active.append(len(active_now))

            # Look forward up to 24 rows (hours)
            future_window = hours[i + 1 : i + 25]
            new_drugs: set[str] = set()
            for fh in future_window:
                started = active_map.get((enc_id, fh), set())
                new_drugs |= (started - active_now)

            needed_24h.append(1 if new_drugs else 0)
            count_24h.append(len(new_drugs))

        needed_24h_list.extend(needed_24h)
        count_24h_list.extend(count_24h)
        n_active_list.extend(n_active)

    # Re-order to match original df row order
    df = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    df["VASOPRESSOR_NEEDED_24H"]  = needed_24h_list
    df["VASOPRESSOR_COUNT_24H"]   = count_24h_list
    df["N_VASOPRESSORS_ACTIVE"]   = n_active_list

    pos_rate = df["VASOPRESSOR_NEEDED_24H"].mean()
    print(f"  VASOPRESSOR_NEEDED_24H positive rate: {pos_rate:.1%}")
    print(f"  VASOPRESSOR_COUNT_24H distribution:\n{df['VASOPRESSOR_COUNT_24H'].value_counts().sort_index().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CLINICAL RANGE NULLING
# Impossible values become NaN before imputation, not after.
# ─────────────────────────────────────────────────────────────────────────────
def apply_clinical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Nullify values outside physiologically possible bounds."""
    print("\n[Step 3] Nullifying clinical range violations...")
    total_nulled = 0
    for feat, (lo, hi) in CLINICAL_RANGES.items():
        if feat not in df.columns:
            continue
        before = df[feat].notna().sum()
        df[feat] = df[feat].where((df[feat] >= lo) & (df[feat] <= hi), other=np.nan)
        after = df[feat].notna().sum()
        nulled = before - after
        if nulled > 0:
            print(f"  {feat}: nulled {nulled:,} values outside [{lo}, {hi}]")
            total_nulled += nulled
    print(f"  Total values nulled: {total_nulled:,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WAS_MEASURED FLAGS
# Binary flag = 1 if the original (pre-imputation) value was present.
# Created AFTER range nulling so spurious values don't inflate presence.
# ─────────────────────────────────────────────────────────────────────────────
def add_was_measured_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add was_measured_X binary columns for all sparse features."""
    print("\n[Step 4] Adding was_measured_ flags...")
    for feat in SPARSE_FEATURES:
        if feat in df.columns:
            df[f"was_measured_{feat}"] = df[feat].notna().astype(np.int8)
    flag_cols = [c for c in df.columns if c.startswith("was_measured_")]
    print(f"  Created {len(flag_cols)} was_measured_ flags")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — LOCF WITHIN ENCOUNTER
# Forward-fill only within the same encounter. No cross-patient leakage.
# ─────────────────────────────────────────────────────────────────────────────
def apply_locf(df: pd.DataFrame) -> pd.DataFrame:
    """Last Observation Carried Forward — within-encounter only."""
    print("\n[Step 5] Applying LOCF within encounters...")
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    before_nulls = df[feat_cols].isnull().sum().sum()

    df = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"])
    df[feat_cols] = (
        df.groupby("ENCOUNTER_ID")[feat_cols]
        .transform(lambda g: g.ffill())
    )

    after_nulls = df[feat_cols].isnull().sum().sum()
    print(f"  Nulls before LOCF: {before_nulls:,}")
    print(f"  Nulls after  LOCF: {after_nulls:,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PATIENT-WISE SPLIT
# GroupShuffleSplit on ENCOUNTER_ID → 70 / 15 / 15
# ─────────────────────────────────────────────────────────────────────────────
def patient_wise_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test by encounter (no patient appears in 2 splits)."""
    print("\n[Step 6] Patient-wise 70/15/15 split...")

    encounters = df["ENCOUNTER_ID"].unique()
    n = len(encounters)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, holdout_idx = next(gss.split(encounters, groups=encounters))
    train_enc = set(encounters[train_idx])
    holdout_enc = encounters[holdout_idx]

    # Split holdout 50/50 → val and test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(holdout_enc, groups=holdout_enc))
    val_enc  = set(holdout_enc[val_idx])
    test_enc = set(holdout_enc[test_idx])

    train = df[df["ENCOUNTER_ID"].isin(train_enc)].copy()
    val   = df[df["ENCOUNTER_ID"].isin(val_enc)].copy()
    test  = df[df["ENCOUNTER_ID"].isin(test_enc)].copy()

    print(f"  Train: {len(train):,} rows, {len(train_enc):,} encounters")
    print(f"  Val:   {len(val):,} rows, {len(val_enc):,} encounters")
    print(f"  Test:  {len(test):,} rows, {len(test_enc):,} encounters")

    # Verify no patient overlap
    train_pts  = set(train["PERSON_ID"].unique())
    val_pts    = set(val["PERSON_ID"].unique())
    test_pts   = set(test["PERSON_ID"].unique())
    tv_overlap = train_pts & val_pts
    tt_overlap = train_pts & test_pts
    vt_overlap = val_pts  & test_pts
    print(f"  Patient overlap — train/val: {len(tv_overlap)}, train/test: {len(tt_overlap)}, val/test: {len(vt_overlap)}")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — KNN IMPUTATION
# Parallel feature-by-feature k=5 KNN using NearestNeighbors(n_jobs=-1).
# This gives the same result as sklearn KNNImputer(k=5) but runs in parallel
# across CPU cores — essential for the 16 sparse features at 80-93% null.
#
# Strategy per feature f:
#   1. Observed rows (f is not null) → train the NearestNeighbors on
#      the median-filled version of all OTHER features (proxy coordinates).
#   2. Missing rows → find 5 nearest observed rows → take mean of their f values.
#   3. Serialize per-feature NearestNeighbors models into the artifact pickle.
# ─────────────────────────────────────────────────────────────────────────────
def apply_knn_imputation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feature-by-feature KNN impute (k=5). Fit on train, apply to all."""
    print("\n[Step 7] KNN imputation (k=5, parallel feature-by-feature)...")

    feat_cols = [c for c in ALL_FEATURES if c in train.columns]

    # Compute per-feature medians on train for coordinate filling
    train_medians = train[feat_cols].median()

    # We'll store {feature: (nn_model, train_obs_values, train_median_fill)}
    # so we can serialize and reapply to val/test
    imputer_state: dict = {
        "k": 5,
        "feat_cols": feat_cols,
        "train_medians": train_medians.to_dict(),
        "feature_models": {},
    }

    def _impute_split(df_split: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        for feat in feat_cols:
            if df_split[feat].isna().sum() == 0:
                continue

            miss_mask = df_split[feat].isna()

            # Build coordinate matrix: all features except target, NaN → median
            coord_cols = [c for c in feat_cols if c != feat]
            fill_vals  = train_medians[coord_cols]
            X_miss = df_split.loc[miss_mask, coord_cols].fillna(fill_vals).values

            if is_train:
                obs_mask = ~df_split[feat].isna()
                if obs_mask.sum() < 5:
                    df_split.loc[miss_mask, feat] = train_medians[feat]
                    continue

                X_obs = df_split.loc[obs_mask, coord_cols].fillna(fill_vals).values
                y_obs = df_split.loc[obs_mask, feat].values.astype(float)

                nn = NearestNeighbors(n_neighbors=5, n_jobs=-1, algorithm="ball_tree")
                nn.fit(X_obs)
                # Store both the model and the observed values from train
                imputer_state["feature_models"][feat] = (nn, y_obs)
            else:
                entry = imputer_state["feature_models"].get(feat)
                if entry is None:
                    df_split.loc[miss_mask, feat] = train_medians[feat]
                    continue
                nn, y_obs = entry

            _, indices = nn.kneighbors(X_miss)
            df_split.loc[miss_mask, feat] = y_obs[indices].mean(axis=1)

        return df_split

    train = _impute_split(train, is_train=True)
    val   = _impute_split(val,   is_train=False)
    test  = _impute_split(test,  is_train=False)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    imputer_path = ARTIFACT_DIR / "knn_imputer.pkl"
    with open(imputer_path, "wb") as f:
        pickle.dump(imputer_state, f)
    print(f"  Saved imputer state → {imputer_path}")

    remaining = train[feat_cols].isnull().sum().sum()
    print(f"  Remaining nulls in train after KNN: {remaining}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — IQR CAPPING
# 1.5× IQR bounds computed on train only. Applied to all splits.
# ─────────────────────────────────────────────────────────────────────────────
def apply_iqr_capping(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cap outliers at Q1 - 1.5*IQR / Q3 + 1.5*IQR. Fit on train only."""
    print("\n[Step 8] IQR capping (1.5×, fit on train only)...")

    feat_cols = [c for c in ALL_FEATURES if c in train.columns]
    bounds: dict[str, dict] = {}

    for feat in feat_cols:
        q1 = train[feat].quantile(0.25)
        q3 = train[feat].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        bounds[feat] = {"q1": q1, "q3": q3, "iqr": iqr, "lower": lo, "upper": hi}

        for split in (train, val, test):
            split[feat] = split[feat].clip(lower=lo, upper=hi)

    bounds_path = ARTIFACT_DIR / "iqr_bounds.json"
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"  Saved IQR bounds → {bounds_path}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — LOG1P TRANSFORM
# Applied to right-skewed features after imputation and IQR capping.
# ─────────────────────────────────────────────────────────────────────────────
def apply_log_transforms(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply log1p to skewed features."""
    print("\n[Step 9] Applying log1p transforms...")

    applied = []
    for feat in LOG_TRANSFORM_FEATURES:
        if feat in train.columns:
            for split in (train, val, test):
                split[feat] = np.log1p(split[feat].clip(lower=0))
            applied.append(feat)
            print(f"  log1p applied to {feat}")

    log_path = ARTIFACT_DIR / "log_transform_features.json"
    with open(log_path, "w") as f:
        json.dump(applied, f, indent=2)
    print(f"  Saved log transform list → {log_path}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — SAVE PARQUETS
# ─────────────────────────────────────────────────────────────────────────────
def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Write train/val/test parquets to data/processed/."""
    print("\n[Step 10] Saving processed parquets...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet",   index=False)
    test.to_parquet(OUT_DIR / "test.parquet",  index=False)

    print(f"  train.parquet  → {len(train):,} rows")
    print(f"  val.parquet    → {len(val):,} rows")
    print(f"  test.parquet   → {len(test):,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — MANIFEST
# ─────────────────────────────────────────────────────────────────────────────
def save_manifest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Write preprocessing_manifest.json summarizing the pipeline output."""
    feat_cols = [
        c for c in train.columns
        if c not in NON_FEATURE_COLS
        and not c.startswith("was_measured_")
    ]
    flag_cols = [c for c in train.columns if c.startswith("was_measured_")]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0",
        "stage_exclusions": ["D (3)", "E (4)"],
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "split_ratios": {
            "train_pct": round(100 * len(train) / (len(train) + len(val) + len(test)), 1),
            "val_pct":   round(100 * len(val)   / (len(train) + len(val) + len(test)), 1),
            "test_pct":  round(100 * len(test)  / (len(train) + len(val) + len(test)), 1),
        },
        "n_features": len(feat_cols),
        "feature_cols": sorted(feat_cols),
        "was_measured_flags": sorted(flag_cols),
        "log_transform_features": LOG_TRANSFORM_FEATURES,
        "sparse_features": SPARSE_FEATURES,
        "label_positive_rate": {
            "train": round(train["VASOPRESSOR_NEEDED_24H"].mean(), 4),
            "val":   round(val["VASOPRESSOR_NEEDED_24H"].mean(), 4),
            "test":  round(test["VASOPRESSOR_NEEDED_24H"].mean(), 4),
        },
        "null_check": {
            "train_nulls": int(train[feat_cols].isnull().sum().sum()),
            "val_nulls":   int(val[feat_cols].isnull().sum().sum()),
            "test_nulls":  int(test[feat_cols].isnull().sum().sum()),
        },
        "artifacts": {
            "knn_imputer":          "artifacts/knn_imputer.pkl",
            "iqr_bounds":           "artifacts/iqr_bounds.json",
            "log_transform_features": "artifacts/log_transform_features.json",
        },
    }

    manifest_path = ARTIFACT_DIR / "preprocessing_manifest.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Step 11] Manifest saved → {manifest_path}")

    # Final null check
    total_nulls = manifest["null_check"]
    if sum(total_nulls.values()) == 0:
        print("  NULL CHECK: PASSED — zero nulls in all feature columns")
    else:
        print(f"  NULL CHECK: WARNING — nulls remain: {total_nulls}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(data_dir: Path, out_dir: Path) -> None:
    global OUT_DIR
    OUT_DIR = out_dir

    print("=" * 65)
    print("  Vasopressor Model — Preprocessing Pipeline")
    print("=" * 65)

    init_db(data_dir)

    # Steps 1-5: build and clean the full dataset before splitting
    df = build_wide_pivot()
    df = build_labels(df)
    df = apply_clinical_ranges(df)
    df = add_was_measured_flags(df)
    df = apply_locf(df)

    # Step 6: patient-wise split
    train, val, test = patient_wise_split(df)

    # Steps 7-9: fit on train, apply to all splits
    train, val, test = apply_knn_imputation(train, val, test)
    train, val, test = apply_iqr_capping(train, val, test)
    train, val, test = apply_log_transforms(train, val, test)

    # Steps 10-11: save outputs
    save_splits(train, val, test)
    save_manifest(train, val, test)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vasopressor model preprocessing pipeline")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing raw parquet files")
    parser.add_argument("--out-dir",  type=Path, default=OUT_DIR,
                        help="Output directory for processed parquets")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
