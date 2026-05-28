from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from model_development import (
    HEMO_CODES,
    LOOKBACK_HOURS,
    MIN_HOUR,
    NUMERIC_CODES,
    STATIC_CAT,
    STATIC_NUMERIC,
    TIME_FEATURES,
    build_hourly_wide,
    build_treatment_state,
    build_window_features,
    dx_category,
    load_tables,
    patient_split,
)

MCS_HORIZON_HOURS = 12
MCS_LABEL_COL = "Y_MCS_ESCALATION"
RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.5
# Fixed operating point when missing escalation is the dominant risk (see MODEL_CARD).
CLINICAL_ALERT_THRESHOLD = 0.10

SENSITIVITY_TRAIN_KWARGS: dict[str, Any] = {
    "ensemble_seeds": 5,
    "oof_splits": 5,
    "tighten_regularization": True,
    "oof_calibration": True,
    "calibration_method": "sigmoid",
    "eval_metric": "aucpr",
    "tune_threshold": False,
    "scale_pos_weight_multiplier": 1.35,
    "feature_select": True,
}
MEDIUM_LOOKBACK_HOURS = 6
MEDIUM_LOOKBACK_SUFFIX = f"lb{MEDIUM_LOOKBACK_HOURS}h"
LONG_LOOKBACK_HOURS = 12
LONG_LOOKBACK_SUFFIX = f"lb{LONG_LOOKBACK_HOURS}h"
PRE_MCS_CONTEXT_HOURS = 48
EARLY_STOPPING_ROUNDS = 40
TIME_FEATURES_MCS = [c for c in TIME_FEATURES if c != "ON_MCS"]
TREATMENT_TREND_FEATURES = [
    "N_VASOPRESSORS_ACTIVE_delta1h",
    "N_INOTROPES_ACTIVE_delta1h",
]
STAGE_TRAJECTORY_FEATURES = ["HOURS_ON_CURRENT_STAGE"]
ESCALATION_PROXIMITY_FEATURES = [
    "STAGE_WORSENED_6H",
    "PRESSOR_RISING_6H",
    "VIS_SPIKE_4H",
    "LACTATE_SPIKE_4H",
    "HIGH_SHOCK_BURDEN",
]

# Derived hemodynamic indices computed from per-hour code values.
# (name, numerator code, denominator code, callable)
SHOCK_INDEX_DEFS: list[tuple[str, str, str, Any]] = [
    ("SHOCK_INDEX", "HR", "SBP", lambda a, b: a / b),
    ("LACTATE_X_HR", "LACTATE", "HR", lambda a, b: a * b / 100.0),
    ("VIS_PER_MAP", "VIS", "MAP", lambda a, b: a / b),
    ("MAP_HR_RATIO", "MAP", "HR", lambda a, b: a / b),
]

# Hours-since-last-abnormal-value features.
# (code, direction, threshold) where direction is "high" or "low".
ABNORMAL_RULES: list[tuple[str, str, float]] = [
    ("LACTATE", "high", 4.0),
    ("MAP", "low", 65.0),
    ("SBP", "low", 90.0),
    ("SPO2", "low", 90.0),
    ("HR", "high", 120.0),
]

CUMULATIVE_ESCALATION_WINDOWS = (6, 12)
STAGE_LAG_HOURS = (6, 12)

CODES_FOR_SHOCK_INDICES = sorted({code for _, a, b, _ in SHOCK_INDEX_DEFS for code in (a, b)})

DASHBOARD_NUMERIC_CODES = [
    "HR",
    "SBP",
    "DBP",
    "MAP",
    "SPO2",
    "URINE_OUT_HR",
    "LACTATE",
    "PH",
    "HCO3",
    "BUN",
    "CREATININE",
    "AST",
    "WBC",
    "CI",
    "PCWP",
    "VIS",
]

DASHBOARD_TIME_FIELDS = [
    "HOUR_FROM_ADMIT",
    "CURRENT_STAGE_NUM",
    "N_VASOPRESSORS_ACTIVE",
    "N_INOTROPES_ACTIVE",
    "ON_INTUBATION",
    "ON_CRRT",
]

DASHBOARD_STATIC_NUMERIC = ["AGE", "WEIGHT_KG", "BSA_M2"]


def mcs_escalation_encounter_ids(proc: pd.DataFrame) -> np.ndarray:
    return proc.loc[proc.PROC_CATEGORY_CD == "MCS", "ENCOUNTER_ID"].dropna().unique()


def filter_tables_to_encounters(
    tbls: dict[str, pd.DataFrame],
    encounter_ids: set | frozenset,
) -> dict[str, pd.DataFrame]:
    """Subset parquet tables to a cohort before frame build (reduces memory)."""
    out = dict(tbls)
    for key in ("scai_stage_hourly", "clinical_event", "medication_admin", "procedure_event"):
        if key in out:
            out[key] = out[key][out[key]["ENCOUNTER_ID"].isin(encounter_ids)].copy()
    out["diagnosis"] = out["diagnosis"][out["diagnosis"]["ENCOUNTER_ID"].isin(encounter_ids)].copy()
    out["encounter"] = out["encounter"][out["encounter"]["ENCOUNTER_ID"].isin(encounter_ids)].copy()
    return out


def build_hourly_on_mcs(scai: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    hourly = scai.merge(tx, on=["ENCOUNTER_ID", "HOUR_FROM_ADMIT"], how="left")
    hourly["ON_MCS"] = hourly["ON_MCS"].fillna(0).astype(int)
    return hourly


def mcs_start_hours(hourly_on_mcs: pd.DataFrame) -> pd.DataFrame:
    return (
        hourly_on_mcs.loc[hourly_on_mcs["ON_MCS"] == 1, ["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]]
        .groupby("ENCOUNTER_ID", as_index=False)
        .agg(MCS_START_HOUR=("HOUR_FROM_ADMIT", "min"))
    )


def prefix_feature_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    key = ["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]
    feature_cols = [c for c in df.columns if c not in key]
    return df[key].join(df[feature_cols].rename(columns=lambda c: f"{c}_{suffix}"))


def add_hours_on_current_stage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    stage_block = out.groupby("ENCOUNTER_ID")["CURRENT_STAGE_NUM"].transform(
        lambda s: (s != s.shift()).cumsum()
    )
    out["HOURS_ON_CURRENT_STAGE"] = out.groupby(["ENCOUNTER_ID", stage_block]).cumcount()
    return out


def add_hemodynamic_missingness(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for code in HEMO_CODES:
        now_col = f"{code}_now"
        if now_col in out.columns:
            out[f"{code}_missing"] = out[now_col].isna().astype(int)
    return out


def add_treatment_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    for col in ["N_VASOPRESSORS_ACTIVE", "N_INOTROPES_ACTIVE"]:
        out[f"{col}_delta1h"] = (
            out.groupby("ENCOUNTER_ID")[col].diff().fillna(0.0)
        )
    return out


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _compute_shock_index(name: str, a: float, b: float) -> float:
    """Compute one shock-index value from a single (a, b) pair (used by overrides)."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    if name == "SHOCK_INDEX":
        return a / b if b else np.nan
    if name == "LACTATE_X_HR":
        return a * b / 100.0
    if name == "VIS_PER_MAP":
        return a / b if b else np.nan
    if name == "MAP_HR_RATIO":
        return a / b if b else np.nan
    return np.nan


def shock_index_feature_names() -> list[str]:
    names: list[str] = []
    for name, _, _, _ in SHOCK_INDEX_DEFS:
        names.append(f"{name}_now")
        names.append(f"{name}_mean{LOOKBACK_HOURS}h")
        names.append(f"{name}_max{LOOKBACK_HOURS}h")
    return names


def add_shock_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Hemodynamic index features (HR/SBP, LACTATE*HR/100, VIS/MAP, MAP/HR)."""
    out = df.copy()
    for name, a_code, b_code, fn in SHOCK_INDEX_DEFS:
        a_now = f"{a_code}_now"
        b_now = f"{b_code}_now"
        if a_now not in out.columns or b_now not in out.columns:
            continue
        out[f"{name}_now"] = fn(out[a_now], out[b_now].replace(0, np.nan))
        a_mean = f"{a_code}_mean{LOOKBACK_HOURS}h"
        b_mean = f"{b_code}_mean{LOOKBACK_HOURS}h"
        if a_mean in out.columns and b_mean in out.columns:
            out[f"{name}_mean{LOOKBACK_HOURS}h"] = fn(
                out[a_mean], out[b_mean].replace(0, np.nan)
            )
        a_max = f"{a_code}_max{LOOKBACK_HOURS}h"
        b_min = f"{b_code}_min{LOOKBACK_HOURS}h"
        if a_max in out.columns and b_min in out.columns:
            out[f"{name}_max{LOOKBACK_HOURS}h"] = fn(
                out[a_max], out[b_min].replace(0, np.nan)
            )
    return out


def time_since_abnormal_feature_names() -> list[str]:
    return [f"HOURS_SINCE_{code}_{direction.upper()}" for code, direction, _ in ABNORMAL_RULES]


def add_time_since_abnormal(df: pd.DataFrame) -> pd.DataFrame:
    """Hours since the last time each vital crossed its abnormal threshold within an encounter."""
    out = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    for code, direction, threshold in ABNORMAL_RULES:
        col = f"{code}_now"
        feature = f"HOURS_SINCE_{code}_{direction.upper()}"
        if col not in out.columns:
            out[feature] = -1.0
            continue
        if direction == "high":
            crossed = out[col] > threshold
        else:
            crossed = out[col] < threshold
        marker = np.where(crossed.fillna(False), out["HOUR_FROM_ADMIT"], np.nan)
        marker_series = pd.Series(marker, index=out.index)
        last_marker = marker_series.groupby(out["ENCOUNTER_ID"]).cummax()
        hours_since = out["HOUR_FROM_ADMIT"] - last_marker
        out[feature] = hours_since.fillna(-1.0).astype(float)
    return out


def cumulative_escalation_feature_names() -> list[str]:
    names: list[str] = []
    for col in ("N_VASOPRESSORS_ACTIVE", "N_INOTROPES_ACTIVE"):
        for window in CUMULATIVE_ESCALATION_WINDOWS:
            names.append(f"{col}_ADDITIONS_{window}H")
    return names


def add_cumulative_escalations(df: pd.DataFrame) -> pd.DataFrame:
    """Count of vasopressor / inotrope additions in rolling 6h and 12h windows."""
    out = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    for col in ("N_VASOPRESSORS_ACTIVE", "N_INOTROPES_ACTIVE"):
        if col not in out.columns:
            continue
        delta = out.groupby("ENCOUNTER_ID")[col].diff().fillna(0.0)
        addition = (delta > 0).astype(int)
        out["_addition"] = addition
        for window in CUMULATIVE_ESCALATION_WINDOWS:
            out[f"{col}_ADDITIONS_{window}H"] = (
                out.groupby("ENCOUNTER_ID")["_addition"]
                .transform(lambda s, w=window: s.rolling(window=w, min_periods=1).sum())
                .astype(float)
            )
        out = out.drop(columns="_addition")
    return out


def stage_trajectory_feature_names() -> list[str]:
    names = []
    for lag in STAGE_LAG_HOURS:
        names.append(f"STAGE_LAG{lag}H")
        names.append(f"STAGE_DELTA{lag}H")
    return names


# Shock-burden composite mirrors the simulator's MCS trigger score so the
# model gets the same summary statistic the trigger logic actually uses.
# Each component is clipped exactly as in the simulator's score()
# computation, giving the model one feature whose physical meaning is
# "how close is this patient to the trigger threshold right now".
_SHOCK_BURDEN_COMPONENTS = [
    # (code, transform, max_contribution, threshold, scale, direction)
    # direction: "high" means (val - threshold) / scale; "low" means (threshold - val) / scale
    ("LACTATE",  2.0,   2.0,  2.0,   "high"),
    ("MAP",      2.0,  65.0, 10.0,   "low"),
    ("VIS",      2.0,  15.0, 15.0,   "high"),
    ("HR",       2.0, 100.0, 25.0,   "high"),
    ("SPO2",     1.5,  92.0, 12.0,   "low"),
    ("CI",       2.0,   2.2,  0.6,   "low"),
    ("PCWP",     1.5,  18.0, 10.0,   "high"),
]
_SHOCK_BURDEN_SUFFIXES = ("now", "mean4h", "mean6h_lb6h", "mean12h_lb12h")
_SHOCK_BURDEN_NAMES = tuple(f"SHOCK_BURDEN_{s}" for s in _SHOCK_BURDEN_SUFFIXES)


def shock_burden_feature_names() -> list[str]:
    return list(_SHOCK_BURDEN_NAMES) + ["SHOCK_BURDEN_INT_12h"]


def add_shock_burden_features(df: pd.DataFrame) -> pd.DataFrame:
    """Composite shock-burden score, mirroring the simulator's trigger
    logic at three time scales plus its 12h rolling integral.

    For each lookback (now / 4h mean / 6h mean / 12h mean) we sum the
    clipped contributions of lactate, MAP, VIS, HR, SpO2, CI, PCWP
    plus stage severity. CI and PCWP are optional (PA catheter
    presence) and skipped when the column is missing.

    XGB can in principle re-derive this composite from raw features
    but giving it the exact monotone combination cuts the number of
    splits needed to recover the trigger boundary, which empirically
    helps on small cohorts.
    """
    out = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    stage_col = "CURRENT_STAGE_NUM"

    def _component(col_name: str, max_c: float, thr: float, scale: float,
                   direction: str) -> pd.Series:
        if col_name not in out.columns:
            return pd.Series(0.0, index=out.index)
        s = out[col_name].astype(float)
        if direction == "high":
            raw = (s - thr) / scale
        else:
            raw = (thr - s) / scale
        return raw.clip(lower=0.0, upper=max_c).fillna(0.0)

    suffix_to_window = {
        "now": "now",
        "mean4h": f"mean{LOOKBACK_HOURS}h",
        "mean6h_lb6h": f"mean{MEDIUM_LOOKBACK_HOURS}h_{MEDIUM_LOOKBACK_SUFFIX}",
        "mean12h_lb12h": f"mean{LONG_LOOKBACK_HOURS}h_{LONG_LOOKBACK_SUFFIX}",
    }

    for suffix, window_token in suffix_to_window.items():
        total = pd.Series(0.0, index=out.index)
        for code, max_c, thr, scale, direction in _SHOCK_BURDEN_COMPONENTS:
            col = f"{code}_{window_token}"
            total = total + _component(col, max_c, thr, scale, direction)
        if stage_col in out.columns:
            total = total + (out[stage_col].astype(float).clip(lower=1.0) - 1.0).fillna(0.0)
        out[f"SHOCK_BURDEN_{suffix}"] = total

    # 12-hour rolling integral of the current-hour burden — captures
    # how long the patient has been at high score, the same way the
    # simulator's streak counter does.
    burden_now = out["SHOCK_BURDEN_now"]
    out["SHOCK_BURDEN_INT_12h"] = (
        burden_now.groupby(out["ENCOUNTER_ID"])
        .transform(lambda s: s.rolling(window=12, min_periods=1).sum())
        .astype(float)
    )
    return out


def add_escalation_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Short-horizon deterioration flags aimed at pre-escalation hours."""
    out = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    if "STAGE_LAG6H" in out.columns:
        out["STAGE_WORSENED_6H"] = (
            out["CURRENT_STAGE_NUM"].astype(float) > out["STAGE_LAG6H"].astype(float)
        ).fillna(0).astype(int)
    if "N_VASOPRESSORS_ACTIVE" in out.columns:
        prior = out.groupby("ENCOUNTER_ID")["N_VASOPRESSORS_ACTIVE"].shift(6)
        out["PRESSOR_RISING_6H"] = (
            out["N_VASOPRESSORS_ACTIVE"].astype(float) > prior.astype(float)
        ).fillna(0).astype(int)
    if "VIS_now" in out.columns and "VIS_mean4h" in out.columns:
        out["VIS_SPIKE_4H"] = (
            out["VIS_now"].astype(float) > out["VIS_mean4h"].astype(float) * 1.25
        ).fillna(0).astype(int)
    if "LACTATE_now" in out.columns and "LACTATE_mean4h" in out.columns:
        out["LACTATE_SPIKE_4H"] = (
            out["LACTATE_now"].astype(float) > out["LACTATE_mean4h"].astype(float) + 1.0
        ).fillna(0).astype(int)
    if "SHOCK_BURDEN_now" in out.columns:
        out["HIGH_SHOCK_BURDEN"] = (out["SHOCK_BURDEN_now"].astype(float) >= 4.0).astype(int)
    return out


def add_stage_trajectory_lags(df: pd.DataFrame) -> pd.DataFrame:
    """SCAI stage values from `lag` hours ago plus the change since then."""
    out = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    for lag in STAGE_LAG_HOURS:
        lagged = out.groupby("ENCOUNTER_ID")["CURRENT_STAGE_NUM"].shift(lag)
        out[f"STAGE_LAG{lag}H"] = lagged
        out[f"STAGE_DELTA{lag}H"] = (out["CURRENT_STAGE_NUM"] - lagged).astype(float)
    return out


def mcs_window_feature_names(lookback: int, *, suffix: str = "") -> list[str]:
    suffix_token = f"_{suffix}" if suffix else ""
    names: list[str] = []
    for code in NUMERIC_CODES:
        names.append(f"{code}_now{suffix_token}")
        for part in ("mean", "min", "max", "delta", "slope"):
            names.append(f"{code}_{part}{lookback}h{suffix_token}")
    return names


def mcs_feature_schema(
    df: pd.DataFrame,
    *,
    extra_lookbacks: tuple[tuple[int, str], ...] = (),
) -> tuple[list[str], list[str]]:
    """Build the feature schema.

    ``extra_lookbacks`` is a tuple of ``(lookback_hours, suffix)`` pairs
    that downstream modules (e.g. the ECMO predictor with its 24h
    window) can use to register additional rolling-window feature
    families without forking this function.
    """
    num_feats = (
        mcs_window_feature_names(LOOKBACK_HOURS)
        + mcs_window_feature_names(MEDIUM_LOOKBACK_HOURS, suffix=MEDIUM_LOOKBACK_SUFFIX)
        + mcs_window_feature_names(LONG_LOOKBACK_HOURS, suffix=LONG_LOOKBACK_SUFFIX)
    )
    for lookback, suffix in extra_lookbacks:
        num_feats = num_feats + mcs_window_feature_names(lookback, suffix=suffix)
    num_feats = (
        num_feats
        + STATIC_NUMERIC
        + TIME_FEATURES_MCS
        + STAGE_TRAJECTORY_FEATURES
        + TREATMENT_TREND_FEATURES
        + [f"{code}_missing" for code in HEMO_CODES]
        + shock_index_feature_names()
        + time_since_abnormal_feature_names()
        + cumulative_escalation_feature_names()
        + stage_trajectory_feature_names()
        + shock_burden_feature_names()
        + ESCALATION_PROXIMITY_FEATURES
    )
    cat_feats = STATIC_CAT
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = [c for c in cat_feats if c in df.columns]
    return num_feats, cat_feats


def build_mcs_prediction_frame(
    tbls: dict[str, pd.DataFrame],
    *,
    inference_mode: bool = False,
) -> pd.DataFrame:
    wide = build_hourly_wide(tbls["clinical_event"], tbls["scai_stage_hourly"])
    feats_4h = build_window_features(wide, lookback=LOOKBACK_HOURS)
    feats_6h = prefix_feature_columns(
        build_window_features(wide, lookback=MEDIUM_LOOKBACK_HOURS),
        MEDIUM_LOOKBACK_SUFFIX,
    )
    feats_12h = prefix_feature_columns(
        build_window_features(wide, lookback=LONG_LOOKBACK_HOURS),
        LONG_LOOKBACK_SUFFIX,
    )
    tx = build_treatment_state(
        tbls["encounter"],
        tbls["medication_admin"],
        tbls["procedure_event"],
        tbls["scai_stage_hourly"],
    )
    hourly_on_mcs = build_hourly_on_mcs(tbls["scai_stage_hourly"], tx)
    mcs_starts = mcs_start_hours(hourly_on_mcs)

    df = (
        hourly_on_mcs[
            [
                "ENCOUNTER_ID",
                "HOUR_FROM_ADMIT",
                "EVENT_DT_TM",
                "SCAI_STAGE_NUM",
                "ON_MCS",
            ]
        ]
        .rename(columns={"SCAI_STAGE_NUM": "CURRENT_STAGE_NUM"})
        .merge(feats_4h, on=["ENCOUNTER_ID", "HOUR_FROM_ADMIT"], how="left")
        .merge(feats_6h, on=["ENCOUNTER_ID", "HOUR_FROM_ADMIT"], how="left")
        .merge(feats_12h, on=["ENCOUNTER_ID", "HOUR_FROM_ADMIT"], how="left")
        .merge(tx.drop(columns=["ON_MCS"], errors="ignore"), on=["ENCOUNTER_ID", "HOUR_FROM_ADMIT"], how="left")
        .merge(mcs_starts, on="ENCOUNTER_ID", how="left" if inference_mode else "inner")
    )

    enc = tbls["encounter"]
    person = tbls["person"]
    diag = tbls["diagnosis"]
    principal = (
        diag[diag.DIAG_PRIORITY == 1][["ENCOUNTER_ID", "NOMENCLATURE_CD"]]
        .rename(columns={"NOMENCLATURE_CD": "PRINCIPAL_DX_CODE"})
    )
    principal["PRINCIPAL_DX_CATEGORY"] = principal["PRINCIPAL_DX_CODE"].map(dx_category)

    static = (
        enc[["ENCOUNTER_ID", "PERSON_ID", "ADMIT_TYPE_CD", "ADMIT_SRC_CD", "REG_DT_TM"]]
        .merge(person[["PERSON_ID", "BIRTH_DT_TM", "SEX_CD", "RACE_CD", "ETHNICITY_CD"]], on="PERSON_ID", how="left")
        .merge(principal[["ENCOUNTER_ID", "PRINCIPAL_DX_CATEGORY"]], on="ENCOUNTER_ID", how="left")
    )
    static["AGE"] = ((static.REG_DT_TM - static.BIRTH_DT_TM).dt.days / 365.25).astype(int)
    static["WEIGHT_KG"] = np.where(static.SEX_CD == "M", 82.0, 70.0)
    static["BSA_M2"] = np.where(static.SEX_CD == "M", 1.97, 1.74)

    df = df.merge(
        static[
            [
                "ENCOUNTER_ID",
                "PERSON_ID",
                "AGE",
                "WEIGHT_KG",
                "BSA_M2",
                "SEX_CD",
                "RACE_CD",
                "ETHNICITY_CD",
                "ADMIT_TYPE_CD",
                "ADMIT_SRC_CD",
                "PRINCIPAL_DX_CATEGORY",
            ]
        ],
        on="ENCOUNTER_ID",
        how="left",
    )

    if inference_mode:
        df["MCS_START_HOUR"] = df["MCS_START_HOUR"].fillna(99999)
    else:
        mcs_encounters = set(mcs_escalation_encounter_ids(tbls["procedure_event"]))
        df = df[df["ENCOUNTER_ID"].isin(mcs_encounters)].copy()
    df["HOURS_UNTIL_MCS_START"] = df["MCS_START_HOUR"] - df["HOUR_FROM_ADMIT"]
    df = df[df["HOUR_FROM_ADMIT"] >= MIN_HOUR].copy()
    if not inference_mode:
        df = df[df["HOURS_UNTIL_MCS_START"] >= 1].copy()
    df = add_hours_on_current_stage(df)
    df = add_treatment_trend_features(df)
    df = add_hemodynamic_missingness(df)
    df = add_shock_indices(df)
    df = add_time_since_abnormal(df)
    df = add_cumulative_escalations(df)
    df = add_stage_trajectory_lags(df)
    df = add_shock_burden_features(df)
    df = add_escalation_proximity_features(df)
    df[MCS_LABEL_COL] = (df["HOURS_UNTIL_MCS_START"] <= MCS_HORIZON_HOURS).astype(int)
    return df


def apply_numeric_code_override(row: pd.Series, code: str, value: float) -> pd.Series:
    """Apply a clinician-supplied snapshot value for ``code`` as a
    **half-sustained** edit: assume the patient has been at this level
    for roughly the recent half of every lookback window, ramping up
    from the template's prior baseline over the older half. This
    matches a clinician's intuition when reading a current-hour
    snapshot ("patient is at HR 170 right now after deteriorating to
    get here") far better than a one-hour spike or a flat-forever
    rewrite, and produces a coherent feature row where ``_now``,
    ``_mean*h``, ``_min*h``, ``_max*h``, ``_delta*h`` and ``_slope*h``
    all agree with the same implied trajectory.

    For each window ``W`` with template values ``mean_T``, ``min_T``,
    ``max_T`` and ``delta_T`` (= now_T - value_T_hours_ago):

      - ``_now``  = ``value``
      - ``_mean`` = ``0.5 * value + 0.5 * mean_T`` (half new level,
        half prior trajectory)
      - ``_min``  = ``min(min_T, value)``
      - ``_max``  = ``max(max_T, value)``
      - ``_delta`` = ``value - (now_T - delta_T)`` (the value
        ``W`` hours ago comes from the template's history, unchanged)
      - ``_slope`` = ``_delta / W``
    """
    updated = row.copy()
    template_now_raw = row.get(f"{code}_now", value)
    template_now = float(template_now_raw) if pd.notna(template_now_raw) else float(value)

    window_specs = [
        (LOOKBACK_HOURS, ""),
        (MEDIUM_LOOKBACK_HOURS, f"_{MEDIUM_LOOKBACK_SUFFIX}"),
        (LONG_LOOKBACK_HOURS, f"_{LONG_LOOKBACK_SUFFIX}"),
    ]

    for window, lb_suffix in window_specs:
        now_col = f"{code}_now{lb_suffix}"
        mean_col = f"{code}_mean{window}h{lb_suffix}"
        min_col = f"{code}_min{window}h{lb_suffix}"
        max_col = f"{code}_max{window}h{lb_suffix}"
        delta_col = f"{code}_delta{window}h{lb_suffix}"
        slope_col = f"{code}_slope{window}h{lb_suffix}"

        prior_mean = (
            float(updated[mean_col])
            if mean_col in updated.index and pd.notna(updated[mean_col])
            else float(value)
        )
        prior_delta = (
            float(updated[delta_col])
            if delta_col in updated.index and pd.notna(updated[delta_col])
            else 0.0
        )
        value_W_ago = template_now - prior_delta

        if now_col in updated.index:
            updated[now_col] = float(value)
        if mean_col in updated.index:
            updated[mean_col] = 0.5 * float(value) + 0.5 * prior_mean
        if min_col in updated.index:
            prior_min = float(updated[min_col]) if pd.notna(updated[min_col]) else float(value)
            updated[min_col] = min(prior_min, float(value))
        if max_col in updated.index:
            prior_max = float(updated[max_col]) if pd.notna(updated[max_col]) else float(value)
            updated[max_col] = max(prior_max, float(value))
        if delta_col in updated.index:
            updated[delta_col] = float(value) - value_W_ago
        if slope_col in updated.index:
            updated[slope_col] = (float(value) - value_W_ago) / window

    missing_col = f"{code}_missing"
    if missing_col in updated.index:
        updated[missing_col] = 0
    for rule_code, direction, threshold in ABNORMAL_RULES:
        if rule_code != code:
            continue
        feature = f"HOURS_SINCE_{rule_code}_{direction.upper()}"
        if feature not in updated.index:
            continue
        crossed = (direction == "high" and value > threshold) or (
            direction == "low" and value < threshold
        )
        updated[feature] = 0.0 if crossed else -1.0
    if code in CODES_FOR_SHOCK_INDICES:
        updated = _refresh_shock_indices(updated)
    return updated


def _refresh_shock_indices(row: pd.Series) -> pd.Series:
    """Recompute shock-index features from the row's `_now` values after an override."""
    updated = row.copy()
    for name, a_code, b_code, _ in SHOCK_INDEX_DEFS:
        a_now = updated.get(f"{a_code}_now", np.nan)
        b_now = updated.get(f"{b_code}_now", np.nan)
        feat = f"{name}_now"
        if feat in updated.index:
            updated[feat] = _compute_shock_index(name, a_now, b_now)
        a_mean = updated.get(f"{a_code}_mean{LOOKBACK_HOURS}h", np.nan)
        b_mean = updated.get(f"{b_code}_mean{LOOKBACK_HOURS}h", np.nan)
        feat_mean = f"{name}_mean{LOOKBACK_HOURS}h"
        if feat_mean in updated.index:
            updated[feat_mean] = _compute_shock_index(name, a_mean, b_mean)
        a_max = updated.get(f"{a_code}_max{LOOKBACK_HOURS}h", np.nan)
        b_min = updated.get(f"{b_code}_min{LOOKBACK_HOURS}h", np.nan)
        feat_max = f"{name}_max{LOOKBACK_HOURS}h"
        if feat_max in updated.index:
            updated[feat_max] = _compute_shock_index(name, a_max, b_min)
    return updated


def build_feature_row(
    template: pd.Series,
    *,
    hour_from_admit: int,
    current_stage_num: int,
    n_vasopressors_active: int,
    n_inotropes_active: int,
    on_intubation: bool,
    on_crrt: bool,
    age: int,
    weight_kg: float,
    bsa_m2: float,
    vitals_labs: dict[str, float],
) -> pd.Series:
    row = template.copy()
    row["HOUR_FROM_ADMIT"] = hour_from_admit
    row["CURRENT_STAGE_NUM"] = current_stage_num
    row["N_VASOPRESSORS_ACTIVE"] = n_vasopressors_active
    row["N_INOTROPES_ACTIVE"] = n_inotropes_active
    row["ON_INTUBATION"] = int(on_intubation)
    row["ON_CRRT"] = int(on_crrt)
    row["AGE"] = age
    row["WEIGHT_KG"] = weight_kg
    row["BSA_M2"] = bsa_m2
    for code, value in vitals_labs.items():
        row = apply_numeric_code_override(row, code, value)
    return row


def frame_to_model_matrix(
    row: pd.Series,
    feature_cols: list[str],
    cat_feats: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame([row[feature_cols]])
    for col in cat_feats:
        matrix[col] = matrix[col].astype("category")
    return matrix


@dataclass(frozen=True)
class MCSModelBundle:
    model: Any
    feature_cols: list[str]
    cat_feats: list[str]
    df: pd.DataFrame
    default_threshold: float = DEFAULT_THRESHOLD
    scale_pos_weight: float | None = None


def trim_pre_mcs_context(
    df: pd.DataFrame,
    max_hours_until: int = PRE_MCS_CONTEXT_HOURS,
) -> pd.DataFrame:
    """Keep pre-MCS hours in the lead-up window before placement."""
    return df[df["HOURS_UNTIL_MCS_START"] <= max_hours_until].copy()


def prepare_feature_frame(
    df: pd.DataFrame,
    indices: np.ndarray,
    feature_cols: list[str],
    cat_feats: list[str],
) -> pd.DataFrame:
    frame = df.iloc[indices][feature_cols].copy()
    for col in cat_feats:
        frame[col] = frame[col].astype("category")
    return frame


def encounter_balanced_weights(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    encounters = df.iloc[indices]["ENCOUNTER_ID"]
    counts = encounters.map(encounters.value_counts()).to_numpy(dtype=float)
    return 1.0 / counts


def assert_no_patient_leakage(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    group_col: str = "PERSON_ID",
) -> None:
    """Raise AssertionError if any patient appears in more than one split.

    Group leakage – the same patient's rows ending up on both sides of
    the train/test split – is the single biggest source of optimistic
    performance estimates with temporally repeated measurements. We
    split with GroupShuffleSplit on PERSON_ID, but call this guard
    anywhere a split is consumed so a refactor cannot silently break
    the invariant.
    """
    if group_col not in df.columns:
        raise KeyError(f"{group_col!r} not in dataframe columns")
    train_patients = set(df.iloc[train_idx][group_col].unique())
    val_patients = set(df.iloc[val_idx][group_col].unique())
    test_patients = set(df.iloc[test_idx][group_col].unique())
    train_val = train_patients & val_patients
    train_test = train_patients & test_patients
    val_test = val_patients & test_patients
    if train_val or train_test or val_test:
        msg = [
            f"Patient leakage detected on group {group_col!r}:",
            f"  train∩val  ({len(train_val)} patients): {sorted(train_val)[:5]}{'...' if len(train_val) > 5 else ''}",
            f"  train∩test ({len(train_test)} patients): {sorted(train_test)[:5]}{'...' if len(train_test) > 5 else ''}",
            f"  val∩test   ({len(val_test)} patients): {sorted(val_test)[:5]}{'...' if len(val_test) > 5 else ''}",
        ]
        raise AssertionError("\n".join(msg))


def base_scale_pos_weight(y_train: np.ndarray) -> float:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    return neg / max(pos, 1)


def _strong_xgb_params(
    *,
    scale_pos_weight: float,
    seed: int,
    early_stopping: bool = True,
    tighten_regularization: bool = False,
    eval_metric: str = "aucpr",
    slow_lr: bool = False,
) -> dict[str, Any]:
    """Hyperparameters tuned for the MCS task: enough capacity to capture
    the shock-score boundary while keeping overfit small via early stopping
    and moderate regularization.

    Knobs:
      - ``tighten_regularization``: shrinks the model further (``max_depth``
        5→4, ``min_child_weight`` 15→25, stronger L2) to bring the
        train→test Brier gap down at a small cost to AUROC.
      - ``eval_metric``: "aucpr" optimizes ranking; "logloss" optimizes
        the calibration directly, which usually nudges Brier down a few
        percent because early stopping halts before the model starts
        sharpening probabilities at the expense of log-loss.
      - ``slow_lr``: drops ``learning_rate`` 0.03→0.01 and bumps
        ``n_estimators`` (and the early-stopping patience) so the model
        converges to better-calibrated probabilities at the cost of ~2x
        training time.
    """
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": eval_metric,
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 15,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.2,
        "gamma": 0.5,
        "scale_pos_weight": scale_pos_weight,
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": -1,
    }
    if tighten_regularization:
        params.update(
            {
                "max_depth": 4,
                "min_child_weight": 25,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "reg_lambda": 3.0,
                "reg_alpha": 0.3,
                "gamma": 0.8,
            }
        )
    early_stop_rounds = EARLY_STOPPING_ROUNDS
    if slow_lr:
        params["learning_rate"] = 0.01
        params["n_estimators"] = 3000
        early_stop_rounds = max(EARLY_STOPPING_ROUNDS, 60)
    if early_stopping:
        params["early_stopping_rounds"] = early_stop_rounds
    return params


class _SmoothLabelXGBClassifier:
    """Thin adapter that lets XGBoost train with **smoothed targets**
    (y=0 → ``eps``, y=1 → ``1-eps``) by routing through ``XGBRegressor``
    with ``objective="binary:logistic"``. The regressor accepts
    continuous labels in [0, 1]; minimizing log-loss against the smoothed
    target prevents the trees from pushing predicted probabilities all
    the way to 0/1, which is the dominant source of Brier loss on
    confidently-wrong examples.

    The interface mirrors the bits of ``XGBClassifier`` that the rest
    of this module relies on (``fit``, ``predict_proba``, ``predict``,
    ``get_booster``, ``best_iteration``, ``get_params``).
    """

    def __init__(self, eps: float, **xgb_params: Any) -> None:
        from xgboost import XGBRegressor

        params = dict(xgb_params)
        params["objective"] = "binary:logistic"
        # XGBRegressor defaults to "rmse" which would override our choice
        # of "logloss" / "aucpr" eval metric, so make sure ours sticks.
        params.setdefault("eval_metric", "logloss")
        self.eps = float(eps)
        self.params = params
        self._reg = XGBRegressor(**params)
        self.best_iteration: int | None = None

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) * (1.0 - 2.0 * self.eps) + self.eps

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        *,
        eval_set: list[tuple[pd.DataFrame, np.ndarray]] | None = None,
        sample_weight: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "_SmoothLabelXGBClassifier":
        smoothed_eval = (
            [(xe, self._smooth(ye)) for xe, ye in eval_set] if eval_set else None
        )
        fit_kwargs: dict[str, Any] = {"verbose": verbose}
        if smoothed_eval is not None:
            fit_kwargs["eval_set"] = smoothed_eval
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self._reg.fit(x, self._smooth(y), **fit_kwargs)
        self.best_iteration = getattr(self._reg, "best_iteration", None)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.clip(self._reg.predict(x), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def get_booster(self) -> Any:
        return self._reg.get_booster()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self.params)


def _make_xgb_estimator(
    *,
    scale_pos_weight: float,
    seed: int,
    early_stopping: bool,
    tighten_regularization: bool,
    eval_metric: str,
    slow_lr: bool,
    label_smoothing: float,
    n_estimators_override: int | None = None,
) -> Any:
    params = _strong_xgb_params(
        scale_pos_weight=scale_pos_weight,
        seed=seed,
        early_stopping=early_stopping,
        tighten_regularization=tighten_regularization,
        eval_metric=eval_metric,
        slow_lr=slow_lr,
    )
    if n_estimators_override is not None:
        params["n_estimators"] = int(max(50, n_estimators_override))
        params.pop("early_stopping_rounds", None)
    if label_smoothing and label_smoothing > 0:
        return _SmoothLabelXGBClassifier(eps=label_smoothing, **params)
    return XGBClassifier(**params)


def _fit_regularized_xgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    scale_pos_weight: float,
    seed: int,
    sample_weight: np.ndarray | None = None,
    tighten_regularization: bool = False,
    eval_metric: str = "aucpr",
    slow_lr: bool = False,
    label_smoothing: float = 0.0,
) -> Any:
    model = _make_xgb_estimator(
        scale_pos_weight=scale_pos_weight,
        seed=seed,
        early_stopping=True,
        tighten_regularization=tighten_regularization,
        eval_metric=eval_metric,
        slow_lr=slow_lr,
        label_smoothing=label_smoothing,
    )
    fit_kwargs: dict[str, Any] = {"verbose": False, "eval_set": [(x_val, y_val)]}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def _fit_regularized_xgb_fixed_rounds(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    scale_pos_weight: float,
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None = None,
    tighten_regularization: bool = False,
    eval_metric: str = "aucpr",
    slow_lr: bool = False,
    label_smoothing: float = 0.0,
) -> Any:
    """Fit XGB without early stopping (no eval set required), using a
    fixed number of boosting rounds. Used for the final model trained
    on combined train+val data, where the OOF calibration step has
    already estimated the right number of trees."""
    model = _make_xgb_estimator(
        scale_pos_weight=scale_pos_weight,
        seed=seed,
        early_stopping=False,
        tighten_regularization=tighten_regularization,
        eval_metric=eval_metric,
        slow_lr=slow_lr,
        label_smoothing=label_smoothing,
        n_estimators_override=int(max(50, n_estimators)),
    )
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def oof_calibration_predictions(
    df: pd.DataFrame,
    pool_idx: np.ndarray,
    feature_cols: list[str],
    cat_feats: list[str],
    *,
    scale_pos_weight: float,
    seed: int,
    n_splits: int = 5,
    sample_weight_fn: Any = None,
    tighten_regularization: bool = False,
    eval_metric: str = "aucpr",
    slow_lr: bool = False,
    label_smoothing: float = 0.0,
    label_col: str = MCS_LABEL_COL,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate out-of-fold predictions on a patient-grouped calibration
    pool. Each fold's predictions come from a model that never saw the
    fold's patients, which yields unbiased probabilities the calibrator
    can fit on.

    Returns:
        oof_proba: float array aligned with ``pool_idx`` order.
        oof_y:     true labels in the same order.
        median_best_iter: median ``best_iteration`` from the fold
            models — used as ``n_estimators`` for the final retrain
            on the full pool (since no eval set will be available).
    """
    from sklearn.model_selection import GroupKFold

    groups = df.iloc[pool_idx]["PERSON_ID"].to_numpy()
    pool_y = df.iloc[pool_idx][label_col].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros(len(pool_idx), dtype=float)
    fold_best_iters: list[int] = []
    for fold_id, (tr_local, va_local) in enumerate(
        gkf.split(pool_idx, pool_y, groups=groups), start=1
    ):
        tr_idx = pool_idx[tr_local]
        va_idx = pool_idx[va_local]
        x_tr = prepare_feature_frame(df, tr_idx, feature_cols, cat_feats)
        x_va = prepare_feature_frame(df, va_idx, feature_cols, cat_feats)
        y_tr = df.iloc[tr_idx][label_col].to_numpy()
        y_va = df.iloc[va_idx][label_col].to_numpy()
        sw_tr = sample_weight_fn(df, tr_idx) if sample_weight_fn is not None else None
        m = _fit_regularized_xgb(
            x_tr,
            y_tr,
            x_va,
            y_va,
            scale_pos_weight=scale_pos_weight,
            seed=seed + fold_id,
            sample_weight=sw_tr,
            tighten_regularization=tighten_regularization,
            eval_metric=eval_metric,
            slow_lr=slow_lr,
            label_smoothing=label_smoothing,
        )
        oof_proba[va_local] = m.predict_proba(x_va)[:, 1]
        best_iter = getattr(m, "best_iteration", None)
        if best_iter is None:
            best_iter = int(m.get_params().get("n_estimators", 600))
        fold_best_iters.append(int(best_iter))
    median_best_iter = int(np.median(fold_best_iters)) if fold_best_iters else 600
    return oof_proba, pool_y, median_best_iter


def tune_scale_pos_weight(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    base_spw: float,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> float:
    candidates = [0.5 * base_spw, base_spw, 1.5 * base_spw, 2.0 * base_spw]
    best_spw = base_spw
    best_auprc = -1.0
    for spw in candidates:
        candidate = _fit_regularized_xgb(
            x_train,
            y_train,
            x_val,
            y_val,
            scale_pos_weight=spw,
            seed=seed,
            sample_weight=sample_weight,
        )
        val_proba = candidate.predict_proba(x_val)[:, 1]
        val_auprc = average_precision_score(y_val, val_proba)
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_spw = spw
    return best_spw


def select_features_by_importance(
    model: XGBClassifier,
    feature_cols: list[str],
    cat_feats: list[str],
    *,
    min_features: int = 20,
) -> tuple[list[str], list[str]]:
    """Drop features that contributed zero gain in a pilot fit."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    selected = [name for name in feature_cols if gain.get(name, 0.0) > 0.0]
    if len(selected) < min_features:
        return feature_cols, cat_feats
    selected_cats = [c for c in cat_feats if c in selected]
    return selected, selected_cats


def threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion-matrix metrics at a single threshold."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(acc),
    }


def threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.05,
) -> pd.DataFrame:
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    return pd.DataFrame([threshold_metrics(y_true, proba, float(t)) for t in grid])


def find_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    criterion: str = "accuracy",
    min_specificity: float = 0.0,
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.01,
) -> float:
    """Pick threshold on OOF/val predictions.

    ``criterion`` may be ``accuracy``, ``sensitivity``, or ``youden``
    (sensitivity + specificity - 1). For ``sensitivity``, thresholds
    below ``min_specificity`` are skipped.
    """
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    y_true = np.asarray(y_true).astype(int)
    best_t = 0.5
    best_score = -1.0
    for t in grid:
        m = threshold_metrics(y_true, proba, float(t))
        if criterion == "sensitivity":
            if m["specificity"] < min_specificity:
                continue
            score = m["sensitivity"]
        elif criterion == "youden":
            score = m["sensitivity"] + m["specificity"] - 1.0
        else:
            score = m["accuracy"]
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def find_best_accuracy_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.01,
) -> float:
    return find_best_threshold(y_true, proba, criterion="accuracy", lo=lo, hi=hi, step=step)


@dataclass
class EnsembleXGBClassifier:
    """Average probabilities across multiple XGB models."""

    models: list[XGBClassifier]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probas = np.mean([m.predict_proba(x)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1.0 - probas, probas])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


@dataclass
class IsotonicCalibratedModel:
    """Wrap a probabilistic classifier with an isotonic calibrator fit on
    held-out (validation) data. Preserves ranking but spreads probabilities
    to align with the empirical positive rate, which is necessary when the
    base XGB ensemble compresses outputs to a narrow range."""

    base: Any
    calibrator: IsotonicRegression

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(x)[:, 1]
        cal = self.calibrator.predict(raw)
        cal = np.clip(cal, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def fit(base: Any, raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibratedModel":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw.astype(float), y.astype(int))
        return IsotonicCalibratedModel(base=base, calibrator=iso)

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator.predict(raw.astype(float)), 1e-4, 1.0 - 1e-4)


@dataclass
class SigmoidCalibratedModel:
    """Platt (sigmoid) calibration: maps raw probabilities through
    p = 1 / (1 + exp(-(a * logit(raw) + b))) with (a, b) fit on a
    held-out set. Unlike isotonic regression, this mapping is strictly
    monotone and smooth, so two adjacent raw scores always map to two
    adjacent calibrated scores. That's essential for the dashboard,
    where small slider tweaks should produce visible probability
    changes (isotonic creates step plateaus that flatten the output).
    """

    base: Any
    a: float
    b: float

    @staticmethod
    def fit(base: Any, raw: np.ndarray, y: np.ndarray) -> "SigmoidCalibratedModel":
        eps = 1e-6
        raw_clipped = np.clip(raw.astype(float), eps, 1.0 - eps)
        logit = np.log(raw_clipped / (1.0 - raw_clipped)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(logit, y)
        a = float(lr.coef_[0, 0])
        b = float(lr.intercept_[0])
        return SigmoidCalibratedModel(base=base, a=a, b=b)

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        eps = 1e-6
        raw_clipped = np.clip(raw.astype(float), eps, 1.0 - eps)
        logit = np.log(raw_clipped / (1.0 - raw_clipped))
        z = self.a * logit + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(x)[:, 1]
        cal = self._calibrate(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


@dataclass
class BetaCalibratedModel:
    """Beta calibration (Kull et al., 2017): three-parameter family that
    fits ``p_cal = σ(a·log(p) + b·log(1-p) + c)``. It strictly
    generalises Platt scaling (Platt requires ``a = -b``) and tends to
    deliver lower Brier than Platt whenever the model's miscalibration
    is asymmetric between the low and high probability regions, which
    is typical for a tree ensemble whose probabilities saturate near
    0 and 1 but lag in the middle.
    """

    base: Any
    a: float
    b: float
    c: float

    @staticmethod
    def fit(base: Any, raw: np.ndarray, y: np.ndarray) -> "BetaCalibratedModel":
        eps = 1e-6
        p = np.clip(raw.astype(float), eps, 1.0 - eps)
        x = np.column_stack([np.log(p), np.log(1.0 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x, y)
        a = float(lr.coef_[0, 0])
        b = float(lr.coef_[0, 1])
        c = float(lr.intercept_[0])
        return BetaCalibratedModel(base=base, a=a, b=b, c=c)

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(raw.astype(float), eps, 1.0 - eps)
        z = self.a * np.log(p) + self.b * np.log(1.0 - p) + self.c
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(x)[:, 1]
        cal = self._calibrate(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def fit_calibrator(method: str, base: Any, raw: np.ndarray, y: np.ndarray) -> Any:
    """Factory for the supported calibration methods."""
    method = method.lower()
    if method == "sigmoid":
        return SigmoidCalibratedModel.fit(base, raw, y)
    if method == "beta":
        return BetaCalibratedModel.fit(base, raw, y)
    if method == "isotonic":
        return IsotonicCalibratedModel.fit(base, raw, y)
    raise ValueError(f"Unknown calibration method: {method!r}")


def fit_mcs_xgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    scale_pos_weight: float,
    seed: int,
    sample_weight: np.ndarray | None = None,
    early_stopping: bool = False,
) -> XGBClassifier:
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 500 if early_stopping else 400,
        "learning_rate": 0.05,
        "max_depth": 4 if early_stopping else 5,
        "min_child_weight": 25 if early_stopping else 20,
        "subsample": 0.85 if early_stopping else 0.8,
        "colsample_bytree": 0.85 if early_stopping else 0.8,
        "reg_lambda": 1.5 if early_stopping else 1.0,
        "scale_pos_weight": scale_pos_weight,
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": -1,
    }
    if early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = XGBClassifier(**params)
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if early_stopping:
        fit_kwargs["eval_set"] = [(x_val, y_val)]
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def calibrate_mcs_model(model: XGBClassifier, x_val: pd.DataFrame, y_val: np.ndarray) -> Any:
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(x_val, y_val)
    return calibrated


def train_mcs_model(df: pd.DataFrame, seed: int = RANDOM_SEED) -> MCSModelBundle:
    num_feats, cat_feats = mcs_feature_schema(df)
    feature_cols = num_feats + cat_feats
    tr, _, _ = patient_split(df, seed=seed)
    x_train = prepare_feature_frame(df, tr, feature_cols, cat_feats)
    y_train = df.iloc[tr][MCS_LABEL_COL].to_numpy()
    scale_pos_weight = base_scale_pos_weight(y_train)
    model = fit_mcs_xgb(
        x_train,
        y_train,
        x_train,
        y_train,
        scale_pos_weight=scale_pos_weight,
        seed=seed,
        early_stopping=False,
    )
    return MCSModelBundle(
        model=model,
        feature_cols=feature_cols,
        cat_feats=cat_feats,
        df=df,
        scale_pos_weight=scale_pos_weight,
    )


def train_tuned_mcs_model(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    *,
    trim_context: bool = True,
    encounter_balance: bool = False,
    calibrate: bool = False,
    feature_select: bool = True,
    ensemble_seeds: int = 5,
    tune_threshold: bool = True,
    threshold_criterion: str = "accuracy",
    min_specificity: float = 0.0,
    scale_pos_weight_multiplier: float = 1.0,
    tune_scale_pos_weight_flag: bool = False,
    oof_calibration: bool = True,
    oof_splits: int = 5,
    calibration_method: str = "sigmoid",
    tighten_regularization: bool = False,
    eval_metric: str = "aucpr",
    slow_lr: bool = False,
    label_smoothing: float = 0.0,
    label_col: str = MCS_LABEL_COL,
    extra_lookbacks: tuple[tuple[int, str], ...] = (),
) -> MCSModelBundle:
    work = trim_pre_mcs_context(df) if trim_context else df.copy()
    num_feats, cat_feats = mcs_feature_schema(work, extra_lookbacks=extra_lookbacks)
    feature_cols = num_feats + cat_feats
    tr, va, _ = patient_split(work, seed=seed)
    y_train = work.iloc[tr][label_col].to_numpy()
    y_val = work.iloc[va][label_col].to_numpy()
    sample_weight = encounter_balanced_weights(work, tr) if encounter_balance else None
    base_spw = base_scale_pos_weight(y_train)

    x_train = prepare_feature_frame(work, tr, feature_cols, cat_feats)
    x_val = prepare_feature_frame(work, va, feature_cols, cat_feats)

    if feature_select:
        pilot = _fit_regularized_xgb(
            x_train,
            y_train,
            x_val,
            y_val,
            scale_pos_weight=base_spw,
            seed=seed,
            sample_weight=sample_weight,
            tighten_regularization=tighten_regularization,
            eval_metric=eval_metric,
            slow_lr=slow_lr,
            label_smoothing=label_smoothing,
        )
        feature_cols, cat_feats = select_features_by_importance(pilot, feature_cols, cat_feats)
        x_train = prepare_feature_frame(work, tr, feature_cols, cat_feats)
        x_val = prepare_feature_frame(work, va, feature_cols, cat_feats)

    if tune_scale_pos_weight_flag:
        best_spw = tune_scale_pos_weight(
            x_train,
            y_train,
            x_val,
            y_val,
            base_spw=base_spw,
            seed=seed,
            sample_weight=sample_weight,
        )
    else:
        best_spw = base_spw
    best_spw = best_spw * scale_pos_weight_multiplier

    n_seeds = max(1, ensemble_seeds)

    if oof_calibration:
        # 1. Generate unbiased predictions on the combined train+val pool
        #    via patient-grouped K-fold CV. This gives the calibrator ~5x
        #    more data than val-only and the predictions are leak-free
        #    because each fold's held-out rows come from models that
        #    never trained on the patient.
        pool_idx = np.concatenate([tr, va])
        sample_weight_fn = encounter_balanced_weights if encounter_balance else None
        oof_proba, oof_y, median_best_iter = oof_calibration_predictions(
            work,
            pool_idx,
            feature_cols,
            cat_feats,
            scale_pos_weight=best_spw,
            seed=seed,
            n_splits=oof_splits,
            sample_weight_fn=sample_weight_fn,
            tighten_regularization=tighten_regularization,
            eval_metric=eval_metric,
            slow_lr=slow_lr,
            label_smoothing=label_smoothing,
            label_col=label_col,
        )

        # 2. Retrain the final ensemble on the combined train+val pool
        #    with a fixed number of boosting rounds (no held-out eval
        #    set is needed because the OOF folds already told us how
        #    many trees to keep).
        x_pool = prepare_feature_frame(work, pool_idx, feature_cols, cat_feats)
        y_pool = work.iloc[pool_idx][label_col].to_numpy()
        pool_sample_weight = (
            encounter_balanced_weights(work, pool_idx) if encounter_balance else None
        )
        models = [
            _fit_regularized_xgb_fixed_rounds(
                x_pool,
                y_pool,
                scale_pos_weight=best_spw,
                seed=seed + offset,
                n_estimators=median_best_iter,
                sample_weight=pool_sample_weight,
                tighten_regularization=tighten_regularization,
                eval_metric=eval_metric,
                slow_lr=slow_lr,
                label_smoothing=label_smoothing,
            )
            for offset in range(n_seeds)
        ]
        base_model = models[0] if n_seeds == 1 else EnsembleXGBClassifier(models)

        # 3. Fit the chosen calibrator on the OOF predictions instead of
        #    val-only. Same calibrator class; much more data, lower Brier.
        model: Any = fit_calibrator(calibration_method, base_model, oof_proba, oof_y)
        # Calibrated OOF predictions are leak-free (each fold's
        # held-out rows came from a model that never trained on the
        # patient, and the calibrator was fit on the same OOF preds)
        # — use them to pick the threshold below. ``model.predict_proba``
        # on train/val is now in-sample because the final ensemble was
        # retrained on the full train+val pool.
        if hasattr(model, "_calibrate"):
            threshold_proba = model._calibrate(oof_proba)
        else:
            threshold_proba = oof_proba
        threshold_y = oof_y
    else:
        models = [
            _fit_regularized_xgb(
                x_train,
                y_train,
                x_val,
                y_val,
                scale_pos_weight=best_spw,
                seed=seed + offset,
                sample_weight=sample_weight,
                tighten_regularization=tighten_regularization,
                eval_metric=eval_metric,
                slow_lr=slow_lr,
                label_smoothing=label_smoothing,
            )
            for offset in range(n_seeds)
        ]
        base_model = models[0] if n_seeds == 1 else EnsembleXGBClassifier(models)
        raw_val_proba = base_model.predict_proba(x_val)[:, 1]
        model = fit_calibrator(calibration_method, base_model, raw_val_proba, y_val)
        # Non-OOF path: val was not used to train the booster, so
        # calibrated val predictions are a fair signal for threshold
        # tuning (only mildly biased because the calibrator itself was
        # fit on val).
        threshold_proba = model.predict_proba(x_val)[:, 1]
        threshold_y = y_val

    if calibrate and n_seeds == 1:
        model = calibrate_mcs_model(models[0], x_val, y_val)
        threshold_proba = model.predict_proba(x_val)[:, 1]
        threshold_y = y_val

    default_threshold = DEFAULT_THRESHOLD
    if tune_threshold:
        default_threshold = find_best_threshold(
            threshold_y,
            threshold_proba,
            criterion=threshold_criterion,
            min_specificity=min_specificity,
        )

    return MCSModelBundle(
        model=model,
        feature_cols=feature_cols,
        cat_feats=cat_feats,
        df=work,
        default_threshold=default_threshold,
        scale_pos_weight=best_spw,
    )


def load_mcs_model_bundle(
    data_dir: str | Path = "data",
    seed: int = RANDOM_SEED,
    *,
    alert_threshold: float = CLINICAL_ALERT_THRESHOLD,
) -> MCSModelBundle:
    """Load data and train MCS 12h for high-sensitivity screening.

    Uses sensitivity-oriented training (AUPRC, class-weight boost,
    escalation-proximity features) and a fixed alert threshold of 0.10
    by default — tuned to minimize missed imminent escalations.
    """
    tables = load_tables(str(data_dir))
    mcs_encs = frozenset(mcs_escalation_encounter_ids(tables["procedure_event"]))
    df = build_mcs_prediction_frame(filter_tables_to_encounters(tables, mcs_encs))
    bundle = train_tuned_mcs_model(df, seed=seed, **SENSITIVITY_TRAIN_KWARGS)
    return replace(bundle, default_threshold=alert_threshold)


def predict_imminent_mcs_probability(
    bundle: MCSModelBundle,
    row: pd.Series,
) -> float:
    matrix = frame_to_model_matrix(row, bundle.feature_cols, bundle.cat_feats)
    return float(bundle.model.predict_proba(matrix)[:, 1][0])


def smooth_predictions_within_encounter(
    df: pd.DataFrame,
    proba: np.ndarray,
    *,
    alpha: float = 0.5,
    encounter_col: str = "ENCOUNTER_ID",
    hour_col: str = "HOUR_FROM_ADMIT",
) -> np.ndarray:
    """Apply causal exponential smoothing of predictions within each
    encounter (sorted by hour). At hour t we emit
    ``alpha * raw_t + (1 - alpha) * smoothed_{t-1}``, which means
    smoothing only uses past predictions and never leaks future
    information — safe for both training-time evaluation and live
    inference if the model is being polled hour-by-hour.

    A row-by-row model that ignores temporal correlation is high
    variance: at the prediction-horizon boundary, consecutive hours
    can flip the predicted class on noise. Smoothing damps that
    high-frequency component and typically lowers Brier and the
    flip-rate without hurting AUROC/AUPRC.

    Parameters
    ----------
    df : DataFrame
        Must contain ``encounter_col`` and ``hour_col``. Index/order
        of ``df`` defines the order of ``proba``.
    proba : ndarray of shape (len(df),)
        Raw per-row predicted positive-class probabilities.
    alpha : float in (0, 1]
        EMA weight on the current hour's raw prediction. Smaller
        values smooth more aggressively. 0.5 = recent-hour weight
        ~50%, prior smoothed ~50%; equivalent to a half-life of 1h.
        1.0 disables smoothing (returns ``proba`` unchanged).
    """
    if alpha >= 1.0:
        return np.asarray(proba, dtype=float)
    if len(proba) != len(df):
        raise ValueError(
            f"len(proba)={len(proba)} but len(df)={len(df)}; they must align."
        )
    work = pd.DataFrame(
        {
            "_proba": np.asarray(proba, dtype=float),
            encounter_col: df[encounter_col].values,
            hour_col: df[hour_col].values,
            "_orig": np.arange(len(df)),
        }
    )
    work = work.sort_values([encounter_col, hour_col], kind="mergesort")
    work["_smooth"] = (
        work.groupby(encounter_col)["_proba"]
        .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        .astype(float)
    )
    work = work.sort_values("_orig", kind="mergesort")
    return work["_smooth"].to_numpy()


def representative_template_row(df: pd.DataFrame, positive: bool) -> pd.Series:
    subset = df[df[MCS_LABEL_COL] == int(positive)]
    if subset.empty:
        subset = df
    base = subset.iloc[0].copy()
    numeric = subset.select_dtypes(include="number")
    if not numeric.empty:
        base.loc[numeric.columns] = numeric.median()
    return base


def cohort_reference_medians(df: pd.DataFrame) -> dict[str, float]:
    fields = (
        DASHBOARD_TIME_FIELDS
        + DASHBOARD_STATIC_NUMERIC
        + [f"{code}_now" for code in DASHBOARD_NUMERIC_CODES]
    )
    medians: dict[str, float] = {}
    for field in fields:
        if field not in df.columns:
            continue
        value = df[field].median()
        if pd.notna(value):
            medians[field] = float(value)
    return medians
