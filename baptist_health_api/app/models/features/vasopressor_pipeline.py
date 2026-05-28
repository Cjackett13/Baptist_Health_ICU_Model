"""
vasopressor_pipeline.py — Inference-only preprocessing + feature engineering.

Steps mirroring the training pipeline:
  1. Clinical range nulling   (preprocessing.py step 3)
  2. was_measured_ flags      (preprocessing.py step 4)
  3. LOCF                     (preprocessing.py step 5)
  4. KNN imputation           (preprocessing.py step 7, inference path)
  5. IQR capping              (preprocessing.py step 8, inference path)
  6. log1p transforms         (preprocessing.py step 9)
  7. DuckDB feature engineering — Blocks A (deltas), B (rolling), C (scores), D (vasopressor)
  8. V3 additional features   (HOURS_SINCE_LAST_REAL, composite shock, stage flags,
                               journey/persistence, lab cadence)

VERIFY: Step 8 thresholds and the COMPOSITE_SHOCK_SCORE formula must be confirmed
against Christie's v3 feature_engineering.py before production use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

# ── Constants (verbatim from preprocessing.py) ────────────────────────────────

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

CLINICAL_RANGES: dict[str, tuple[float, float]] = {
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
    "NT_PROBNP":    (5,     50000),
    "TROPONIN_I":   (0,     50000),
    "PH":           (6.5,   7.9),
    "HCO3":         (5,     50),
    "WBC":          (0.5,   100),
    "HGB":          (2,     25),
    "PLT":          (5,     1500),
    "VIS":          (0,     200),
}

# Features included in rolling windows (verbatim from feature_engineering.py)
TRENDING_FEATURES = [
    "HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP", "URINE_OUT_HR",
    "CVP", "CO", "CI", "SVO2", "CPO", "PAPI",
    "LACTATE", "CREATININE", "BUN", "NT_PROBNP", "TROPONIN_I",
    "PH", "HCO3", "WBC", "HGB", "PLT",
    "VIS", "SCAI_STAGE_NUM",
]
WINDOWS = [1, 4, 24]

VASOPRESSORS = ["NOREPINEPHRINE", "EPINEPHRINE", "VASOPRESSIN", "PHENYLEPHRINE", "DOPAMINE"]
_vaso_list   = ", ".join(f"'{v}'" for v in VASOPRESSORS)

# Hemodynamic-monitoring sparse features that get HOURS_SINCE_LAST_REAL
HEMO_SPARSE = ["CO", "CI", "CPO", "CVP", "PAPI", "SVO2"]


# ── Helper ────────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index, dtype=np.float64)


# ── Step 3 — Clinical range nulling ──────────────────────────────────────────

def apply_clinical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat, (lo, hi) in CLINICAL_RANGES.items():
        if feat in df.columns:
            df[feat] = df[feat].where((df[feat] >= lo) & (df[feat] <= hi), other=np.nan)
    return df


# ── Step 4 — was_measured_ flags ─────────────────────────────────────────────

def add_was_measured_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat in SPARSE_FEATURES:
        if feat in df.columns:
            df[f"was_measured_{feat}"] = df[feat].notna().astype(np.int8)
    return df


# ── Step 5 — LOCF ─────────────────────────────────────────────────────────────

def apply_locf(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    df = df.sort_values(["ENCOUNTER_ID", "HOUR_FROM_ADMIT"]).copy()
    df[feat_cols] = df.groupby("ENCOUNTER_ID")[feat_cols].transform(lambda g: g.ffill())
    return df


# ── Step 7 — KNN imputation (inference path) ─────────────────────────────────

def apply_knn_imputation_inference(df: pd.DataFrame, imputer_state: dict) -> pd.DataFrame:
    df = df.copy()
    feat_cols      = imputer_state["feat_cols"]
    train_medians  = imputer_state["train_medians"]
    feature_models = imputer_state["feature_models"]

    for feat in feat_cols:
        if feat not in df.columns:
            continue
        miss_mask = df[feat].isna()
        if not miss_mask.any():
            continue

        entry = feature_models.get(feat)
        if entry is None:
            df.loc[miss_mask, feat] = train_medians.get(feat, 0.0)
            continue

        nn, y_obs = entry
        coord_cols = [c for c in feat_cols if c != feat]
        fill_vals  = {c: train_medians.get(c, 0.0) for c in coord_cols}
        X_miss     = df.loc[miss_mask, coord_cols].fillna(fill_vals).values
        _, indices = nn.kneighbors(X_miss)
        df.loc[miss_mask, feat] = y_obs[indices].mean(axis=1)

    return df


# ── Step 8 — IQR capping (inference path) ────────────────────────────────────

def apply_iqr_capping_inference(df: pd.DataFrame, iqr_bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for feat, bounds in iqr_bounds.items():
        if feat in df.columns:
            df[feat] = df[feat].clip(lower=bounds["lower"], upper=bounds["upper"])
    return df


# ── Step 9 — log1p transforms ────────────────────────────────────────────────

def apply_log_transforms_inference(df: pd.DataFrame, log_features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for feat in log_features:
        if feat in df.columns:
            df[feat] = np.log1p(df[feat].clip(lower=0))
    return df


# ── Steps 7 (DuckDB) — Blocks A, B, C, D ────────────────────────────────────

def _delta_exprs() -> list[str]:
    exprs = []
    for feat in TRENDING_FEATURES:
        for w in WINDOWS:
            exprs.append(
                f"{feat} - LAG({feat}, {w}) OVER "
                f"(PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) AS {feat}_DELTA_{w}H"
            )
    return exprs


def _rolling_exprs() -> list[str]:
    exprs = []
    for feat in TRENDING_FEATURES:
        for w in WINDOWS:
            frame = f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            exprs += [
                f"AVG({feat}) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT {frame}) AS {feat}_MEAN_{w}H",
                f"MAX({feat}) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT {frame}) AS {feat}_MAX_{w}H",
                f"MIN({feat}) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT {frame}) AS {feat}_MIN_{w}H",
            ]
    return exprs


def run_duckdb_features(
    df: pd.DataFrame,
    medications_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Blocks A–D verbatim from feature_engineering.py, running on an in-memory
    pandas DataFrame registered as a DuckDB view.

    Block D (vasopressor context) requires medications_df with columns
    ENCOUNTER_ID, MEDICATION_CD, ADMIN_START_DT_TM, ADMIN_END_DT_TM, and
    an EVENT_DT_TM column in df. Falls back to zeros when data is missing.
    """
    con = duckdb.connect(":memory:")
    con.register("base", df)

    # Block A — deltas
    con.execute(
        "CREATE OR REPLACE VIEW block_a AS SELECT *, "
        + ", ".join(_delta_exprs())
        + " FROM base"
    )

    # Block B — rolling statistics
    con.execute(
        "CREATE OR REPLACE VIEW block_b AS SELECT *, "
        + ", ".join(_rolling_exprs())
        + " FROM block_a"
    )

    # Block C step 1 — tag stage-change rows
    con.execute("""
        CREATE OR REPLACE VIEW stage_flags AS
        SELECT *,
            CASE WHEN SCAI_STAGE_NUM != LAG(SCAI_STAGE_NUM, 1) OVER (
                     PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT)
                 THEN 1 ELSE 0 END AS _IS_STAGE_CHANGE
        FROM block_b
    """)

    # Block C step 2 — cumulative run ID per encounter
    con.execute("""
        CREATE OR REPLACE VIEW stage_changes AS
        SELECT * EXCLUDE (_IS_STAGE_CHANGE),
            SUM(_IS_STAGE_CHANGE) OVER (
                PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT
            ) AS STAGE_RUN_ID
        FROM stage_flags
    """)

    # Block C — 9 derived clinical scores
    con.execute("""
        CREATE OR REPLACE VIEW block_c AS
        SELECT
            * EXCLUDE (STAGE_RUN_ID),

            HR / NULLIF(SBP, 0) AS SHOCK_INDEX,
            SBP - DBP           AS PULSE_PRESSURE,
            MAP / NULLIF(SBP, 0) AS MAP_SBP_RATIO,

            CASE WHEN LAG(LACTATE, 4) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) IS NULL
                   OR LAG(LACTATE, 4) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) = 0
                 THEN NULL
                 ELSE (LAG(LACTATE, 4) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) - LACTATE)
                      / LAG(LACTATE, 4) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) * 100.0
            END AS LACTATE_CLEARANCE_4H,

            CASE WHEN LAG(LACTATE, 24) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) IS NULL
                   OR LAG(LACTATE, 24) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) = 0
                 THEN NULL
                 ELSE (LAG(LACTATE, 24) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) - LACTATE)
                      / LAG(LACTATE, 24) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) * 100.0
            END AS LACTATE_CLEARANCE_24H,

            (HR / NULLIF(SBP, 0)) - LAG(HR / NULLIF(SBP, 0), 4) OVER (
                PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT
            ) AS SHOCK_INDEX_DELTA_4H,

            CASE WHEN LAG(SCAI_STAGE_NUM, 12) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) IS NULL
                 THEN NULL
                 WHEN SCAI_STAGE_NUM > LAG(SCAI_STAGE_NUM, 12) OVER (PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT)
                 THEN 1.0 ELSE 0.0
            END AS SCAI_WORSENING_12H,

            ROW_NUMBER() OVER (
                PARTITION BY ENCOUNTER_ID, STAGE_RUN_ID ORDER BY HOUR_FROM_ADMIT
            ) AS HOURS_AT_CURRENT_STAGE,

            URINE_OUT_HR / NULLIF(MAP, 0) AS URINE_SHOCK_RATIO
        FROM stage_changes
    """)

    # Block D — vasopressor context (timestamp-based)
    has_meds = (
        medications_df is not None
        and not medications_df.empty
        and "EVENT_DT_TM" in df.columns
    )

    if has_meds:
        con.register("medication_admin", medications_df)
        con.execute(f"""
            CREATE OR REPLACE VIEW vaso_events AS
            SELECT ENCOUNTER_ID, UPPER(MEDICATION_CD) AS MEDICATION_CD,
                   MIN(ADMIN_START_DT_TM) AS FIRST_START_DT_TM
            FROM medication_admin
            WHERE UPPER(MEDICATION_CD) IN ({_vaso_list})
            GROUP BY ENCOUNTER_ID, UPPER(MEDICATION_CD)
        """)
        con.execute("""
            CREATE OR REPLACE VIEW first_vaso AS
            SELECT ENCOUNTER_ID, MIN(FIRST_START_DT_TM) AS FIRST_VASO_DT_TM
            FROM vaso_events GROUP BY ENCOUNTER_ID
        """)
        con.execute("""
            CREATE OR REPLACE VIEW block_d AS
            SELECT c.*,
                CASE WHEN fv.FIRST_VASO_DT_TM IS NULL           THEN 0.0
                     WHEN fv.FIRST_VASO_DT_TM > c.EVENT_DT_TM   THEN 0.0
                     ELSE DATEDIFF('hour', fv.FIRST_VASO_DT_TM, c.EVENT_DT_TM)
                END AS HOURS_SINCE_FIRST_VASOPRESSOR,
                COALESCE((
                    SELECT COUNT(DISTINCT ve.MEDICATION_CD) FROM vaso_events ve
                    WHERE ve.ENCOUNTER_ID = c.ENCOUNTER_ID
                      AND ve.FIRST_START_DT_TM <= c.EVENT_DT_TM
                      AND ve.FIRST_START_DT_TM > c.EVENT_DT_TM - INTERVAL '4 hours'
                ), 0) AS VASOPRESSOR_ESCALATIONS_4H,
                COALESCE((
                    SELECT COUNT(DISTINCT ve.MEDICATION_CD) FROM vaso_events ve
                    WHERE ve.ENCOUNTER_ID = c.ENCOUNTER_ID
                      AND ve.FIRST_START_DT_TM <= c.EVENT_DT_TM
                      AND ve.FIRST_START_DT_TM > c.EVENT_DT_TM - INTERVAL '24 hours'
                ), 0) AS VASOPRESSOR_ESCALATIONS_24H,
                CASE WHEN EXISTS (
                    SELECT 1 FROM medication_admin ma
                    WHERE ma.ENCOUNTER_ID = c.ENCOUNTER_ID
                      AND UPPER(ma.MEDICATION_CD) IN (""" + _vaso_list + f""")
                      AND ma.ADMIN_START_DT_TM <= c.EVENT_DT_TM
                      AND (ma.ADMIN_END_DT_TM IS NULL OR ma.ADMIN_END_DT_TM >= c.EVENT_DT_TM)
                ) THEN 1.0 ELSE 0.0 END AS CURRENTLY_ON_VASOPRESSOR
            FROM block_c c LEFT JOIN first_vaso fv ON c.ENCOUNTER_ID = fv.ENCOUNTER_ID
        """)
        result = con.execute(
            "SELECT * FROM block_d ORDER BY ENCOUNTER_ID, HOUR_FROM_ADMIT"
        ).df()
    else:
        result = con.execute(
            "SELECT * FROM block_c ORDER BY ENCOUNTER_ID, HOUR_FROM_ADMIT"
        ).df()
        result["HOURS_SINCE_FIRST_VASOPRESSOR"] = 0.0
        result["VASOPRESSOR_ESCALATIONS_4H"]    = 0.0
        result["VASOPRESSOR_ESCALATIONS_24H"]   = 0.0
        # Use N_VASOPRESSORS_ACTIVE as proxy when medication timestamps unavailable
        if "N_VASOPRESSORS_ACTIVE" in result.columns:
            result["CURRENTLY_ON_VASOPRESSOR"] = (result["N_VASOPRESSORS_ACTIVE"] > 0).astype(float)
        else:
            result["CURRENTLY_ON_VASOPRESSOR"] = 0.0

    return result


# ── V3 feature helpers ────────────────────────────────────────────────────────

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope over a rolling window."""
    def _slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 2:
            return 0.0
        x  = np.arange(n, dtype=float)
        xm, ym = x.mean(), y.mean()
        denom  = ((x - xm) ** 2).sum()
        return 0.0 if denom == 0 else ((x - xm) * (y - ym)).sum() / denom
    return series.rolling(window, min_periods=2).apply(_slope, raw=True)


def _consecutive_direction(series: pd.Series, direction: str) -> pd.Series:
    """Consecutive-hours count of monotone trend (falling or rising)."""
    delta    = series.diff()
    flag     = (delta < 0 if direction == "falling" else delta > 0).astype(int)
    streak   = (flag != flag.shift()).cumsum()
    return flag.groupby(streak).cumsum().fillna(0)


# ── Step 8 (v3) — Additional feature blocks ───────────────────────────────────

def add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute v3 features not present in feature_engineering.py Blocks A–D.

    VERIFY: thresholds and the COMPOSITE_SHOCK_SCORE formula must be confirmed
    against Christie's v3 feature_engineering.py before production deployment.
    """
    enc = "ENCOUNTER_ID"
    df  = df.sort_values([enc, "HOUR_FROM_ADMIT"]).copy()

    # ── HOURS_SINCE_LAST_REAL for 6 hemodynamic sparse features ──────────────
    for feat in HEMO_SPARSE:
        meas  = f"was_measured_{feat}"
        out   = f"{feat}_HOURS_SINCE_LAST_REAL"
        hours = df["HOUR_FROM_ADMIT"].values

        if meas not in df.columns:
            df[out] = 0.0
            continue

        measured = df[meas].values
        result   = np.zeros(len(df), dtype=np.float64)
        enc_ids  = df[enc].values
        last_h: dict[int, float] = {}

        for i in range(len(df)):
            eid = enc_ids[i]
            if measured[i] == 1:
                last_h[eid] = hours[i]
                result[i]   = 0.0
            else:
                prev = last_h.get(eid)
                result[i] = (hours[i] - prev) if prev is not None else hours[i]

        df[out] = result

    # ── COMPOSITE_SHOCK_SCORE and its 4H rolling stats ────────────────────────
    # VERIFY: confirm this formula matches the v3 training pipeline
    si   = _col(df, "SHOCK_INDEX",    0.7)
    lac  = np.expm1(_col(df, "LACTATE", 0.0).clip(0))   # revert log1p for threshold
    scai = _col(df, "SCAI_STAGE_NUM", 0.0)
    vis  = np.expm1(_col(df, "VIS",    0.0).clip(0))

    df["COMPOSITE_SHOCK_SCORE"] = (
        si.fillna(0.7)
        + lac.fillna(1.0).clip(0, 20) / 10.0
        + scai.fillna(0) / 4.0
        + vis.fillna(0).clip(0, 200) / 100.0
    )
    df["SHOCK_SCORE_MEAN_4H"] = df.groupby(enc)["COMPOSITE_SHOCK_SCORE"].transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )
    df["SHOCK_SCORE_MAX_4H"] = df.groupby(enc)["COMPOSITE_SHOCK_SCORE"].transform(
        lambda x: x.rolling(4, min_periods=1).max()
    )
    df["SHOCK_SCORE_DELTA_4H"] = (
        df["COMPOSITE_SHOCK_SCORE"]
        - df.groupby(enc)["COMPOSITE_SHOCK_SCORE"].transform(lambda x: x.shift(4))
    ).fillna(0.0)

    # ── Stage interaction flags (use natural-unit values for clinical thresholds)
    map_nat   = _col(df, "MAP",           70.0)    # MAP is NOT log-transformed
    sbp_nat   = _col(df, "SBP",          120.0)
    hr_nat    = _col(df, "HR",            80.0)
    co_nat    = _col(df, "CO",             4.0)
    cpo_nat   = _col(df, "CPO",            0.6)
    ci_nat    = _col(df, "CI",             2.5)
    urine_nat = _col(df, "URINE_OUT_HR",  50.0)
    map_d1    = _col(df, "MAP_DELTA_1H",   0.0)
    lac_d1    = _col(df, "LACTATE_DELTA_1H", 0.0)

    df["STAGE_B_MAP_FALLING"]    = ((scai >= 2) & (map_d1.fillna(0) < 0)).astype(np.float32)
    df["STAGE_B_LACTATE_RISING"] = ((scai >= 2) & (lac_d1.fillna(0) > 0)).astype(np.float32)
    df["STAGE_C_HIGH_VIS"]       = ((scai >= 3) & (vis.fillna(0) > 20)).astype(np.float32)
    df["LOW_MAP_LOW_CO"]         = ((map_nat.fillna(70) < 65) & (co_nat.fillna(4) < 4.0)).astype(np.float32)
    df["TACHYCARDIA_HYPOTENSION"] = ((hr_nat.fillna(80) > 100) & (sbp_nat.fillna(120) < 90)).astype(np.float32)
    df["LACTATE_HIGH_URINE_LOW"] = ((lac.fillna(1) > 2.0) & (urine_nat.fillna(50) < 30)).astype(np.float32)
    df["MULTI_ORGAN_STRESS"]     = (
        (map_nat.fillna(70) < 65).astype(int)
        + (lac.fillna(1) > 2.0).astype(int)
        + (urine_nat.fillna(50) < 30).astype(int)
        + (scai >= 2).astype(int)
    ).astype(np.float32)
    df["STAGE_MISMATCH_VIS"]     = ((vis.fillna(0) > 30) & (scai < 2)).astype(np.float32)
    df["CPO_CI_RATIO"]           = (cpo_nat.fillna(0.6) / ci_nat.fillna(2.5).clip(lower=0.1)).astype(np.float32)
    df["SHOCK_INDEX_STAGE"]      = (si.fillna(0.7) * scai.fillna(0)).astype(np.float32)

    # ── Rolling 4H counts of stage interaction flags ──────────────────────────
    for flag in [
        "STAGE_B_MAP_FALLING", "STAGE_B_LACTATE_RISING", "STAGE_C_HIGH_VIS",
        "LOW_MAP_LOW_CO", "TACHYCARDIA_HYPOTENSION", "LACTATE_HIGH_URINE_LOW",
        "MULTI_ORGAN_STRESS",
    ]:
        df[f"{flag}_4H_COUNT"] = df.groupby(enc)[flag].transform(
            lambda x: x.rolling(4, min_periods=1).sum()
        )

    # ── Journey / persistence features (expanding-window across stay) ─────────
    df["WORST_MAP_THIS_STAY"]     = df.groupby(enc)["MAP"].transform(lambda x: x.expanding().min()).fillna(70)
    df["WORST_LACTATE_THIS_STAY"] = df.groupby(enc)["LACTATE"].transform(lambda x: x.expanding().max()).fillna(0)
    df["WORST_SCAI_THIS_STAY"]    = df.groupby(enc)["SCAI_STAGE_NUM"].transform(lambda x: x.expanding().max()).fillna(0)
    df["HAS_EVER_BEEN_STAGE_C"]   = (df["WORST_SCAI_THIS_STAY"] >= 2).astype(np.float32)
    df["HAS_EVER_BEEN_STAGE_D"]   = (df["WORST_SCAI_THIS_STAY"] >= 3).astype(np.float32)

    scai_d1 = _col(df, "SCAI_STAGE_NUM_DELTA_1H", 0.0).fillna(0)
    df["TIMES_STAGE_WORSENED"] = (scai_d1 > 0).astype(int).groupby(df[enc]).cumsum()
    df["TIMES_STAGE_IMPROVED"] = (scai_d1 < 0).astype(int).groupby(df[enc]).cumsum()
    df["HOURS_SPENT_STAGE_C_OR_WORSE"] = (
        (df["SCAI_STAGE_NUM"] >= 2).astype(int).groupby(df[enc]).cumsum()
    )

    # ── Trend persistence ─────────────────────────────────────────────────────
    df["MAP_TREND_SLOPE_12H"] = df.groupby(enc)["MAP"].transform(
        lambda x: _rolling_slope(x, 12)
    ).fillna(0.0)
    df["LACTATE_TREND_SLOPE_12H"] = df.groupby(enc)["LACTATE"].transform(
        lambda x: _rolling_slope(x, 12)
    ).fillna(0.0)
    df["MAP_CONSECUTIVE_HOURS_FALLING"] = df.groupby(enc)["MAP"].transform(
        lambda x: _consecutive_direction(x, "falling")
    ).fillna(0.0)
    df["LACTATE_CONSECUTIVE_HOURS_RISING"] = df.groupby(enc)["LACTATE"].transform(
        lambda x: _consecutive_direction(x, "rising")
    ).fillna(0.0)
    df["HR_CONSECUTIVE_HOURS_RISING"] = df.groupby(enc)["HR"].transform(
        lambda x: _consecutive_direction(x, "rising")
    ).fillna(0.0)

    # ── Fraction features ─────────────────────────────────────────────────────
    df["MAP_FALLING_FRACTION_12H"] = df.groupby(enc)["MAP"].transform(
        lambda x: x.diff().lt(0).rolling(12, min_periods=1).mean()
    ).fillna(0.0)
    df["LACTATE_RISING_FRACTION_12H"] = df.groupby(enc)["LACTATE"].transform(
        lambda x: x.diff().gt(0).rolling(12, min_periods=1).mean()
    ).fillna(0.0)

    # ── Lab cadence features ──────────────────────────────────────────────────
    all_meas   = [f"was_measured_{f}" for f in SPARSE_FEATURES if f"was_measured_{f}" in df.columns]
    hemo_meas  = [f"was_measured_{f}" for f in HEMO_SPARSE       if f"was_measured_{f}" in df.columns]
    lac_meas   = "was_measured_LACTATE"
    crit_meas  = [c for c in ["was_measured_LACTATE", "was_measured_TROPONIN_I"] if c in df.columns]

    if all_meas:
        any_lab = df[all_meas].max(axis=1)
        df["LAB_ORDER_FREQUENCY_4H"]  = any_lab.groupby(df[enc]).transform(
            lambda x: x.rolling(4,  min_periods=1).sum()
        )
        df["LAB_ORDER_FREQUENCY_12H"] = any_lab.groupby(df[enc]).transform(
            lambda x: x.rolling(12, min_periods=1).sum()
        )
    else:
        df["LAB_ORDER_FREQUENCY_4H"]  = 0.0
        df["LAB_ORDER_FREQUENCY_12H"] = 0.0

    df["HEMO_MONITORING_ACTIVE"] = (
        df[hemo_meas].sum(axis=1) > 0 if hemo_meas else pd.Series(0.0, index=df.index)
    ).astype(np.float32)

    df["LAB_CADENCE_INCREASING"] = (
        df["LAB_ORDER_FREQUENCY_4H"] > (df["LAB_ORDER_FREQUENCY_12H"] / 3.0)
    ).astype(np.float32)

    if lac_meas in df.columns:
        df["LACTATE_ORDER_FREQUENCY_12H"] = df.groupby(enc)[lac_meas].transform(
            lambda x: x.rolling(12, min_periods=1).sum()
        )
    else:
        df["LACTATE_ORDER_FREQUENCY_12H"] = 0.0

    if crit_meas:
        any_crit = df[crit_meas].max(axis=1)
        df["CRITICAL_LAB_DRAWN_LAST_4H"] = any_crit.groupby(df[enc]).transform(
            lambda x: x.rolling(4, min_periods=1).max()
        )
    else:
        df["CRITICAL_LAB_DRAWN_LAST_4H"] = 0.0

    return df


# ── Main pipeline entry point ─────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    imputer_state: dict,
    iqr_bounds: dict,
    log_features: list[str],
    medications_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run the full vasopressor inference pipeline (steps 3–9 + v3 features).

    Args:
        df:             Wide DataFrame, one row per (ENCOUNTER_ID, HOUR_FROM_ADMIT).
                        Required columns: ENCOUNTER_ID, HOUR_FROM_ADMIT, SCAI_STAGE_NUM.
                        Optional: N_VASOPRESSORS_ACTIVE, EVENT_DT_TM, any of ALL_FEATURES.
        imputer_state:  From knn_imputer.pkl
        iqr_bounds:     From iqr_bounds.json
        log_features:   From log_transform_features.json
        medications_df: Optional — ENCOUNTER_ID, MEDICATION_CD, ADMIN_START_DT_TM,
                        ADMIN_END_DT_TM (enables full Block D vasopressor context).

    Returns:
        DataFrame with all engineered features. Feature selection
        (to the model's exact 360 columns) is done by the caller using
        feature_manifest["feature_columns"].
    """
    df = apply_clinical_ranges(df)
    df = add_was_measured_flags(df)
    df = apply_locf(df)
    df = apply_knn_imputation_inference(df, imputer_state)
    df = apply_iqr_capping_inference(df, iqr_bounds)
    df = apply_log_transforms_inference(df, log_features)
    df = run_duckdb_features(df, medications_df)
    df = add_v3_features(df)
    return df
