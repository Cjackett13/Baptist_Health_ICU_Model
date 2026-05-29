"""
SCAI inference pipeline — extracted from unified_enhanced.py.

Contains only the constants and feature engineering functions needed at
inference time. No training code, no data loading, no file I/O.

Source: unified_enhanced.py lines 35-36, 54-70, 115-259.
"""

import numpy as np
import pandas as pd

# ── Column name constants (must match training data schema) ───────────────────
ENC_COL   = "ENCOUNTER_ID"
STAGE_COL = "SCAI_STAGE_NUM"
MIN_STD   = 1e-3

# ── Features extended with 12h / 24h rolling lookback ────────────────────────
LOOKBACK_BASE = [
    "VIS_mean4h",        "VIS_delta4h",
    "MAP_mean4h",        "MAP_delta4h",
    "LACTATE_mean4h",    "LACTATE_delta4h",
    "CREATININE_mean4h", "CREATININE_delta4h",
    "CVP_mean4h",        "CVP_delta4h",
    "HR_mean4h",         "HR_delta4h",
    "SBP_mean4h",
]

# ── VIS features normalised within-stage ─────────────────────────────────────
VIS_FEATURES = [
    "VIS_mean4h", "VIS_delta4h", "VIS_slope4h",
    "VIS_max4h",  "VIS_min4h",
    "ENOXIMONE_DOSE", "MILRINONE_DOSE",
    "VASOPRESSIN_DOSE",
]


# ── Feature engineering functions (verbatim from unified_enhanced.py) ─────────

def build_trajectory_features(df_main: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Four features derived from hourly SCAI stage trajectory."""
    h = hourly_df.sort_values([ENC_COL, "HOUR_FROM_ADMIT"]).copy()

    rows = []
    for enc_id, grp in h.groupby(ENC_COL):
        stages = grp["SCAI_STAGE_NUM"].values
        hours  = grp["HOUR_FROM_ADMIT"].values
        n      = len(stages)

        consec_d     = np.zeros(n, dtype=float)
        hours_in_run = np.zeros(n, dtype=float)
        prior_max    = np.zeros(n, dtype=float)
        changes_24h  = np.zeros(n, dtype=float)

        run_len = 0

        for i in range(n):
            if stages[i] == 3:
                run_len += 1
            else:
                run_len = 0
            consec_d[i] = run_len

            if i == 0:
                hours_in_run[i] = 0.0
            elif stages[i] != stages[i - 1]:
                hours_in_run[i] = 0.0
            else:
                hours_in_run[i] = hours[i] - hours[i - 1] + hours_in_run[i - 1]

            if i == 0:
                prior_max[i] = stages[0]
            else:
                prior_max[i] = float(np.max(stages[:i]))

            if i == 0:
                changes_24h[i] = 0.0
            else:
                window_mask = (hours >= hours[i] - 24) & (hours < hours[i])
                w_stages    = stages[window_mask]
                changes_24h[i] = float(np.sum(np.diff(w_stages) != 0)) if len(w_stages) > 1 else 0.0

        for i in range(n):
            rows.append({
                ENC_COL:                      enc_id,
                "HOUR_FROM_ADMIT":            hours[i],
                "TRAJ_consec_stage_d_hours":  consec_d[i],
                "TRAJ_hours_in_current_stage": hours_in_run[i],
                "TRAJ_prior_max_stage":        prior_max[i],
                "TRAJ_stage_changes_24h":      changes_24h[i],
            })

    traj_df  = pd.DataFrame(rows)
    traj_cols = [
        "TRAJ_consec_stage_d_hours", "TRAJ_hours_in_current_stage",
        "TRAJ_prior_max_stage",      "TRAJ_stage_changes_24h",
    ]
    merged = df_main.merge(traj_df, on=[ENC_COL, "HOUR_FROM_ADMIT"], how="left")
    merged[traj_cols] = merged[traj_cols].fillna(0.0)
    return merged


def add_extended_lookback(df: pd.DataFrame, base_features: list) -> pd.DataFrame:
    """Add 12h and 24h rolling mean + delta columns for each base feature."""
    out     = df.copy()
    present = [f for f in base_features if f in out.columns]

    for feat in present:
        for w in (12, 24):
            col_mean  = f"{feat}_mean{w}h"
            col_delta = f"{feat}_delta{w}h"

            out[col_mean] = (
                out.groupby(ENC_COL)[feat]
                   .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            prior_mean = (
                out.groupby(ENC_COL)[feat]
                   .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            out[col_delta] = (out[feat] - prior_mean).fillna(0.0)

    return out


def apply_vis_normalisation(df: pd.DataFrame, stats: dict, features: list) -> pd.DataFrame:
    """Z-score VIS features within each SCAI stage using pre-computed stats."""
    out = df.copy()
    for stage, sdict in stats.items():
        mask = out[STAGE_COL] == stage
        if mask.sum() == 0:
            continue
        for feat in features:
            if feat not in sdict or feat not in out.columns:
                continue
            mean = sdict[feat]["mean"]
            std  = sdict[feat]["std"]
            ok   = sdict[feat].get("normalizable", std >= MIN_STD)
            if not ok:
                continue
            out[feat] = out[feat].astype(np.float64)
            out.loc[mask, feat] = (out.loc[mask, feat] - mean) / std
    return out


def build_features(
    df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    norm_stats_dict: dict | None = None,
) -> pd.DataFrame:
    """Run feature engineering steps.  Pass norm_stats_dict=None to skip VIS normalisation
    (required for scai_deterioration_model and scai_stage_d_model — those were trained on
    raw feature values with no z-scoring)."""
    df = build_trajectory_features(df, hourly_df)
    df = add_extended_lookback(df, LOOKBACK_BASE)
    if norm_stats_dict:
        df = apply_vis_normalisation(df, norm_stats_dict, VIS_FEATURES)
    return df
