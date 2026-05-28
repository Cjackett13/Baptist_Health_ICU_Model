"""
SCAI feature extractor — Christie's model 1 (xgboost_unified_final.pkl, 292 features).

Feature engineering must replicate unified_enhanced.py exactly:
  1. Within-stage VIS normalization (from iqr_fence_stats or norm_stats)
  2. 12h/24h extended lookback rolling stats
  3. Four TRAJ trajectory features from scai_stage_hourly
  4. Fill all delta/slope NaN with 0

TODO (Christie): Copy unified_enhanced.py into this directory and replace the
body of extract() with a call to its build_features() / compute_features() function.

The final DataFrame returned by extract() must have exactly the columns in
bundle["feature_cols"] (loaded by loader.py), in that order.
"""

import numpy as np
import pandas as pd

# 211 base features (from scai_deterioration_model training).
# xgboost_unified_final.pkl extends these with TRAJ features + extended lookbacks
# to reach 292 total. Use bundle["feature_cols"] at runtime for the exact list.
SCAI_BASE_FEATURES: list[str] = [
    'AGE', 'ALT', 'ALT_delta4h', 'ALT_max4h', 'ALT_mean4h', 'ALT_min4h', 'ALT_slope4h',
    'AST', 'AST_delta4h', 'AST_max4h', 'AST_mean4h', 'AST_min4h', 'AST_slope4h',
    'BUN', 'BUN_delta4h', 'BUN_max4h', 'BUN_mean4h', 'BUN_min4h', 'BUN_slope4h',
    'CI', 'CI_delta4h', 'CI_max4h', 'CI_mean4h', 'CI_min4h', 'CI_slope4h',
    'CO', 'CO_delta4h', 'CO_max4h', 'CO_mean4h', 'CO_min4h', 'CO_slope4h',
    'CPO', 'CPO_delta4h', 'CPO_max4h', 'CPO_mean4h', 'CPO_min4h', 'CPO_slope4h',
    'CREATININE', 'CREATININE_delta4h', 'CREATININE_max4h', 'CREATININE_mean4h',
    'CREATININE_min4h', 'CREATININE_slope4h',
    'CVP', 'CVP_delta4h', 'CVP_max4h', 'CVP_mean4h', 'CVP_min4h', 'CVP_slope4h',
    'DBP', 'DBP_delta4h', 'DBP_max4h', 'DBP_mean4h', 'DBP_min4h', 'DBP_slope4h',
    'HCO3', 'HCO3_delta4h', 'HCO3_max4h', 'HCO3_mean4h', 'HCO3_min4h', 'HCO3_slope4h',
    'HGB', 'HGB_delta4h', 'HGB_max4h', 'HGB_mean4h', 'HGB_min4h', 'HGB_slope4h',
    'HR', 'HR_delta4h', 'HR_max4h', 'HR_mean4h', 'HR_min4h', 'HR_slope4h',
    'INR', 'INR_delta4h', 'INR_max4h', 'INR_mean4h', 'INR_min4h', 'INR_slope4h',
    'LACTATE', 'LACTATE_delta4h', 'LACTATE_max4h', 'LACTATE_mean4h', 'LACTATE_min4h',
    'LACTATE_slope4h',
    'MAP', 'MAP_delta4h', 'MAP_max4h', 'MAP_mean4h', 'MAP_min4h', 'MAP_slope4h',
    'NT_PROBNP', 'NT_PROBNP_delta4h', 'NT_PROBNP_max4h', 'NT_PROBNP_mean4h',
    'NT_PROBNP_min4h', 'NT_PROBNP_slope4h',
    'N_INOTROPES_ACTIVE', 'N_VASOPRESSORS_ACTIVE', 'ON_CRRT', 'ON_INTUBATION', 'ON_MCS',
    'PAD', 'PAD_delta4h', 'PAD_max4h', 'PAD_mean4h', 'PAD_min4h', 'PAD_slope4h',
    'PAM', 'PAM_delta4h', 'PAM_max4h', 'PAM_mean4h', 'PAM_min4h', 'PAM_slope4h',
    'PAPI', 'PAPI_delta4h', 'PAPI_max4h', 'PAPI_mean4h', 'PAPI_min4h', 'PAPI_slope4h',
    'PAS', 'PAS_delta4h', 'PAS_max4h', 'PAS_mean4h', 'PAS_min4h', 'PAS_slope4h',
    'PCO2', 'PCO2_delta4h', 'PCO2_max4h', 'PCO2_mean4h', 'PCO2_min4h', 'PCO2_slope4h',
    'PCWP', 'PCWP_delta4h', 'PCWP_max4h', 'PCWP_mean4h', 'PCWP_min4h', 'PCWP_slope4h',
    'PH', 'PH_delta4h', 'PH_max4h', 'PH_mean4h', 'PH_min4h', 'PH_slope4h',
    'PLT', 'PLT_delta4h', 'PLT_max4h', 'PLT_mean4h', 'PLT_min4h', 'PLT_slope4h',
    'RR', 'RR_delta4h', 'RR_max4h', 'RR_mean4h', 'RR_min4h', 'RR_slope4h',
    'SBP', 'SBP_delta4h', 'SBP_max4h', 'SBP_mean4h', 'SBP_min4h', 'SBP_slope4h',
    'SCAI_STAGE_NUM',
    'SPO2', 'SPO2_delta4h', 'SPO2_max4h', 'SPO2_mean4h', 'SPO2_min4h', 'SPO2_slope4h',
    'SVO2', 'SVO2_delta4h', 'SVO2_max4h', 'SVO2_mean4h', 'SVO2_min4h', 'SVO2_slope4h',
    'SVR', 'SVR_delta4h', 'SVR_max4h', 'SVR_mean4h', 'SVR_min4h', 'SVR_slope4h',
    'TEMP', 'TEMP_delta4h', 'TEMP_max4h', 'TEMP_mean4h', 'TEMP_min4h', 'TEMP_slope4h',
    'TROPONIN_I', 'TROPONIN_I_delta4h', 'TROPONIN_I_max4h', 'TROPONIN_I_mean4h',
    'TROPONIN_I_min4h', 'TROPONIN_I_slope4h',
    'URINE_OUT_HR', 'URINE_OUT_HR_delta4h', 'URINE_OUT_HR_max4h', 'URINE_OUT_HR_mean4h',
    'URINE_OUT_HR_min4h', 'URINE_OUT_HR_slope4h',
    'VIS', 'VIS_delta4h', 'VIS_max4h', 'VIS_mean4h', 'VIS_min4h', 'VIS_slope4h',
    'WBC', 'WBC_delta4h', 'WBC_max4h', 'WBC_mean4h', 'WBC_min4h', 'WBC_slope4h',
]

SCAI_TRAJ_FEATURES: list[str] = [
    'TRAJ_hours_in_current_stage',
    'TRAJ_prior_max_stage',
    'TRAJ_stage_changes_24h',
]


def extract(raw_patient_data: dict, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Build a feature DataFrame for xgboost_unified_final.pkl.

    raw_patient_data must contain:
      "df"                — pd.DataFrame, one row per scoring hour with base clinical
                            features (same column format as train_unscaled.parquet).
                            Must include ENCOUNTER_ID and HOUR_FROM_ADMIT columns.
      "scai_stage_hourly" — pd.DataFrame, full hourly SCAI stage history for the
                            encounter (same format as scai_stage_hourly.parquet).

    norm_stats and feature_cols are pulled from the loaded bundle — no extra files needed.
    """
    from app.models.features.scai_pipeline import build_features
    from app.models import loader

    bundle          = loader.get("scai")
    norm_stats_dict = bundle["vis_normalization_stats"]

    df_main:   pd.DataFrame = raw_patient_data["df"]
    hourly_df: pd.DataFrame = raw_patient_data["scai_stage_hourly"]

    df = build_features(df_main, hourly_df, norm_stats_dict)

    # Fill all delta / slope NaN with 0 (required by unified_enhanced.py pipeline)
    nan_cols = [c for c in df.columns if "delta" in c.lower() or "slope" in c.lower()]
    df[nan_cols] = df[nan_cols].fillna(0)

    if feature_cols is None:
        feature_cols = bundle["feature_cols"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=missing)],
            axis=1,
        )

    return df[feature_cols].astype(np.float32)
