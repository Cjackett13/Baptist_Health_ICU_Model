"""
Vasopressor feature extractor — Christie's models 3/4/5 (360 features shared by all three).

All three vasopressor pkl files (binary_alert_model, ordinal_count_model,
high_severity_classifier) use the same 360-feature input.

TODO (Christie): Replace the NotImplementedError below with the feature
engineering pipeline from your vasopressor training repo, e.g.:
    from app.models.features.vasopressor_pipeline import build_feature_row
    return build_feature_row(raw_patient_data)[VASOPRESSOR_FEATURES]
"""

import pandas as pd

VASOPRESSOR_FEATURES: list[str] = [
    'CURRENTLY_ON_VASOPRESSOR',
    'BUN', 'BUN_DELTA_1H', 'BUN_DELTA_24H', 'BUN_DELTA_4H',
    'BUN_MAX_1H', 'BUN_MAX_24H', 'BUN_MAX_4H',
    'BUN_MEAN_1H', 'BUN_MEAN_24H', 'BUN_MEAN_4H',
    'BUN_MIN_1H', 'BUN_MIN_24H', 'BUN_MIN_4H',
    'CI_DELTA_24H', 'CI_MAX_24H', 'CI_MEAN_24H', 'CI_MIN_24H', 'CI_MIN_4H',
    'CO_MAX_24H', 'CO_MAX_4H', 'CO_MEAN_24H', 'CO_MIN_24H', 'CO_MIN_4H',
    'CPO_MAX_24H', 'CPO_MAX_4H', 'CPO_MEAN_24H', 'CPO_MIN_24H',
    'CREATININE', 'CREATININE_DELTA_1H', 'CREATININE_DELTA_24H', 'CREATININE_DELTA_4H',
    'CREATININE_MAX_1H', 'CREATININE_MAX_24H', 'CREATININE_MAX_4H',
    'CREATININE_MEAN_1H', 'CREATININE_MEAN_24H', 'CREATININE_MEAN_4H',
    'CREATININE_MIN_1H', 'CREATININE_MIN_24H', 'CREATININE_MIN_4H',
    'CVP_DELTA_24H', 'CVP_MAX_24H', 'CVP_MAX_4H', 'CVP_MEAN_24H', 'CVP_MEAN_4H',
    'CVP_MIN_24H', 'CVP_MIN_4H',
    'DBP', 'DBP_DELTA_1H', 'DBP_DELTA_24H', 'DBP_DELTA_4H',
    'DBP_MAX_1H', 'DBP_MAX_24H', 'DBP_MAX_4H',
    'DBP_MEAN_1H', 'DBP_MEAN_24H', 'DBP_MEAN_4H',
    'DBP_MIN_1H', 'DBP_MIN_24H', 'DBP_MIN_4H',
    'HCO3', 'HCO3_DELTA_1H', 'HCO3_DELTA_24H', 'HCO3_DELTA_4H',
    'HCO3_MAX_1H', 'HCO3_MAX_24H', 'HCO3_MAX_4H',
    'HCO3_MEAN_1H', 'HCO3_MEAN_24H', 'HCO3_MEAN_4H',
    'HCO3_MIN_1H', 'HCO3_MIN_24H', 'HCO3_MIN_4H',
    'HGB', 'HGB_DELTA_1H', 'HGB_DELTA_24H', 'HGB_DELTA_4H',
    'HGB_MAX_1H', 'HGB_MAX_24H', 'HGB_MAX_4H',
    'HGB_MEAN_1H', 'HGB_MEAN_24H', 'HGB_MEAN_4H',
    'HGB_MIN_1H', 'HGB_MIN_24H', 'HGB_MIN_4H',
    'HOURS_AT_CURRENT_STAGE', 'HOURS_SINCE_FIRST_VASOPRESSOR',
    'HR', 'HR_DELTA_1H', 'HR_DELTA_24H', 'HR_DELTA_4H',
    'HR_MAX_1H', 'HR_MAX_24H', 'HR_MAX_4H',
    'HR_MEAN_1H', 'HR_MEAN_24H', 'HR_MEAN_4H',
    'HR_MIN_1H', 'HR_MIN_24H', 'HR_MIN_4H',
    'LACTATE', 'LACTATE_CLEARANCE_24H', 'LACTATE_CLEARANCE_4H',
    'LACTATE_DELTA_1H', 'LACTATE_DELTA_24H', 'LACTATE_DELTA_4H',
    'LACTATE_MAX_1H', 'LACTATE_MAX_24H', 'LACTATE_MAX_4H',
    'LACTATE_MEAN_1H', 'LACTATE_MEAN_24H', 'LACTATE_MEAN_4H',
    'LACTATE_MIN_1H', 'LACTATE_MIN_24H', 'LACTATE_MIN_4H',
    'MAP', 'MAP_DELTA_1H', 'MAP_DELTA_24H', 'MAP_DELTA_4H',
    'MAP_MAX_1H', 'MAP_MAX_24H', 'MAP_MAX_4H',
    'MAP_MEAN_1H', 'MAP_MEAN_24H', 'MAP_MEAN_4H',
    'MAP_MIN_1H', 'MAP_MIN_24H', 'MAP_MIN_4H', 'MAP_SBP_RATIO',
    'NT_PROBNP', 'NT_PROBNP_DELTA_1H', 'NT_PROBNP_DELTA_24H', 'NT_PROBNP_DELTA_4H',
    'NT_PROBNP_MAX_1H', 'NT_PROBNP_MAX_24H', 'NT_PROBNP_MAX_4H',
    'NT_PROBNP_MEAN_1H', 'NT_PROBNP_MEAN_24H', 'NT_PROBNP_MEAN_4H',
    'NT_PROBNP_MIN_1H', 'NT_PROBNP_MIN_24H', 'NT_PROBNP_MIN_4H',
    'PAPI_DELTA_24H', 'PAPI_MAX_24H', 'PAPI_MAX_4H', 'PAPI_MEAN_24H',
    'PAPI_MIN_24H', 'PAPI_MIN_4H',
    'PH', 'PH_DELTA_1H', 'PH_DELTA_24H', 'PH_DELTA_4H',
    'PH_MAX_1H', 'PH_MAX_24H', 'PH_MAX_4H',
    'PH_MEAN_1H', 'PH_MEAN_24H', 'PH_MEAN_4H',
    'PH_MIN_1H', 'PH_MIN_24H', 'PH_MIN_4H',
    'PLT', 'PLT_DELTA_1H', 'PLT_DELTA_24H', 'PLT_DELTA_4H',
    'PLT_MAX_1H', 'PLT_MAX_24H', 'PLT_MAX_4H',
    'PLT_MEAN_1H', 'PLT_MEAN_24H', 'PLT_MEAN_4H',
    'PLT_MIN_1H', 'PLT_MIN_24H', 'PLT_MIN_4H',
    'PULSE_PRESSURE',
    'RR', 'RR_DELTA_1H', 'RR_DELTA_24H', 'RR_DELTA_4H',
    'RR_MAX_1H', 'RR_MAX_24H', 'RR_MAX_4H',
    'RR_MEAN_1H', 'RR_MEAN_24H', 'RR_MEAN_4H',
    'RR_MIN_1H', 'RR_MIN_24H', 'RR_MIN_4H',
    'SBP', 'SBP_DELTA_1H', 'SBP_DELTA_24H', 'SBP_DELTA_4H',
    'SBP_MAX_1H', 'SBP_MAX_24H', 'SBP_MAX_4H',
    'SBP_MEAN_1H', 'SBP_MEAN_24H', 'SBP_MEAN_4H',
    'SBP_MIN_1H', 'SBP_MIN_24H', 'SBP_MIN_4H',
    'SCAI_STAGE_NUM', 'SCAI_STAGE_NUM_DELTA_1H', 'SCAI_STAGE_NUM_DELTA_24H',
    'SCAI_STAGE_NUM_DELTA_4H', 'SCAI_STAGE_NUM_MAX_1H', 'SCAI_STAGE_NUM_MAX_24H',
    'SCAI_STAGE_NUM_MAX_4H', 'SCAI_STAGE_NUM_MEAN_1H', 'SCAI_STAGE_NUM_MEAN_24H',
    'SCAI_STAGE_NUM_MEAN_4H', 'SCAI_STAGE_NUM_MIN_1H', 'SCAI_STAGE_NUM_MIN_24H',
    'SCAI_STAGE_NUM_MIN_4H', 'SCAI_WORSENING_12H',
    'SHOCK_INDEX', 'SHOCK_INDEX_DELTA_4H',
    'SPO2', 'SPO2_DELTA_1H', 'SPO2_DELTA_24H', 'SPO2_DELTA_4H',
    'SPO2_MAX_1H', 'SPO2_MAX_24H', 'SPO2_MAX_4H',
    'SPO2_MEAN_1H', 'SPO2_MEAN_24H', 'SPO2_MEAN_4H',
    'SPO2_MIN_1H', 'SPO2_MIN_24H', 'SPO2_MIN_4H',
    'SVO2_MAX_24H', 'SVO2_MAX_4H', 'SVO2_MEAN_24H', 'SVO2_MEAN_4H',
    'SVO2_MIN_24H', 'SVO2_MIN_4H',
    'TEMP', 'TEMP_DELTA_1H', 'TEMP_DELTA_24H', 'TEMP_DELTA_4H',
    'TEMP_MAX_1H', 'TEMP_MAX_24H', 'TEMP_MAX_4H',
    'TEMP_MEAN_1H', 'TEMP_MEAN_24H', 'TEMP_MEAN_4H',
    'TEMP_MIN_1H', 'TEMP_MIN_24H', 'TEMP_MIN_4H',
    'TROPONIN_I', 'TROPONIN_I_DELTA_1H', 'TROPONIN_I_DELTA_24H', 'TROPONIN_I_DELTA_4H',
    'TROPONIN_I_MAX_1H', 'TROPONIN_I_MAX_24H', 'TROPONIN_I_MAX_4H',
    'TROPONIN_I_MEAN_1H', 'TROPONIN_I_MEAN_24H', 'TROPONIN_I_MEAN_4H',
    'TROPONIN_I_MIN_1H', 'TROPONIN_I_MIN_24H', 'TROPONIN_I_MIN_4H',
    'URINE_OUT_HR', 'URINE_OUT_HR_DELTA_1H', 'URINE_OUT_HR_DELTA_24H', 'URINE_OUT_HR_DELTA_4H',
    'URINE_OUT_HR_MAX_1H', 'URINE_OUT_HR_MAX_24H', 'URINE_OUT_HR_MAX_4H',
    'URINE_OUT_HR_MEAN_1H', 'URINE_OUT_HR_MEAN_24H', 'URINE_OUT_HR_MEAN_4H',
    'URINE_OUT_HR_MIN_1H', 'URINE_OUT_HR_MIN_24H', 'URINE_OUT_HR_MIN_4H',
    'URINE_SHOCK_RATIO',
    'VASOPRESSOR_ESCALATIONS_24H', 'VASOPRESSOR_ESCALATIONS_4H',
    'VIS', 'VIS_DELTA_1H', 'VIS_DELTA_24H', 'VIS_DELTA_4H',
    'VIS_MAX_1H', 'VIS_MAX_24H', 'VIS_MAX_4H',
    'VIS_MEAN_1H', 'VIS_MEAN_24H', 'VIS_MEAN_4H',
    'VIS_MIN_1H', 'VIS_MIN_24H', 'VIS_MIN_4H',
    'WBC', 'WBC_DELTA_1H', 'WBC_DELTA_24H', 'WBC_DELTA_4H',
    'WBC_MAX_1H', 'WBC_MAX_24H', 'WBC_MAX_4H',
    'WBC_MEAN_1H', 'WBC_MEAN_24H', 'WBC_MEAN_4H',
    'WBC_MIN_1H', 'WBC_MIN_24H', 'WBC_MIN_4H',
    'was_measured_CI', 'was_measured_CO', 'was_measured_CPO',
    'was_measured_CVP', 'was_measured_PAPI', 'was_measured_SVO2',
    'CO_HOURS_SINCE_LAST_REAL', 'CI_HOURS_SINCE_LAST_REAL', 'CPO_HOURS_SINCE_LAST_REAL',
    'CVP_HOURS_SINCE_LAST_REAL', 'PAPI_HOURS_SINCE_LAST_REAL', 'SVO2_HOURS_SINCE_LAST_REAL',
    'COMPOSITE_SHOCK_SCORE', 'SHOCK_SCORE_MEAN_4H', 'SHOCK_SCORE_MAX_4H',
    'SHOCK_SCORE_DELTA_4H',
    'STAGE_B_MAP_FALLING', 'STAGE_B_LACTATE_RISING', 'STAGE_C_HIGH_VIS',
    'LOW_MAP_LOW_CO', 'TACHYCARDIA_HYPOTENSION', 'LACTATE_HIGH_URINE_LOW',
    'MULTI_ORGAN_STRESS', 'STAGE_MISMATCH_VIS', 'CPO_CI_RATIO', 'SHOCK_INDEX_STAGE',
    'STAGE_B_MAP_FALLING_4H_COUNT', 'STAGE_B_LACTATE_RISING_4H_COUNT',
    'STAGE_C_HIGH_VIS_4H_COUNT', 'LOW_MAP_LOW_CO_4H_COUNT',
    'TACHYCARDIA_HYPOTENSION_4H_COUNT', 'LACTATE_HIGH_URINE_LOW_4H_COUNT',
    'MULTI_ORGAN_STRESS_4H_COUNT',
    'WORST_MAP_THIS_STAY', 'WORST_LACTATE_THIS_STAY', 'WORST_SCAI_THIS_STAY',
    'HAS_EVER_BEEN_STAGE_C', 'HAS_EVER_BEEN_STAGE_D',
    'TIMES_STAGE_WORSENED', 'TIMES_STAGE_IMPROVED', 'HOURS_SPENT_STAGE_C_OR_WORSE',
    'MAP_TREND_SLOPE_12H', 'LACTATE_TREND_SLOPE_12H',
    'MAP_CONSECUTIVE_HOURS_FALLING', 'LACTATE_CONSECUTIVE_HOURS_RISING',
    'HR_CONSECUTIVE_HOURS_RISING',
    'MAP_FALLING_FRACTION_12H', 'LACTATE_RISING_FRACTION_12H',
    'LAB_ORDER_FREQUENCY_4H', 'LAB_ORDER_FREQUENCY_12H',
    'HEMO_MONITORING_ACTIVE', 'LAB_CADENCE_INCREASING',
    'LACTATE_ORDER_FREQUENCY_12H', 'CRITICAL_LAB_DRAWN_LAST_4H',
]


def extract(raw_patient_data: dict) -> pd.DataFrame:
    """
    Build a feature DataFrame for the three vasopressor models.

    raw_patient_data must contain:
      "df"          — pd.DataFrame, one row per (ENCOUNTER_ID, HOUR_FROM_ADMIT)
                      with base clinical features plus SCAI_STAGE_NUM.
                      Optional columns: N_VASOPRESSORS_ACTIVE, EVENT_DT_TM.
      "medications" — (optional) pd.DataFrame with ENCOUNTER_ID, MEDICATION_CD,
                      ADMIN_START_DT_TM, ADMIN_END_DT_TM for Block D features.

    Returns a DataFrame whose columns are exactly VASOPRESSOR_FEATURES (360).
    """
    import numpy as np
    from app.models.features.vasopressor_pipeline import build_features
    from app.models import loader

    prep = loader.get("vasopressor_prep")
    if prep is None:
        raise RuntimeError("Vasopressor preprocessing artifacts not loaded.")

    df_main       = raw_patient_data["df"]
    medications   = raw_patient_data.get("medications")

    result = build_features(
        df            = df_main,
        imputer_state = prep["knn_imputer"],
        iqr_bounds    = prep["iqr_bounds"],
        log_features  = prep["log_features"],
        medications_df = medications,
    )

    feature_cols = prep.get("feature_cols", VASOPRESSOR_FEATURES)
    missing = [c for c in feature_cols if c not in result.columns]
    if missing:
        for c in missing:
            result[c] = 0.0

    return result[feature_cols].astype(np.float32)
