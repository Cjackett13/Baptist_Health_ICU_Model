"""
MCS feature extractor — Johnathan's model 1 (MCS need in 12h).

357 features sourced from mcs_12h_feature_list.csv.
Naming convention: {METRIC}_{stat}{window}[_lb{lookback}h]
  e.g. LACTATE_min6h_lb6h = min over 6h window starting 6h ago
       VIS_delta4h         = change over last 4h

TODO (Johnathan): Implement extract() using your feature engineering script.
Model file goes in model_files/johnathan/mcs_model.pkl.
"""

import pandas as pd

MCS_FEATURES: list[str] = [
    "ADMIT_SRC_CD",
    "ALT_delta12h_lb12h",
    "ALT_max12h_lb12h",
    "ALT_max6h_lb6h",
    "ALT_mean4h",
    "ALT_min12h_lb12h",
    "ALT_min4h",
    "ALT_min6h_lb6h",
    "ALT_slope12h_lb12h",
    "AST_delta12h_lb12h",
    "AST_delta6h_lb6h",
    "AST_max12h_lb12h",
    "AST_max6h_lb6h",
    "AST_min12h_lb12h",
    "AST_min4h",
    "AST_min6h_lb6h",
    "AST_slope6h_lb6h",
    "BUN_delta12h_lb12h",
    "BUN_max12h_lb12h",
    "BUN_mean12h_lb12h",
    "BUN_mean4h",
    "BUN_min12h_lb12h",
    "BUN_min4h",
    "BUN_min6h_lb6h",
    "BUN_now",
    "CI_max12h_lb12h",
    "CI_max6h_lb6h",
    "CI_mean12h_lb12h",
    "CI_mean6h_lb6h",
    "CI_min12h_lb12h",
    "CI_min4h",
    "CI_min6h_lb6h",
    "CO_max12h_lb12h",
    "CO_max4h",
    "CO_max6h_lb6h",
    "CO_mean12h_lb12h",
    "CO_min12h_lb12h",
    "CO_min6h_lb6h",
    "CO_now",
    "CPO_delta4h",
    "CPO_max6h_lb6h",
    "CPO_mean12h_lb12h",
    "CPO_mean6h_lb6h",
    "CPO_min12h_lb12h",
    "CPO_min6h_lb6h",
    "CREATININE_delta12h_lb12h",
    "CREATININE_delta4h",
    "CREATININE_delta6h_lb6h",
    "CREATININE_max12h_lb12h",
    "CREATININE_mean12h_lb12h",
    "CREATININE_mean4h",
    "CREATININE_mean6h_lb6h",
    "CREATININE_min12h_lb12h",
    "CREATININE_min4h",
    "CREATININE_min6h_lb6h",
    "CREATININE_now",
    "CREATININE_now_lb6h",
    "CREATININE_slope12h_lb12h",
    "CURRENT_STAGE_NUM",
    "CVP_delta12h_lb12h",
    "CVP_max4h",
    "CVP_max6h_lb6h",
    "CVP_mean6h_lb6h",
    "CVP_min12h_lb12h",
    "CVP_min6h_lb6h",
    "DBP_delta12h_lb12h",
    "DBP_delta4h",
    "DBP_max12h_lb12h",
    "DBP_max4h",
    "DBP_max6h_lb6h",
    "DBP_mean6h_lb6h",
    "DBP_min12h_lb12h",
    "DBP_min6h_lb6h",
    "DBP_now",
    "HCO3_delta12h_lb12h",
    "HCO3_delta4h",
    "HCO3_delta6h_lb6h",
    "HCO3_max12h_lb12h",
    "HCO3_max6h_lb6h",
    "HCO3_mean4h",
    "HCO3_min12h_lb12h",
    "HCO3_now_lb6h",
    "HCO3_slope12h_lb12h",
    "HCO3_slope4h",
    "HGB_delta12h_lb12h",
    "HGB_delta6h_lb6h",
    "HGB_max12h_lb12h",
    "HGB_max4h",
    "HGB_max6h_lb6h",
    "HGB_mean12h_lb12h",
    "HGB_mean4h",
    "HGB_mean6h_lb6h",
    "HGB_min12h_lb12h",
    "HGB_min4h",
    "HGB_min6h_lb6h",
    "HGB_now",
    "HGB_slope12h_lb12h",
    "HGB_slope6h_lb6h",
    "HIGH_SHOCK_BURDEN",
    "HOURS_ON_CURRENT_STAGE",
    "HOUR_FROM_ADMIT",
    "HR_delta12h_lb12h",
    "HR_delta4h",
    "HR_delta6h_lb6h",
    "HR_max12h_lb12h",
    "HR_max4h",
    "HR_mean12h_lb12h",
    "HR_min12h_lb12h",
    "HR_min4h",
    "HR_min6h_lb6h",
    "HR_slope6h_lb6h",
    "INR_delta12h_lb12h",
    "INR_max12h_lb12h",
    "INR_max4h",
    "INR_max6h_lb6h",
    "INR_mean12h_lb12h",
    "INR_mean6h_lb6h",
    "INR_min12h_lb12h",
    "INR_min4h",
    "INR_min6h_lb6h",
    "INR_now",
    "INR_slope12h_lb12h",
    "INR_slope6h_lb6h",
    "LACTATE_X_HR_mean4h",
    "LACTATE_delta12h_lb12h",
    "LACTATE_delta6h_lb6h",
    "LACTATE_max12h_lb12h",
    "LACTATE_max6h_lb6h",
    "LACTATE_mean12h_lb12h",
    "LACTATE_min12h_lb12h",
    "LACTATE_min6h_lb6h",
    "LACTATE_now",
    "LACTATE_now_lb6h",
    "LACTATE_slope12h_lb12h",
    "LACTATE_slope4h",
    "MAP_HR_RATIO_max4h",
    "MAP_HR_RATIO_mean4h",
    "MAP_HR_RATIO_now",
    "MAP_delta12h_lb12h",
    "MAP_delta4h",
    "MAP_delta6h_lb6h",
    "MAP_max12h_lb12h",
    "MAP_max6h_lb6h",
    "MAP_mean12h_lb12h",
    "MAP_mean4h",
    "MAP_min12h_lb12h",
    "MAP_min4h",
    "MAP_min6h_lb6h",
    "MAP_now",
    "MAP_slope12h_lb12h",
    "MAP_slope6h_lb6h",
    "NT_PROBNP_delta12h_lb12h",
    "NT_PROBNP_max12h_lb12h",
    "NT_PROBNP_max4h",
    "NT_PROBNP_max6h_lb6h",
    "NT_PROBNP_mean4h",
    "NT_PROBNP_min12h_lb12h",
    "NT_PROBNP_min4h",
    "NT_PROBNP_now",
    "N_VASOPRESSORS_ACTIVE",
    "N_VASOPRESSORS_ACTIVE_ADDITIONS_12H",
    "PAD_delta12h_lb12h",
    "PAD_delta4h",
    "PAD_delta6h_lb6h",
    "PAD_max12h_lb12h",
    "PAD_max6h_lb6h",
    "PAD_mean12h_lb12h",
    "PAD_min12h_lb12h",
    "PAD_min4h",
    "PAD_min6h_lb6h",
    "PAD_slope4h",
    "PAM_delta12h_lb12h",
    "PAM_delta4h",
    "PAM_max12h_lb12h",
    "PAM_mean12h_lb12h",
    "PAM_mean6h_lb6h",
    "PAM_min12h_lb12h",
    "PAM_min6h_lb6h",
    "PAPI_delta4h",
    "PAPI_max12h_lb12h",
    "PAPI_max4h",
    "PAPI_max6h_lb6h",
    "PAPI_mean12h_lb12h",
    "PAPI_mean4h",
    "PAPI_mean6h_lb6h",
    "PAPI_min12h_lb12h",
    "PAPI_min6h_lb6h",
    "PAPI_slope4h",
    "PAPI_slope6h_lb6h",
    "PAS_max12h_lb12h",
    "PAS_max6h_lb6h",
    "PAS_mean4h",
    "PAS_mean6h_lb6h",
    "PAS_min12h_lb12h",
    "PAS_min6h_lb6h",
    "PAS_slope4h",
    "PCO2_delta12h_lb12h",
    "PCO2_delta6h_lb6h",
    "PCO2_max12h_lb12h",
    "PCO2_max4h",
    "PCO2_max6h_lb6h",
    "PCO2_mean12h_lb12h",
    "PCO2_min12h_lb12h",
    "PCO2_min4h",
    "PCO2_min6h_lb6h",
    "PCO2_now",
    "PCO2_now_lb6h",
    "PCO2_slope12h_lb12h",
    "PCO2_slope6h_lb6h",
    "PCWP_delta12h_lb12h",
    "PCWP_delta6h_lb6h",
    "PCWP_max12h_lb12h",
    "PCWP_max4h",
    "PCWP_mean12h_lb12h",
    "PCWP_mean6h_lb6h",
    "PCWP_min12h_lb12h",
    "PCWP_min6h_lb6h",
    "PCWP_slope12h_lb12h",
    "PH_delta12h_lb12h",
    "PH_delta4h",
    "PH_delta6h_lb6h",
    "PH_max12h_lb12h",
    "PH_max4h",
    "PH_mean12h_lb12h",
    "PH_min12h_lb12h",
    "PH_min6h_lb6h",
    "PH_slope12h_lb12h",
    "PLT_delta12h_lb12h",
    "PLT_delta4h",
    "PLT_delta6h_lb6h",
    "PLT_max12h_lb12h",
    "PLT_max4h",
    "PLT_max6h_lb6h",
    "PLT_mean12h_lb12h",
    "PLT_min12h_lb12h",
    "PLT_min6h_lb6h",
    "PLT_now",
    "PLT_slope12h_lb12h",
    "PLT_slope6h_lb6h",
    "PRINCIPAL_DX_CATEGORY",
    "RR_delta12h_lb12h",
    "RR_delta4h",
    "RR_delta6h_lb6h",
    "RR_max12h_lb12h",
    "RR_max4h",
    "RR_max6h_lb6h",
    "RR_mean6h_lb6h",
    "RR_min4h",
    "RR_min6h_lb6h",
    "RR_now",
    "RR_slope4h",
    "SBP_delta4h",
    "SBP_delta6h_lb6h",
    "SBP_max12h_lb12h",
    "SBP_max6h_lb6h",
    "SBP_mean12h_lb12h",
    "SBP_mean4h",
    "SBP_mean6h_lb6h",
    "SBP_min12h_lb12h",
    "SBP_min4h",
    "SBP_min6h_lb6h",
    "SBP_now",
    "SBP_slope6h_lb6h",
    "SHOCK_BURDEN_INT_12h",
    "SHOCK_BURDEN_mean12h_lb12h",
    "SHOCK_BURDEN_mean4h",
    "SHOCK_BURDEN_mean6h_lb6h",
    "SHOCK_BURDEN_now",
    "SHOCK_INDEX_now",
    "SPO2_delta12h_lb12h",
    "SPO2_delta4h",
    "SPO2_delta6h_lb6h",
    "SPO2_max12h_lb12h",
    "SPO2_max6h_lb6h",
    "SPO2_mean12h_lb12h",
    "SPO2_mean4h",
    "SPO2_min12h_lb12h",
    "SPO2_min4h",
    "SPO2_min6h_lb6h",
    "SPO2_now",
    "SPO2_now_lb12h",
    "STAGE_DELTA12H",
    "STAGE_DELTA6H",
    "SVO2_delta4h",
    "SVO2_max12h_lb12h",
    "SVO2_mean12h_lb12h",
    "SVO2_min12h_lb12h",
    "SVO2_min4h",
    "SVO2_min6h_lb6h",
    "SVR_delta12h_lb12h",
    "SVR_max12h_lb12h",
    "SVR_mean12h_lb12h",
    "SVR_min12h_lb12h",
    "SVR_min6h_lb6h",
    "SVR_slope12h_lb12h",
    "TEMP_delta12h_lb12h",
    "TEMP_delta4h",
    "TEMP_max12h_lb12h",
    "TEMP_max4h",
    "TEMP_max6h_lb6h",
    "TEMP_mean12h_lb12h",
    "TEMP_mean4h",
    "TEMP_mean6h_lb6h",
    "TEMP_min12h_lb12h",
    "TEMP_min4h",
    "TEMP_min6h_lb6h",
    "TEMP_now",
    "TEMP_now_lb6h",
    "TROPONIN_I_delta6h_lb6h",
    "TROPONIN_I_max12h_lb12h",
    "TROPONIN_I_max4h",
    "TROPONIN_I_max6h_lb6h",
    "TROPONIN_I_mean12h_lb12h",
    "TROPONIN_I_min12h_lb12h",
    "TROPONIN_I_min4h",
    "TROPONIN_I_min6h_lb6h",
    "TROPONIN_I_now",
    "TROPONIN_I_now_lb6h",
    "URINE_OUT_HR_delta12h_lb12h",
    "URINE_OUT_HR_max12h_lb12h",
    "URINE_OUT_HR_max4h",
    "URINE_OUT_HR_max6h_lb6h",
    "URINE_OUT_HR_mean4h",
    "URINE_OUT_HR_min12h_lb12h",
    "URINE_OUT_HR_min4h",
    "URINE_OUT_HR_now",
    "URINE_OUT_HR_now_lb6h",
    "VIS_PER_MAP_max4h",
    "VIS_PER_MAP_mean4h",
    "VIS_PER_MAP_now",
    "VIS_SPIKE_4H",
    "VIS_delta12h_lb12h",
    "VIS_delta4h",
    "VIS_delta6h_lb6h",
    "VIS_max12h_lb12h",
    "VIS_max6h_lb6h",
    "VIS_min12h_lb12h",
    "VIS_min6h_lb6h",
    "VIS_now",
    "VIS_now_lb6h",
    "VIS_slope4h",
    "VIS_slope6h_lb6h",
    "WBC_delta12h_lb12h",
    "WBC_delta4h",
    "WBC_delta6h_lb6h",
    "WBC_max12h_lb12h",
    "WBC_max4h",
    "WBC_max6h_lb6h",
    "WBC_mean12h_lb12h",
    "WBC_mean4h",
    "WBC_mean6h_lb6h",
    "WBC_min12h_lb12h",
    "WBC_min6h_lb6h",
    "WBC_now",
    "WBC_now_lb6h",
    "WBC_slope12h_lb12h",
]


def extract(raw_patient_data: dict) -> pd.DataFrame:
    """
    Build the MCS feature DataFrame from raw_patient_data.

    Strategy: populate everything computable from the single-point seed observation,
    approximate 6h/12h lookback windows from 4h rolling stats, zero-fill the rest.
    Feature column order is driven by bundle["feature_cols"] at runtime.
    """
    import numpy as np
    from app.models import loader

    df_in = raw_patient_data["df"]
    scai  = raw_patient_data["scai_stage_hourly"]
    meds  = raw_patient_data.get("medications")
    hour  = int(raw_patient_data.get("hour_from_admit", df_in["HOUR_FROM_ADMIT"].iloc[0]))

    row = df_in.iloc[0].to_dict()

    row["CURRENT_STAGE_NUM"] = row.get("SCAI_STAGE_NUM", 1)

    # _now aliases for every vital
    vitals = [
        "HR","SBP","DBP","MAP","RR","SPO2","TEMP","URINE_OUT_HR",
        "CVP","PAS","PAD","PAM","PCWP","CO","CI","SVO2","SVR","CPO","PAPI",
        "LACTATE","CREATININE","BUN","NT_PROBNP","TROPONIN_I",
        "PH","PCO2","HCO3","AST","ALT","WBC","HGB","PLT","INR","VIS",
    ]
    for v in vitals:
        val = row.get(v, 0.0)
        row[f"{v}_now"] = val
        mean4h = row.get(f"{v}_mean4h", val)
        min4h  = row.get(f"{v}_min4h",  val)
        max4h  = row.get(f"{v}_max4h",  val)
        # Approximate 6h and 12h lookback windows from available 4h stats
        for lb in (6, 12):
            row[f"{v}_mean{lb}h_lb{lb}h"]  = mean4h
            row[f"{v}_min{lb}h_lb{lb}h"]   = min4h
            row[f"{v}_max{lb}h_lb{lb}h"]   = max4h
            row[f"{v}_delta{lb}h_lb{lb}h"] = 0.0
            row[f"{v}_slope{lb}h_lb{lb}h"] = 0.0
            row[f"{v}_now_lb{lb}h"]        = val

    # Derived compound features
    hr   = row.get("HR",   80.0)
    sbp  = row.get("SBP",  90.0)
    map_ = row.get("MAP",  70.0)
    lac  = row.get("LACTATE", 1.0)
    vis  = row.get("VIS",  0.0)
    n_vaso = int(row.get("N_VASOPRESSORS_ACTIVE", 0))
    stage  = int(row.get("CURRENT_STAGE_NUM", 1))

    row["SHOCK_INDEX_now"]     = hr / max(sbp, 1)
    row["MAP_HR_RATIO_now"]    = map_ / max(hr, 1)
    row["MAP_HR_RATIO_max4h"]  = row.get("MAP_max4h", map_) / max(row.get("HR_min4h", hr), 1)
    row["MAP_HR_RATIO_mean4h"] = row.get("MAP_mean4h", map_) / max(row.get("HR_mean4h", hr), 1)
    row["LACTATE_X_HR_now"]    = lac * hr
    row["LACTATE_X_HR_mean4h"] = row.get("LACTATE_mean4h", lac) * row.get("HR_mean4h", hr)
    row["LACTATE_X_HR_max4h"]  = row.get("LACTATE_max4h", lac) * row.get("HR_max4h",  hr)
    row["VIS_PER_MAP_now"]     = vis / max(map_, 1)
    row["VIS_PER_MAP_mean4h"]  = row.get("VIS_mean4h", vis) / max(row.get("MAP_mean4h", map_), 1)
    row["VIS_PER_MAP_max4h"]   = row.get("VIS_max4h",  vis) / max(row.get("MAP_min4h",  map_), 1)
    row["VIS_SPIKE_4H"]        = 1 if row.get("VIS_delta4h", 0.0) > 5.0 else 0

    shock_burden = stage * (1 + n_vaso * 0.5)
    row["SHOCK_BURDEN_now"]           = shock_burden
    row["SHOCK_BURDEN_mean4h"]        = shock_burden
    row["SHOCK_BURDEN_mean6h_lb6h"]   = shock_burden
    row["SHOCK_BURDEN_mean12h_lb12h"] = shock_burden
    row["SHOCK_BURDEN_INT_12h"]       = shock_burden * min(hour, 12)
    row["HIGH_SHOCK_BURDEN"]          = 1 if stage >= 3 and n_vaso >= 2 else 0

    # SCAI trajectory from hourly history
    stages = scai["SCAI_STAGE_NUM"].tolist() if len(scai) > 0 else [stage]
    hours_on = sum(1 for s in reversed(stages) if s == stages[-1])
    row["HOURS_ON_CURRENT_STAGE"]         = hours_on
    row["STAGE_DELTA6H"]                  = stages[-1] - (stages[-7]  if len(stages) >= 7  else stages[0])
    row["STAGE_DELTA12H"]                 = stages[-1] - (stages[-13] if len(stages) >= 13 else stages[0])
    row["STAGE_LAG12H"]                   = stages[-13] if len(stages) >= 13 else stages[0]
    row["N_VASOPRESSORS_ACTIVE_ADDITIONS_12H"] = n_vaso
    row["N_INOTROPES_ACTIVE_ADDITIONS_6H"]     = int(row.get("N_INOTROPES_ACTIVE", 0))

    # Both categoricals trained as string categories — keep as str before .astype("category").
    row["PRINCIPAL_DX_CATEGORY"] = str(row.get("PRINCIPAL_DX_CATEGORY", "OTHER")) or "OTHER"
    row["ADMIT_SRC_CD"]          = str(int(row.get("ADMIT_SRC_CD", 0)))

    df = pd.DataFrame([row])

    bundle      = loader.get("mcs")
    feature_cols = bundle["feature_cols"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing)], axis=1)

    df = df[feature_cols].copy()

    for col in bundle.get("cat_feats", []):
        if col in df.columns:
            df[col] = df[col].fillna("OTHER").astype(str).astype("category")

    return df
