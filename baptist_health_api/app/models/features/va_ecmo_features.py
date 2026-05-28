"""
VA-ECMO feature extractor — Johnathan's model 2 (VA-ECMO need in 12h).

329 features sourced from ECMO_12h_feature_list.csv.
Naming convention: {METRIC}_{stat}{window}[_lb{lookback}h]
  e.g. LACTATE_min6h_lb6h = min over 6h window starting 6h ago
       VIS_delta4h         = change over last 4h

TODO (Johnathan): Implement extract() using your feature engineering script.
Model file goes in model_files/johnathan/va_ecmo_model.pkl.
"""

import pandas as pd

VA_ECMO_FEATURES: list[str] = [
    "AGE",
    "ALT_delta12h_lb12h",
    "ALT_delta6h_lb6h",
    "ALT_max12h_lb12h",
    "ALT_max4h",
    "ALT_mean4h",
    "ALT_mean6h_lb6h",
    "ALT_min12h_lb12h",
    "ALT_min6h_lb6h",
    "ALT_slope12h_lb12h",
    "AST_delta12h_lb12h",
    "AST_delta6h_lb6h",
    "AST_max12h_lb12h",
    "AST_max6h_lb6h",
    "AST_mean12h_lb12h",
    "AST_mean4h",
    "AST_min12h_lb12h",
    "AST_min4h",
    "AST_min6h_lb6h",
    "AST_now",
    "AST_now_lb6h",
    "AST_slope4h",
    "AST_slope6h_lb6h",
    "BUN_delta12h_lb12h",
    "BUN_delta6h_lb6h",
    "BUN_max12h_lb12h",
    "BUN_max4h",
    "BUN_max6h_lb6h",
    "BUN_min12h_lb12h",
    "BUN_min6h_lb6h",
    "BUN_now",
    "BUN_slope6h_lb6h",
    "CI_delta12h_lb12h",
    "CI_delta4h",
    "CI_max12h_lb12h",
    "CI_max6h_lb6h",
    "CI_mean12h_lb12h",
    "CI_min12h_lb12h",
    "CI_min4h",
    "CI_min6h_lb6h",
    "CI_slope4h",
    "CO_max12h_lb12h",
    "CO_max6h_lb6h",
    "CO_mean12h_lb12h",
    "CO_mean4h",
    "CO_min6h_lb6h",
    "CPO_max12h_lb12h",
    "CPO_mean12h_lb12h",
    "CPO_min12h_lb12h",
    "CPO_min6h_lb6h",
    "CREATININE_delta12h_lb12h",
    "CREATININE_delta6h_lb6h",
    "CREATININE_max12h_lb12h",
    "CREATININE_max4h",
    "CREATININE_max6h_lb6h",
    "CREATININE_mean12h_lb12h",
    "CREATININE_mean4h",
    "CREATININE_min12h_lb12h",
    "CREATININE_min4h",
    "CREATININE_min6h_lb6h",
    "CREATININE_now",
    "CREATININE_now_lb6h",
    "CREATININE_slope12h_lb12h",
    "CREATININE_slope6h_lb6h",
    "CURRENT_STAGE_NUM",
    "CVP_delta12h_lb12h",
    "CVP_delta6h_lb6h",
    "CVP_max12h_lb12h",
    "CVP_max4h",
    "CVP_max6h_lb6h",
    "CVP_min12h_lb12h",
    "CVP_min4h",
    "DBP_delta6h_lb6h",
    "DBP_max12h_lb12h",
    "DBP_mean4h",
    "DBP_mean6h_lb6h",
    "DBP_min12h_lb12h",
    "DBP_min4h",
    "DBP_min6h_lb6h",
    "DBP_now",
    "HCO3_delta12h_lb12h",
    "HCO3_delta6h_lb6h",
    "HCO3_max12h_lb12h",
    "HCO3_max4h",
    "HCO3_max6h_lb6h",
    "HCO3_mean12h_lb12h",
    "HCO3_mean6h_lb6h",
    "HCO3_min12h_lb12h",
    "HCO3_min6h_lb6h",
    "HCO3_now",
    "HCO3_slope12h_lb12h",
    "HGB_delta12h_lb12h",
    "HGB_delta6h_lb6h",
    "HGB_max12h_lb12h",
    "HGB_max4h",
    "HGB_max6h_lb6h",
    "HGB_mean4h",
    "HGB_min12h_lb12h",
    "HGB_min4h",
    "HGB_min6h_lb6h",
    "HGB_now",
    "HOURS_ON_CURRENT_STAGE",
    "HOUR_FROM_ADMIT",
    "HR_delta4h",
    "HR_max12h_lb12h",
    "HR_mean12h_lb12h",
    "HR_min12h_lb12h",
    "HR_min6h_lb6h",
    "HR_slope4h",
    "INR_delta12h_lb12h",
    "INR_max4h",
    "INR_mean12h_lb12h",
    "INR_min12h_lb12h",
    "INR_min4h",
    "INR_now",
    "INR_slope12h_lb12h",
    "INR_slope6h_lb6h",
    "LACTATE_X_HR_max4h",
    "LACTATE_X_HR_now",
    "LACTATE_delta12h_lb12h",
    "LACTATE_delta6h_lb6h",
    "LACTATE_max12h_lb12h",
    "LACTATE_max4h",
    "LACTATE_max6h_lb6h",
    "LACTATE_mean6h_lb6h",
    "LACTATE_min12h_lb12h",
    "LACTATE_min4h",
    "LACTATE_min6h_lb6h",
    "LACTATE_now",
    "LACTATE_now_lb12h",
    "LACTATE_slope12h_lb12h",
    "MAP_HR_RATIO_max4h",
    "MAP_HR_RATIO_now",
    "MAP_delta12h_lb12h",
    "MAP_max12h_lb12h",
    "MAP_max4h",
    "MAP_max6h_lb6h",
    "MAP_mean4h",
    "MAP_mean6h_lb6h",
    "MAP_min12h_lb12h",
    "MAP_min6h_lb6h",
    "MAP_now",
    "MAP_now_lb6h",
    "MAP_slope12h_lb12h",
    "NT_PROBNP_delta12h_lb12h",
    "NT_PROBNP_delta6h_lb6h",
    "NT_PROBNP_max12h_lb12h",
    "NT_PROBNP_max6h_lb6h",
    "NT_PROBNP_mean12h_lb12h",
    "NT_PROBNP_mean6h_lb6h",
    "NT_PROBNP_min12h_lb12h",
    "NT_PROBNP_min4h",
    "N_INOTROPES_ACTIVE_ADDITIONS_6H",
    "N_VASOPRESSORS_ACTIVE",
    "PAD_delta12h_lb12h",
    "PAD_delta4h",
    "PAD_max12h_lb12h",
    "PAD_max4h",
    "PAD_max6h_lb6h",
    "PAD_mean12h_lb12h",
    "PAD_mean6h_lb6h",
    "PAD_min12h_lb12h",
    "PAD_min4h",
    "PAD_min6h_lb6h",
    "PAD_slope4h",
    "PAM_delta6h_lb6h",
    "PAM_max12h_lb12h",
    "PAM_max4h",
    "PAM_max6h_lb6h",
    "PAM_mean12h_lb12h",
    "PAM_mean4h",
    "PAM_min12h_lb12h",
    "PAM_min4h",
    "PAM_min6h_lb6h",
    "PAPI_delta4h",
    "PAPI_max12h_lb12h",
    "PAPI_max4h",
    "PAPI_mean12h_lb12h",
    "PAPI_mean4h",
    "PAPI_min12h_lb12h",
    "PAPI_min4h",
    "PAPI_min6h_lb6h",
    "PAPI_slope6h_lb6h",
    "PAS_delta12h_lb12h",
    "PAS_delta4h",
    "PAS_delta6h_lb6h",
    "PAS_max4h",
    "PAS_max6h_lb6h",
    "PAS_mean12h_lb12h",
    "PAS_min4h",
    "PAS_now_lb6h",
    "PCO2_delta12h_lb12h",
    "PCO2_delta6h_lb6h",
    "PCO2_max12h_lb12h",
    "PCO2_max4h",
    "PCO2_max6h_lb6h",
    "PCO2_mean4h",
    "PCO2_min4h",
    "PCO2_min6h_lb6h",
    "PCO2_slope12h_lb12h",
    "PCO2_slope6h_lb6h",
    "PCWP_delta6h_lb6h",
    "PCWP_max12h_lb12h",
    "PCWP_max6h_lb6h",
    "PCWP_min12h_lb12h",
    "PH_delta12h_lb12h",
    "PH_delta4h",
    "PH_delta6h_lb6h",
    "PH_max12h_lb12h",
    "PH_max4h",
    "PH_max6h_lb6h",
    "PH_mean12h_lb12h",
    "PH_min12h_lb12h",
    "PH_min4h",
    "PH_min6h_lb6h",
    "PH_now",
    "PH_slope4h",
    "PH_slope6h_lb6h",
    "PLT_delta12h_lb12h",
    "PLT_delta6h_lb6h",
    "PLT_max12h_lb12h",
    "PLT_max6h_lb6h",
    "PLT_mean12h_lb12h",
    "PLT_min12h_lb12h",
    "PLT_min4h",
    "PLT_min6h_lb6h",
    "PLT_slope12h_lb12h",
    "PRINCIPAL_DX_CATEGORY",
    "RR_delta12h_lb12h",
    "RR_delta4h",
    "RR_delta6h_lb6h",
    "RR_max12h_lb12h",
    "RR_max4h",
    "RR_mean12h_lb12h",
    "RR_mean6h_lb6h",
    "RR_min6h_lb6h",
    "RR_now",
    "SBP_delta12h_lb12h",
    "SBP_delta4h",
    "SBP_max12h_lb12h",
    "SBP_mean12h_lb12h",
    "SBP_mean4h",
    "SBP_mean6h_lb6h",
    "SBP_min12h_lb12h",
    "SBP_min4h",
    "SBP_min6h_lb6h",
    "SBP_now",
    "SBP_now_lb12h",
    "SHOCK_BURDEN_mean12h_lb12h",
    "SHOCK_BURDEN_mean4h",
    "SHOCK_BURDEN_mean6h_lb6h",
    "SHOCK_BURDEN_now",
    "SHOCK_INDEX_max4h",
    "SHOCK_INDEX_mean4h",
    "SHOCK_INDEX_now",
    "SPO2_delta12h_lb12h",
    "SPO2_delta6h_lb6h",
    "SPO2_max12h_lb12h",
    "SPO2_mean6h_lb6h",
    "SPO2_min12h_lb12h",
    "STAGE_DELTA12H",
    "STAGE_LAG12H",
    "SVO2_delta4h",
    "SVO2_max6h_lb6h",
    "SVO2_min4h",
    "SVO2_slope4h",
    "SVR_max12h_lb12h",
    "SVR_max4h",
    "SVR_max6h_lb6h",
    "SVR_mean12h_lb12h",
    "SVR_min12h_lb12h",
    "SVR_now",
    "TEMP_delta12h_lb12h",
    "TEMP_delta6h_lb6h",
    "TEMP_max12h_lb12h",
    "TEMP_mean12h_lb12h",
    "TEMP_min12h_lb12h",
    "TEMP_min4h",
    "TEMP_min6h_lb6h",
    "TROPONIN_I_delta12h_lb12h",
    "TROPONIN_I_delta4h",
    "TROPONIN_I_delta6h_lb6h",
    "TROPONIN_I_max12h_lb12h",
    "TROPONIN_I_max4h",
    "TROPONIN_I_mean12h_lb12h",
    "TROPONIN_I_min12h_lb12h",
    "TROPONIN_I_min4h",
    "TROPONIN_I_min6h_lb6h",
    "TROPONIN_I_slope12h_lb12h",
    "URINE_OUT_HR_delta4h",
    "URINE_OUT_HR_delta6h_lb6h",
    "URINE_OUT_HR_max12h_lb12h",
    "URINE_OUT_HR_max6h_lb6h",
    "URINE_OUT_HR_mean12h_lb12h",
    "URINE_OUT_HR_mean6h_lb6h",
    "URINE_OUT_HR_min12h_lb12h",
    "URINE_OUT_HR_min4h",
    "URINE_OUT_HR_min6h_lb6h",
    "URINE_OUT_HR_slope6h_lb6h",
    "VIS_PER_MAP_max4h",
    "VIS_PER_MAP_mean4h",
    "VIS_PER_MAP_now",
    "VIS_SPIKE_4H",
    "VIS_delta12h_lb12h",
    "VIS_delta4h",
    "VIS_delta6h_lb6h",
    "VIS_max4h",
    "VIS_max6h_lb6h",
    "VIS_mean12h_lb12h",
    "VIS_min12h_lb12h",
    "VIS_min4h",
    "VIS_min6h_lb6h",
    "VIS_now",
    "VIS_now_lb6h",
    "VIS_slope12h_lb12h",
    "VIS_slope4h",
    "VIS_slope6h_lb6h",
    "WBC_delta12h_lb12h",
    "WBC_delta4h",
    "WBC_delta6h_lb6h",
    "WBC_max12h_lb12h",
    "WBC_max6h_lb6h",
    "WBC_mean12h_lb12h",
    "WBC_mean6h_lb6h",
    "WBC_min12h_lb12h",
    "WBC_min4h",
    "WBC_min6h_lb6h",
    "WBC_now_lb6h",
]


def extract(raw_patient_data: dict) -> pd.DataFrame:
    """
    Build the VA-ECMO feature DataFrame from raw_patient_data.

    Same approximation strategy as mcs_features.extract(): populate everything
    computable from the single-point observation, approximate lookback windows
    from 4h rolling stats, zero-fill the rest. Column order comes from
    bundle["feature_cols"] at runtime.
    """
    import numpy as np
    from app.models import loader

    df_in = raw_patient_data["df"]
    scai  = raw_patient_data["scai_stage_hourly"]
    hour  = int(raw_patient_data.get("hour_from_admit", df_in["HOUR_FROM_ADMIT"].iloc[0]))

    row = df_in.iloc[0].to_dict()

    row["CURRENT_STAGE_NUM"] = row.get("SCAI_STAGE_NUM", 1)
    row["AGE"]               = row.get("AGE", raw_patient_data.get("age", 65))

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
        for lb in (6, 12):
            row[f"{v}_mean{lb}h_lb{lb}h"]  = mean4h
            row[f"{v}_min{lb}h_lb{lb}h"]   = min4h
            row[f"{v}_max{lb}h_lb{lb}h"]   = max4h
            row[f"{v}_delta{lb}h_lb{lb}h"] = 0.0
            row[f"{v}_slope{lb}h_lb{lb}h"] = 0.0
            row[f"{v}_now_lb{lb}h"]        = val

    hr   = row.get("HR",   80.0)
    sbp  = row.get("SBP",  90.0)
    map_ = row.get("MAP",  70.0)
    lac  = row.get("LACTATE", 1.0)
    vis  = row.get("VIS",  0.0)
    n_vaso = int(row.get("N_VASOPRESSORS_ACTIVE", 0))
    stage  = int(row.get("CURRENT_STAGE_NUM", 1))

    row["SHOCK_INDEX_now"]    = hr / max(sbp, 1)
    row["SHOCK_INDEX_max4h"]  = row.get("HR_max4h", hr) / max(row.get("SBP_min4h", sbp), 1)
    row["SHOCK_INDEX_mean4h"] = row.get("HR_mean4h", hr) / max(row.get("SBP_mean4h", sbp), 1)
    row["MAP_HR_RATIO_now"]   = map_ / max(hr, 1)
    row["MAP_HR_RATIO_max4h"] = row.get("MAP_max4h", map_) / max(row.get("HR_min4h", hr), 1)
    row["LACTATE_X_HR_now"]   = lac * hr
    row["LACTATE_X_HR_max4h"] = row.get("LACTATE_max4h", lac) * row.get("HR_max4h", hr)
    row["VIS_PER_MAP_now"]    = vis / max(map_, 1)
    row["VIS_PER_MAP_mean4h"] = row.get("VIS_mean4h", vis) / max(row.get("MAP_mean4h", map_), 1)
    row["VIS_PER_MAP_max4h"]  = row.get("VIS_max4h",  vis) / max(row.get("MAP_min4h",  map_), 1)
    row["VIS_SPIKE_4H"]       = 1 if row.get("VIS_delta4h", 0.0) > 5.0 else 0

    shock_burden = stage * (1 + n_vaso * 0.5)
    row["SHOCK_BURDEN_now"]           = shock_burden
    row["SHOCK_BURDEN_mean4h"]        = shock_burden
    row["SHOCK_BURDEN_mean6h_lb6h"]   = shock_burden
    row["SHOCK_BURDEN_mean12h_lb12h"] = shock_burden

    stages = scai["SCAI_STAGE_NUM"].tolist() if len(scai) > 0 else [stage]
    hours_on = sum(1 for s in reversed(stages) if s == stages[-1])
    row["HOURS_ON_CURRENT_STAGE"]             = hours_on
    row["STAGE_DELTA12H"]                     = stages[-1] - (stages[-13] if len(stages) >= 13 else stages[0])
    row["STAGE_LAG12H"]                       = stages[-13] if len(stages) >= 13 else stages[0]
    row["N_INOTROPES_ACTIVE_ADDITIONS_6H"]    = int(row.get("N_INOTROPES_ACTIVE", 0))

    row["PRINCIPAL_DX_CATEGORY"] = str(row.get("PRINCIPAL_DX_CATEGORY", "OTHER")) or "OTHER"
    row["ADMIT_SRC_CD"]          = int(row.get("ADMIT_SRC_CD", 0))

    df = pd.DataFrame([row])

    bundle       = loader.get("va_ecmo")
    feature_cols = bundle["feature_cols"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing)], axis=1)

    df = df[feature_cols].copy()

    _STR_CATS = {"PRINCIPAL_DX_CATEGORY"}
    for col in bundle.get("cat_feats", []):
        if col not in df.columns:
            continue
        if col in _STR_CATS:
            df[col] = df[col].astype(str).astype("category")
        else:
            df[col] = df[col].fillna(0).astype(int).astype("category")

    return df
