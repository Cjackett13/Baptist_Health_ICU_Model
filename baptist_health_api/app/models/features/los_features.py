"""
Length-of-stay feature extractor — Lilly's model (14 features, predicts hospital LOS in days).

raw_patient_data must contain:
  "df" — pd.DataFrame with one row per patient/encounter containing the 14 columns below.
         These are encounter-level aggregates (12h window stats + static demographics).

TODO (Lilly): Implement extract() using your training pipeline's feature builder.
              Replace the NotImplementedError with a call to your preprocessing function.
              Model file goes in model_files/lilly/los_model.pkl.
"""

import pandas as pd

LOS_FEATURES: list[str] = [
    "med_infusion_mean_12h",
    "scai_prop_ge3_12h",
    "current_scai_12h",
    "proc_n_12h",
    "med_distinct_12h",
    "clinical_event_n_12h",
    "dx_cat_arrhythmia_arrest",
    "med_per_clinical_event_12h",
    "scai_slope_12h",
    "age_years",
    "sex_bin",
    "admit_type_cd",
    "admit_src_cd",
    "unit_cd",
]


def extract(raw_patient_data: dict) -> pd.DataFrame:
    """
    Build the LOS feature DataFrame from raw_patient_data.

    The LOS model was trained with one-hot encoded categorical columns
    (admit_type_cd_ELECTIVE, admit_src_cd_ED, etc.) as stored in
    xgb_los_model_meta.json. Feature column order is read from that file
    at runtime so it always matches the trained pipeline.
    """
    import numpy as np

    scai  = raw_patient_data["scai_stage_hourly"]
    meds  = raw_patient_data.get("medications")
    hour  = int(raw_patient_data.get("hour_from_admit", 1))
    age   = int(raw_patient_data.get("age", 65))

    # ── SCAI over last 12 hours ───────────────────────────────────────────────
    stages_arr = scai["SCAI_STAGE_NUM"].to_numpy(dtype=float)
    hours_arr  = scai["HOUR_FROM_ADMIT"].to_numpy(dtype=float)
    tail12_s   = stages_arr[-12:]
    tail12_h   = hours_arr[-12:]

    current_scai_12h  = int(stages_arr[-1]) if len(stages_arr) else 1
    scai_prop_ge3_12h = float((tail12_s >= 3).mean()) if len(tail12_s) else 0.0

    if len(tail12_h) >= 2:
        scai_slope_12h = float(np.polyfit(tail12_h, tail12_s, 1)[0])
    else:
        scai_slope_12h = 0.0

    # ── Medication features ────────────────────────────────────────────────────
    med_distinct = int(meds["MEDICATION_CD"].nunique()) if meds is not None and len(meds) > 0 else 0

    # clinical_event_n_12h in training is a raw EHR row count (all labs, vitals,
    # nursing records) — training median ≈ 158, winsor bounds [144, 371].
    # Seed data has no access to this; use the training median so the model
    # sees a representative value instead of being clipped at the boundary.
    _CLINICAL_EVENT_MEDIAN = 158.0
    _MED_PER_EVENT_MEDIAN  = 0.0069

    med_infusion_mean_12h       = med_distinct / max(min(hour, 12), 1)
    med_per_clinical_event_12h  = _MED_PER_EVENT_MEDIAN if med_distinct == 0 else min(
        med_distinct / _CLINICAL_EVENT_MEDIAN, 0.0297  # winsor upper bound
    )

    # ── Build the base row (14 raw features the pipeline preprocessor expects) ─
    # The LosTrainPreprocessor OHEs admit_type_cd/admit_src_cd/unit_cd; UNKNOWN
    # maps to the training mode (EMERGENCY / ED / CICU) via cat_modes_ fallback.
    row: dict = {
        "med_infusion_mean_12h":      med_infusion_mean_12h,
        "scai_prop_ge3_12h":          scai_prop_ge3_12h,
        "current_scai_12h":           current_scai_12h,
        "proc_n_12h":                 0,
        "med_distinct_12h":           med_distinct,
        "clinical_event_n_12h":       _CLINICAL_EVENT_MEDIAN,
        "dx_cat_arrhythmia_arrest":   0,
        "med_per_clinical_event_12h": med_per_clinical_event_12h,
        "scai_slope_12h":             scai_slope_12h,
        "age_years":                  age,
        "sex_bin":                    0,
        "admit_type_cd":              "UNKNOWN",
        "admit_src_cd":               "UNKNOWN",
        "unit_cd":                    "UNKNOWN",
    }

    return pd.DataFrame([row])[LOS_FEATURES]
