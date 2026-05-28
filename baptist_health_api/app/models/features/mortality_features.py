"""
Mortality feature extractor — Lilly's model (26 features, predicts hospital mortality).

raw_patient_data must contain:
  "df" — pd.DataFrame with one row per patient/encounter containing the 26 columns below.
         These are encounter-level aggregates (SCAI trajectory stats, medication/procedure
         intensity, clinical event rates, and demographic features).

TODO (Lilly): Implement extract() using your training pipeline's feature builder.
              Replace the NotImplementedError with a call to your preprocessing function.
              Model file goes in model_files/lilly/mortality_model.pkl.
"""

import pandas as pd

MORTALITY_FEATURES: list[str] = [
    "med_distinct",
    "med_infusion_mean",
    "proc_n",
    "proc_distinct_nom",
    "current_scai",
    "scai_prop_ge3",
    "scai_std",
    "scai_slope_12h",
    "scai_max",
    "scai_increase_count",
    "inf_early_minus_all",
    "infusion_early_above_mean",
    "clinical_events_last_4h",
    "clinical_events_per_icu_hour",
    "med_admin_per_icu_hour",
    "proc_per_icu_hour",
    "med_per_clinical_event",
    "proc_n_bucket",
    "med_distinct_bucket",
    "med_residual_within_scai",
    "proc_residual_within_scai",
    "clinical_event_residual_within_scai",
    "infusion_x_scai_severity",
    "demo_profile_bucket",
    "age_years",
    "frailty_index",
]


def extract(raw_patient_data: dict) -> pd.DataFrame:
    """
    Build the mortality feature DataFrame from raw_patient_data.

    Feature list is read from the loaded bundle at runtime (bundle["feature_names"]),
    so the column order always matches what the model was trained on.
    """
    import numpy as np
    from app.models import loader

    scai  = raw_patient_data["scai_stage_hourly"]
    meds  = raw_patient_data.get("medications")
    hour  = int(raw_patient_data.get("hour_from_admit", 1))
    age   = int(raw_patient_data.get("age", 65))

    # ── SCAI trajectory ───────────────────────────────────────────────────────
    stages = scai["SCAI_STAGE_NUM"]
    hours_arr  = scai["HOUR_FROM_ADMIT"].to_numpy(dtype=float)
    stages_arr = stages.to_numpy(dtype=float)

    current_scai      = int(stages_arr[-1]) if len(stages_arr) else 1
    scai_prop_ge3     = float((stages_arr >= 3).mean()) if len(stages_arr) else 0.0
    scai_std          = float(stages_arr.std())          if len(stages_arr) > 1 else 0.0
    scai_max          = int(stages_arr.max())            if len(stages_arr) else current_scai
    scai_increase_count = int((np.diff(stages_arr) > 0).sum()) if len(stages_arr) > 1 else 0

    # Linear slope of SCAI over last 12 rows
    tail_h = hours_arr[-12:]
    tail_s = stages_arr[-12:]
    if len(tail_h) >= 2:
        scai_slope_12h = float(np.polyfit(tail_h, tail_s, 1)[0])
    else:
        scai_slope_12h = 0.0

    # ── Clinical event rates ──────────────────────────────────────────────────
    total_changes       = int((np.diff(stages_arr) != 0).sum()) if len(stages_arr) > 1 else 0
    clinical_events_last_4h   = int((np.diff(stages_arr[-4:]) != 0).sum()) if len(stages_arr) > 1 else 0
    clinical_events_per_icu_hour = total_changes / max(hour, 1)

    # ── Medication features ────────────────────────────────────────────────────
    if meds is not None and len(meds) > 0:
        med_distinct     = int(meds["MEDICATION_CD"].nunique())
    else:
        med_distinct = 0

    med_infusion_mean    = med_distinct / max(hour, 1)
    med_admin_per_icu_hour = med_distinct / max(hour, 1)
    med_per_clinical_event = med_distinct / max(clinical_events_per_icu_hour, 0.001)

    # ── Derived / interaction features ────────────────────────────────────────
    infusion_x_scai_severity = med_infusion_mean * current_scai
    infusion_early_above_mean = 1 if med_infusion_mean > 2.0 else 0
    med_distinct_bucket       = min(int(med_distinct / 3), 4)

    row = {
        "med_distinct":                       med_distinct,
        "med_infusion_mean":                  med_infusion_mean,
        "proc_n":                             0,
        "proc_distinct_nom":                  0,
        "current_scai":                       current_scai,
        "scai_prop_ge3":                      scai_prop_ge3,
        "scai_std":                           scai_std,
        "scai_slope_12h":                     scai_slope_12h,
        "scai_max":                           scai_max,
        "scai_increase_count":                scai_increase_count,
        "inf_early_minus_all":                0.0,
        "infusion_early_above_mean":          infusion_early_above_mean,
        "clinical_events_last_4h":            clinical_events_last_4h,
        "clinical_events_per_icu_hour":       clinical_events_per_icu_hour,
        "med_admin_per_icu_hour":             med_admin_per_icu_hour,
        "proc_per_icu_hour":                  0.0,
        "med_per_clinical_event":             med_per_clinical_event,
        "proc_n_bucket":                      0,
        "med_distinct_bucket":                med_distinct_bucket,
        "med_residual_within_scai":           0.0,
        "proc_residual_within_scai":          0.0,
        "clinical_event_residual_within_scai": 0.0,
        "infusion_x_scai_severity":           infusion_x_scai_severity,
        "demo_profile_bucket":                1,
        "age_years":                          age,
        "frailty_index":                      0.2,
    }

    df = pd.DataFrame([row])

    # Use the feature list baked into the bundle to guarantee correct column order
    bundle = loader.get("mortality")
    feature_cols = bundle["feature_names"] if isinstance(bundle, dict) else MORTALITY_FEATURES

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing)], axis=1)

    return df[feature_cols]
