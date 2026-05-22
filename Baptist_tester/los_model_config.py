"""
Final LOS v1 model feature manifest (≤12h snapshot, encounter grain).

Used by ``los_publish_artifacts.py`` and ``models.length_of_stay.train``.

v1: features censored at 12h (``_12h`` suffix) at prediction_hour=12.
v2 rolling: set ``ROLLING_FEATURE_MODE='rolling'`` in ``los_feature_policy`` and rebuild DuckDB sidecar.
"""

from __future__ import annotations

from typing import Any

LABEL_PRIMARY = "los_hours_total"
LABEL_SECONDARY = "remaining_los_hours"
LABEL_TRAIN_TRANSFORM = "log1p_los_hours_total"

ID_COLUMNS = ["ENCOUNTER_ID", "PERSON_ID"]

# Raw columns fed to the training pipeline (before one-hot expansion)
MODEL_NUMERIC_FEATURES: list[str] = [
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
]

MODEL_CATEGORICAL_FEATURES: list[str] = [
    "admit_type_cd",
    "admit_src_cd",
    "unit_cd",
]

MODEL_FEATURE_COLUMNS: list[str] = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES

DROPPED_FEATURES: list[str] = [
    "infusion_mean_12h",
    "med_admin_rows_12h",
    "proc_distinct_nom_12h",
    "scai_first_12h",
    "scai_last_12h",
    "scai_mean_12h",
    "scai_std_12h",
    "med_distinct_bucket_12h",
    "proc_n_bucket_12h",
    "demo_profile_bucket",
    "med_residual_within_scai_12h",
    "infusion_residual_within_scai_12h",
    "proc_residual_within_scai_12h",
    "admit_type_emergency",
    "admit_type_elective",
    "admit_src_ed",
    "admit_src_outside_hospital",
    "unit_cicu",
    "unit_cvicu",
    "unit_micu",
    "dx_cat_other",
    "dx_cat_acute_mi",
    "dx_cat_adhf",
    "dx_cat_myocarditis_cmp",
    "dx_cat_aortic_valve",
    "dx_cat_post_cardiotomy",
    "n_diagnoses_12h",
    "race_cd",
    "ethnicity_cd",
    "icd_prefix",
    "prediction_hour",
    "feature_window_hours",
]

TRANSFORMS: dict[str, str] = {
    "med_infusion_mean_12h": "sqrt",
    "scai_prop_ge3_12h": "none",
    "current_scai_12h": "none",
    "proc_n_12h": "sqrt",
    "med_distinct_12h": "sqrt",
    "clinical_event_n_12h": "log1p",
    "dx_cat_arrhythmia_arrest": "none",
    "med_per_clinical_event_12h": "sqrt",
    "scai_slope_12h": "none",
    "age_years": "none",
    "sex_bin": "none",
    "admit_type_cd": "one_hot",
    "admit_src_cd": "one_hot",
    "unit_cd": "one_hot",
    LABEL_PRIMARY: "none",
    LABEL_TRAIN_TRANSFORM: "log1p",
}

WINSORIZE_TRAIN_PCT: list[str] = [
    "med_infusion_mean_12h",
    "clinical_event_n_12h",
    "age_years",
    "med_per_clinical_event_12h",
]

ONE_HOT_FEATURES: list[str] = list(MODEL_CATEGORICAL_FEATURES)

SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def build_feature_columns_doc() -> dict[str, Any]:
    return {
        "schema_version": 4,
        "grain": "encounter",
        "feature_window_hours": 12,
        "labels": {
            LABEL_PRIMARY: "Total ICU LOS (hours); primary regression target.",
            LABEL_SECONDARY: "max(0, los_hours_total - prediction_hour); secondary target.",
            LABEL_TRAIN_TRANSFORM: "Training label transform (recommended).",
        },
        "label_columns": [LABEL_PRIMARY, LABEL_SECONDARY],
        "label_for_training": LABEL_TRAIN_TRANSFORM,
        "id_columns": ID_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "model_numeric_features": MODEL_NUMERIC_FEATURES,
        "model_categorical_features": MODEL_CATEGORICAL_FEATURES,
        "one_hot_features": ONE_HOT_FEATURES,
        "dropped_features": DROPPED_FEATURES,
        "transforms": TRANSFORMS,
        "winsorize_train_percentile": WINSORIZE_TRAIN_PCT,
        "ordinal_columns": ["current_scai_12h"],
        "bucket_columns_dropped": ["med_distinct_bucket_12h", "proc_n_bucket_12h"],
        "imputation": {
            "when": "train_fold_only",
            "numeric": "median",
            "categorical": "mode",
        },
        "separate_manifest_file": "los_final_model_features.json",
    }
