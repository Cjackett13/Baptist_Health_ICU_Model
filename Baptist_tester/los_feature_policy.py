"""
LOS feature / label guardrails — single source of truth for prep and training.

Import ``validate_modeling_columns`` before fitting; use ``FEATURE_COLUMNS_ALLOWED``
when building X. See ``los_prediction_policy.json`` for narrative policy.
"""

from __future__ import annotations

from typing import Any

# --- Prediction schedule (v2 rolling) ---
PREDICTION_CADENCE_HOURS = 24
FIRST_PREDICTION_HOUR = 24
V1_FEATURE_WINDOW_HOURS = 12  # legacy single-row snapshot (features through 12h only)

# --- Labels (both predicted at each checkpoint; neither may appear in X) ---
LABEL_TOTAL = "los_hours_total"
LABEL_REMAINING = "remaining_los_hours"
LABEL_COLUMNS = frozenset(
    {
        LABEL_TOTAL,
        LABEL_REMAINING,
        "los_hours",
        "log1p_los_hours",
        "log1p_los_hours_total",
        "log1p_remaining_los_hours",
    }
)

# --- Label-derived / outcome columns: audit or y only, never X ---
LABEL_DERIVED_FORBIDDEN_AS_FEATURES: frozenset[str] = frozenset(
    {
        # Raw encounter outcome fields
        "DISCH_DT_TM",
        "DISCH_DISPOSITION_CD",
        "LOS_HOURS",
        "los_hours",
        "los_hours_reconciled",
        "los_hours_calc",
        "los_reconcile_delta_h",
        "los_reconcile_ok",
        "DECEASED_DT_TM",
        # Transforms of labels
        "log1p_los_hours",
        "log1p_los_hours_total",
        "log1p_remaining_los_hours",
        LABEL_TOTAL,
        LABEL_REMAINING,
        # Truncation flags (derived from final disposition / death)
        "truncated_death",
        "truncated_hospice",
        "truncated_ama",
        "los_truncated",
        "truncated_disposition_cd",
        # Synthetic admit-time LOS index (same formula as label assignment; not for modeling)
        "admit_los_index_h",
        "ADMIT_LOS_INDEX_H",
    }
)

# --- Full-stay aggregates (duckdb mortality sidecar + unsuffixed names) ---
FULL_STAY_AGGREGATE_FORBIDDEN: frozenset[str] = frozenset(
    {
        "med_distinct",
        "med_infusion_mean",
        "med_admin_rows",
        "proc_n",
        "proc_distinct_nom",
        "current_scai",
        "scai_prop_ge3",
        "scai_std",
        "infusion_early_over_all_ratio",
        "clinical_event_n",
        "med_per_clinical_event",
        "proc_n_bucket",
        "med_distinct_bucket",
        "med_residual_within_scai",
        "infusion_residual_within_scai",
        "proc_residual_within_scai",
        "inf_mean_all",
        "died",
    }
)

# --- Identifiers / PHI ---
IDENTIFIER_FORBIDDEN_AS_FEATURES: frozenset[str] = frozenset(
    {
        "ENCOUNTER_ID",
        "PERSON_ID",
        "NAME_LAST_TXT",
        "NAME_FIRST_TXT",
    }
)

# Race/ethnicity: stratified evaluation only — never train on these directly.
DEMOGRAPHIC_EVAL_ONLY_COLUMNS: frozenset[str] = frozenset(
    {
        "race_cd",
        "ethnicity_cd",
        "demo_profile_bucket",
    }
)

DEMOGRAPHIC_TRAINING_FORBIDDEN: frozenset[str] = DEMOGRAPHIC_EVAL_ONLY_COLUMNS

FORBIDDEN_AS_FEATURES: frozenset[str] = (
    LABEL_DERIVED_FORBIDDEN_AS_FEATURES
    | FULL_STAY_AGGREGATE_FORBIDDEN
    | IDENTIFIER_FORBIDDEN_AS_FEATURES
    | DEMOGRAPHIC_TRAINING_FORBIDDEN
)

# v1 admit snapshot uses _12h suffix (valid only when prediction_hour <= 12)
V1_WINDOW_SUFFIX = "_12h"
V1_PREDICTION_HOUR = 12

# v2 rolling: feature aggregates through min(prediction_hour, stay) — not yet in DuckDB SQL.
ROLLING_FEATURE_MODE = "v1_static_12h"  # set to "rolling" when los_duckdb supports dynamic suffixes


def feature_window_hours_for_prediction(prediction_hour: int) -> int:
    """Hours of data included when building features for a checkpoint."""
    if ROLLING_FEATURE_MODE == "v1_static_12h":
        return V1_FEATURE_WINDOW_HOURS
    return max(V1_FEATURE_WINDOW_HOURS, int(prediction_hour))


def aggregate_suffix_for_prediction_hour(prediction_hour: int) -> str:
    """
    Column suffix for time-censored aggregates.

    v1: always ``_12h`` (first-shift snapshot). v2 (future): ``_{h}h`` or ``_0_to_t``.
    """
    if ROLLING_FEATURE_MODE == "v1_static_12h":
        return V1_WINDOW_SUFFIX
    h = max(V1_FEATURE_WINDOW_HOURS, int(prediction_hour))
    return f"_{h}h"

# Checkpoint metadata allowed in X
CHECKPOINT_META_FEATURES: frozenset[str] = frozenset(
    {
        "prediction_hour",
        "hours_elapsed",
        "feature_window_hours",
    }
)

EARLY_WINDOW_PER_TABLE: dict[str, dict[str, str]] = {
    "encounter": {
        "time_column": "REG_DT_TM (admit anchor)",
        "window_rule": "Static at admit only: ADMIT_TYPE_CD, ADMIT_SRC_CD, UNIT_CD, FACILITY_CD. "
        "Never use DISCH_DT_TM or DISCH_DISPOSITION_CD as features.",
        "allowed_features": "admit_type_cd, admit_src_cd, unit_cd (encoded at admit)",
    },
    "person": {
        "time_column": "BIRTH_DT_TM",
        "window_rule": "Demographics known at admit; age computed vs REG_DT_TM.",
        "allowed_features": "age_years, sex_bin, race_cd, ethnicity_cd",
    },
    "diagnosis": {
        "time_column": "DIAGNOSIS_DT_TM",
        "window_rule": "Include rows where DIAGNOSIS_DT_TM <= REG_DT_TM + prediction_hour.",
        "allowed_features": "icd_prefix, n_diagnoses_0_to_t, principal_icd_0_to_t",
    },
    "clinical_event": {
        "time_column": "EVENT_END_DT_TM",
        "window_rule": "EVENT_END_DT_TM <= REG_DT_TM + prediction_hour. "
        "Pivot vitals/labs to mean_*_0_to_t.",
        "allowed_features": "mean_HR_0_to_t, mean_MAP_0_to_t, clinical_event_n_0_to_t, ...",
    },
    "medication_admin": {
        "time_column": "ADMIN_START_DT_TM",
        "window_rule": "ADMIN_START_DT_TM <= REG_DT_TM + prediction_hour.",
        "allowed_features": "med_distinct_0_to_t, med_infusion_mean_0_to_t",
    },
    "procedure_event": {
        "time_column": "PROC_START_DT_TM",
        "window_rule": "PROC_START_DT_TM <= REG_DT_TM + prediction_hour.",
        "allowed_features": "proc_n_0_to_t",
    },
    "scai_stage_hourly": {
        "time_column": "HOUR_FROM_ADMIT / EVENT_DT_TM",
        "window_rule": "HOUR_FROM_ADMIT <= prediction_hour.",
        "allowed_features": "scai_first_0_to_t, scai_last_0_to_t, scai_mean_0_to_t, scai_std_0_to_t, scai_slope_0_to_t",
    },
    "duckdb_patient_features": {
        "time_column": "N/A (mortality full-stay sidecar)",
        "window_rule": "BLOCK for LOS — mortality aggregates leak future stay.",
        "allowed_features": "none",
    },
    "duckdb_los_features": {
        "time_column": "≤12h from REG_DT_TM (v1 snapshot)",
        "window_rule": "All columns suffixed _12h or admit-time flags; built by los_duckdb_features.py.",
        "allowed_features": "demo_profile_bucket, med_*_12h, scai_*_12h, proc_*_12h, dx_cat_*, residuals *_12h",
    },
}


def build_policy_dict() -> dict[str, Any]:
    return {
        "version": 2,
        "prediction_schedule": {
            "cadence_hours": PREDICTION_CADENCE_HOURS,
            "first_prediction_hour": FIRST_PREDICTION_HOUR,
            "feature_window_rule": "For checkpoint T: use data with hours_from_admit <= T (per-table time columns below).",
        },
        "labels": {
            "los_hours_total": "Total ICU length of stay in hours (same value on all checkpoints for an encounter).",
            "remaining_los_hours": "max(0, los_hours_total - prediction_hour).",
            "both_are_targets": True,
            "neither_may_be_features": True,
        },
        "rolling_feature_mode": ROLLING_FEATURE_MODE,
        "v1_prediction_hour": V1_PREDICTION_HOUR,
        "demographic_eval_only": sorted(DEMOGRAPHIC_EVAL_ONLY_COLUMNS),
        "forbidden_as_features": {
            "label_derived": sorted(LABEL_DERIVED_FORBIDDEN_AS_FEATURES),
            "full_stay_aggregates": sorted(FULL_STAY_AGGREGATE_FORBIDDEN),
            "identifiers": sorted(IDENTIFIER_FORBIDDEN_AS_FEATURES),
            "demographic_training_forbidden": sorted(DEMOGRAPHIC_TRAINING_FORBIDDEN),
            "combined": sorted(FORBIDDEN_AS_FEATURES),
        },
        "early_window_per_table": EARLY_WINDOW_PER_TABLE,
        "train_test_split": {
            "rule": "Split by PERSON_ID; all checkpoints for a patient share the same fold.",
        },
    }


def validate_modeling_columns(
    feature_columns: list[str],
    *,
    prediction_hour: int | None = None,
    strict_v1_12h: bool = False,
) -> list[str]:
    """
    Return ``feature_columns`` minus forbidden / label / full-stay columns.

    Raises ``ValueError`` if any forbidden column is present.
    """
    bad = [c for c in feature_columns if c in FORBIDDEN_AS_FEATURES]
    if bad:
        raise ValueError(f"Forbidden feature columns blocked: {bad}")

    demo = [c for c in feature_columns if c in DEMOGRAPHIC_TRAINING_FORBIDDEN]
    if demo:
        raise ValueError(
            f"Demographic columns are eval-only (use subgroup MAE), not training features: {demo}"
        )

    label_in_x = [c for c in feature_columns if c in LABEL_COLUMNS]
    if label_in_x:
        raise ValueError(f"Label columns cannot be features: {label_in_x}")

    if strict_v1_12h and prediction_hour is not None and prediction_hour > 12:
        stale = [c for c in feature_columns if c.endswith(V1_WINDOW_SUFFIX)]
        if stale:
            raise ValueError(
                f"Columns with {V1_WINDOW_SUFFIX!r} invalid when prediction_hour={prediction_hour}: {stale}"
            )

    return [c for c in feature_columns if c not in FORBIDDEN_AS_FEATURES and c not in LABEL_COLUMNS]


def assert_frame_separated(
    df_columns: list[str],
    feature_columns: list[str],
    *,
    label_columns: list[str] | None = None,
) -> None:
    """Ensure labels and forbidden columns are not in the feature list."""
    validate_modeling_columns(feature_columns)
    labels = list(label_columns or [LABEL_TOTAL, LABEL_REMAINING])
    overlap = set(feature_columns) & set(labels)
    if overlap:
        raise ValueError(f"Features overlap labels: {overlap}")
    leaked = set(feature_columns) & LABEL_DERIVED_FORBIDDEN_AS_FEATURES
    if leaked:
        raise ValueError(f"Label-derived columns in features: {leaked}")
    full_stay = set(feature_columns) & FULL_STAY_AGGREGATE_FORBIDDEN
    if full_stay:
        raise ValueError(f"Full-stay aggregates in features: {full_stay}")
