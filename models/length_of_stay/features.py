"""
LOS training feature manifest — single source of truth from ``los_model_config.py``.

Training must use ``MODEL_FEATURE_COLUMNS`` (14 raw features), not exploratory plan rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from models.length_of_stay.config import BAPTIST_TESTER, PLAN_NAME, REPO_ROOT

if str(BAPTIST_TESTER) not in sys.path:
    sys.path.insert(0, str(BAPTIST_TESTER))

from los_feature_policy import (  # noqa: E402
    DEMOGRAPHIC_EVAL_ONLY_COLUMNS,
    validate_modeling_columns,
)
from los_model_config import (  # noqa: E402
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_FEATURES,
    TRANSFORMS,
    WINSORIZE_TRAIN_PCT,
)

# Weights for raw manifest columns (not PCA); used when plan has no entry.
DEFAULT_RAW_FEATURE_WEIGHTS: dict[str, float] = {
    "current_scai_12h": 10.0,
    "scai_prop_ge3_12h": 8.0,
    "scai_slope_12h": 6.0,
    "med_infusion_mean_12h": 5.0,
    "clinical_event_n_12h": 4.0,
    "med_distinct_12h": 3.0,
    "proc_n_12h": 3.0,
    "med_per_clinical_event_12h": 3.0,
    "dx_cat_arrhythmia_arrest": 2.0,
    "age_years": 1.0,
    "sex_bin": 1.0,
    "admit_type_cd": 1.0,
    "admit_src_cd": 1.0,
    "unit_cd": 2.0,
}

LOG1P_COLUMNS: list[str] = [c for c, t in TRANSFORMS.items() if t == "log1p"]
SQRT_COLUMNS: list[str] = [c for c, t in TRANSFORMS.items() if t == "sqrt"]
CATEGORICAL_COLUMNS: list[str] = list(MODEL_CATEGORICAL_FEATURES)
WINSORIZE_COLUMNS: list[str] = list(WINSORIZE_TRAIN_PCT)

# Columns required in parquet to engineer residuals (not in final X)
RESIDUAL_AUX_COLUMNS: frozenset[str] = frozenset(
    {"med_infusion_mean_12h", "current_scai_12h"}
)


def manifest_feature_columns() -> list[str]:
    """Return validated 14-feature manifest (numeric + categorical)."""
    cols = list(MODEL_FEATURE_COLUMNS)
    validate_modeling_columns(cols)
    blocked = set(cols) & DEMOGRAPHIC_EVAL_ONLY_COLUMNS
    if blocked:
        raise ValueError(f"Manifest must not include eval-only demographics: {sorted(blocked)}")
    return cols


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_optional_plan_weights(model_dir: Path, data_dir: Path) -> dict[str, float]:
    """Merge plan weights only for manifest columns; ignore exploratory plan features."""
    weights = dict(DEFAULT_RAW_FEATURE_WEIGHTS)
    plan_path = _resolve(model_dir) / PLAN_NAME
    if not plan_path.is_file():
        src = _resolve(data_dir) / PLAN_NAME
        if src.is_file():
            plan_path = src
        else:
            return weights
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    allowed = set(MODEL_FEATURE_COLUMNS)
    for row in plan.get("features_included", []):
        feat = row.get("feature")
        if feat not in allowed:
            continue
        w = row.get("planned_xgb_feature_weight")
        if w is not None:
            weights[feat] = float(w)
    return weights


def load_training_feature_spec(
    model_dir: Path,
    data_dir: Path,
    *,
    use_plan_weights: bool = True,
) -> tuple[list[str], dict[str, float]]:
    included = manifest_feature_columns()
    weight_map = load_optional_plan_weights(model_dir, data_dir) if use_plan_weights else {}
    return included, weight_map


def feature_matrix_columns(included: list[str]) -> set[str]:
    return set(included) | RESIDUAL_AUX_COLUMNS


def raw_matrix_for_pipeline(df: pd.DataFrame, included: list[str]) -> pd.DataFrame:
    """Columns passed into LosTrainPreprocessor (manifest + residual aux only)."""
    need = feature_matrix_columns(included)
    cols = [c for c in df.columns if c in need]
    return df[cols].copy()


# Must never appear in feature_columns_raw_included (post-retrain audit).
FORBIDDEN_MANIFEST_VIOLATIONS: frozenset[str] = frozenset(
    {
        "race_cd",
        "ethnicity_cd",
        "demo_profile_bucket",
        "scai_mean_12h",
        "scai_last_12h",
        "scai_first_12h",
        "scai_std_12h",
        "med_admin_rows_12h",
        "los_hours_total",
        "remaining_los_hours",
        "log1p_los_hours_total",
        "admit_los_index_h",
    }
)

EXPECTED_RAW_FEATURE_COUNT = len(MODEL_FEATURE_COLUMNS)


def assert_parquet_has_manifest_columns(df_columns: list[str]) -> None:
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in df_columns]
    if missing:
        raise ValueError(
            f"Training frame missing manifest columns: {missing}. "
            "Re-run los_publish_artifacts.py / los_data_prep.py."
        )
