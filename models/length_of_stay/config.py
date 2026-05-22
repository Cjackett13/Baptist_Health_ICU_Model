"""Paths and constants for the LOS model (not mortality)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BAPTIST_TESTER = REPO_ROOT / "Baptist_tester"

DATA_DIR_DEFAULT = REPO_ROOT / "data"
ARTIFACT_DIR_DEFAULT = DATA_DIR_DEFAULT / "los_model"

LABEL_TRAIN = "log1p_los_hours_total"
LABEL_EVAL = "los_hours_total"
TARGET_EDA = "los_hours_total"

BUNDLE_NAME = "xgb_los_pipeline.joblib"
META_NAME = "xgb_los_model_meta.json"
PLAN_NAME = "los_training_feature_plan.json"
SPLIT_NAME = "los_split_manifest.json"
HYPERPARAMS_NAME = "xgb_los_best_hyperparams.json"

MODEL_TASK = "length_of_stay_regression"
MORTALITY_MODEL_PATH = "Frontend/lib/models/mortality_model.py"

# User performance goals (held-out test set, hours unless noted)
PERFORMANCE_TARGETS = {
    "mae_hours_max": 36.0,  # 1.5 days
    "rmse_hours_max": 48.0,  # 2.0 days
    "r2_hours_min": 0.65,
    "mae_days_max": 1.5,
    "rmse_days_max": 2.0,
}
