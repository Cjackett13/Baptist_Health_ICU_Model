#!/usr/bin/env python3
"""
Train XGBoost regressor for ICU length of stay (LOS).

Uses the 14-feature manifest from ``Baptist_tester/los_model_config.py`` (not the exploratory plan).

  python3 -m models.length_of_stay.train --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from models.length_of_stay.config import (
    ARTIFACT_DIR_DEFAULT,
    BUNDLE_NAME,
    DATA_DIR_DEFAULT,
    HYPERPARAMS_NAME,
    LABEL_EVAL,
    LABEL_TRAIN,
    META_NAME,
    MODEL_TASK,
    PERFORMANCE_TARGETS,
    REPO_ROOT,
    SPLIT_NAME,
)
from models.length_of_stay.features import (
    assert_parquet_has_manifest_columns,
    feature_matrix_columns,
    load_training_feature_spec,
    manifest_feature_columns,
)
from models.length_of_stay.transforms import (
    RESIDUAL_AUX_COLUMNS,
    SCAI_PCA_INPUT_COLUMNS,
    SCAI_PCA_OUTPUT_COLUMNS,
    LosTrainPreprocessor,
)

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.032,
    "subsample": 0.94,
    "colsample_bytree": 0.72,
    "colsample_bylevel": 0.86,
    "reg_lambda": 1.2,
    "reg_alpha": 0.04,
    "min_child_weight": 0.5,
    "gamma": 0.03,
    "max_delta_step": 1.0,
}


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _base_columns_for_training(included: list[str], *, use_scai_pca: bool) -> list[str]:
    """Manifest columns only; optional PCA outputs when explicitly enabled."""
    if use_scai_pca:
        keep = [c for c in included if c not in SCAI_PCA_INPUT_COLUMNS]
        for pc in SCAI_PCA_OUTPUT_COLUMNS:
            if pc not in keep:
                keep.append(pc)
        return keep
    return list(included)


def load_training_frame(data_dir: Path) -> pd.DataFrame:
    feat = pd.read_parquet(data_dir / "los_modeling_features.parquet")
    labels = pd.read_parquet(data_dir / "los_encounter_labels.parquet")
    assert_parquet_has_manifest_columns(list(feat.columns))
    df = feat.merge(
        labels[["ENCOUNTER_ID", "PERSON_ID", LABEL_EVAL, LABEL_TRAIN]],
        on="ENCOUNTER_ID",
        how="inner",
        suffixes=("", "_lbl"),
    )
    if LABEL_TRAIN not in df.columns or df[LABEL_TRAIN].isna().all():
        df[LABEL_TRAIN] = np.log1p(pd.to_numeric(df[LABEL_EVAL], errors="coerce").clip(lower=0))
    return df


def split_by_manifest(df: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((_resolve(data_dir) / SPLIT_NAME).read_text(encoding="utf-8"))
    tr = df[df["PERSON_ID"].isin(manifest["train_person_ids"])].copy()
    va = df[df["PERSON_ID"].isin(manifest["val_person_ids"])].copy()
    te = df[df["PERSON_ID"].isin(manifest["test_person_ids"])].copy()
    return tr, va, te


def _feature_matrix(df: pd.DataFrame, included: list[str]) -> pd.DataFrame:
    need = feature_matrix_columns(included)
    cols = [c for c in df.columns if c in need]
    return df[cols + [LABEL_TRAIN, LABEL_EVAL]].copy()


def _baseline_mae_hours(y_true_hours: np.ndarray) -> float:
    med = float(np.median(y_true_hours))
    return float(np.mean(np.abs(y_true_hours - med)))


def _metrics(y_true_hours: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    pred_h = np.expm1(np.clip(pred_log, 0, None))
    mae_h = float(mean_absolute_error(y_true_hours, pred_h))
    rmse_h = float(np.sqrt(mean_squared_error(y_true_hours, pred_h)))
    r2_h = float(r2_score(y_true_hours, pred_h))
    baseline_mae = _baseline_mae_hours(y_true_hours)
    within_24 = float(np.mean(np.abs(y_true_hours - pred_h) <= 24.0))
    within_36 = float(np.mean(np.abs(y_true_hours - pred_h) <= 36.0))
    within_48 = float(np.mean(np.abs(y_true_hours - pred_h) <= 48.0))
    return {
        "mae_hours": mae_h,
        "rmse_hours": rmse_h,
        "r2_hours": r2_h,
        "mae_days": round(mae_h / 24.0, 3),
        "rmse_days": round(rmse_h / 24.0, 3),
        "mae_log1p": float(mean_absolute_error(np.log1p(np.clip(y_true_hours, 0, None)), pred_log)),
        "baseline_mae_hours_median_predictor": baseline_mae,
        "mae_improvement_vs_baseline_pct": round(
            100.0 * (1.0 - mae_h / baseline_mae) if baseline_mae > 0 else 0.0, 2
        ),
        "pct_within_24h": round(100.0 * within_24, 2),
        "pct_within_36h": round(100.0 * within_36, 2),
        "pct_within_48h": round(100.0 * within_48, 2),
    }


def _targets_met(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "mae_hours": metrics["mae_hours"] <= PERFORMANCE_TARGETS["mae_hours_max"],
        "rmse_hours": metrics["rmse_hours"] <= PERFORMANCE_TARGETS["rmse_hours_max"],
        "r2_hours": metrics["r2_hours"] >= PERFORMANCE_TARGETS["r2_hours_min"],
    }


def _load_hyperparams(model_dir: Path) -> dict[str, Any] | None:
    path = model_dir / HYPERPARAMS_NAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("xgb_params") if isinstance(payload, dict) else None


def train_los_model(
    data_dir: Path,
    *,
    model_dir: Path | None = None,
    n_estimators: int = 1200,
    early_stopping_rounds: int = 50,
    xgb_overrides: dict[str, Any] | None = None,
    use_feature_weights: bool = True,
    use_scai_pca: bool = False,
) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir or (data_dir / "los_model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    included, weight_map = load_training_feature_spec(
        model_dir, data_dir, use_plan_weights=use_feature_weights
    )
    base_cols = _base_columns_for_training(included, use_scai_pca=use_scai_pca)
    df = load_training_frame(data_dir)
    tr, va, te = split_by_manifest(df, data_dir)

    def _xy(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        X = _feature_matrix(split_df, included)
        y = X.pop(LABEL_TRAIN)
        y_h = X.pop(LABEL_EVAL)
        return X, y, y_h

    X_train, y_train, y_train_h = _xy(tr)
    X_val, y_val, y_val_h = _xy(va)
    X_test, y_test, y_test_h = _xy(te)

    prep = LosTrainPreprocessor(
        base_feature_columns=base_cols,
        weight_by_column=weight_map if use_feature_weights else {},
        scai_pca_n_components=2 if use_scai_pca else 0,
    )
    X_tr = prep.fit_transform(X_train)
    X_va = prep.transform(X_val)
    X_te = prep.transform(X_test)
    fw = prep.get_feature_weights()

    xgb_kw = {
        **DEFAULT_XGB_PARAMS,
        "n_estimators": int(n_estimators),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "early_stopping_rounds": int(early_stopping_rounds),
    }
    tuned = _load_hyperparams(model_dir)
    if tuned:
        xgb_kw.update({k: v for k, v in tuned.items() if k != "early_stopping_rounds"})
        if "n_estimators" in tuned:
            xgb_kw["n_estimators"] = int(tuned["n_estimators"])
    if xgb_overrides:
        xgb_kw.update(xgb_overrides)

    reg = XGBRegressor(**xgb_kw, feature_weights=fw)
    reg.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
    pipe = Pipeline([("prep", prep), ("xgb", reg)])

    meta: dict[str, Any] = {
        "model_family": MODEL_TASK,
        "separate_from": "mortality_model (in-hospital death classifier)",
        "artifact_dir": str(model_dir),
        "data_dir": str(data_dir),
        "label_train": LABEL_TRAIN,
        "label_eval_hours": LABEL_EVAL,
        "metrics_type": "regression_hours",
        "feature_manifest_source": "Baptist_tester/los_model_config.MODEL_FEATURE_COLUMNS",
        "feature_manifest_n_raw": len(manifest_feature_columns()),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "feature_columns_raw_included": included,
        "feature_columns_model": list(prep.feature_names_),
        "residual_aux_columns": sorted(RESIDUAL_AUX_COLUMNS),
        "scai_pca_input_columns": SCAI_PCA_INPUT_COLUMNS if use_scai_pca else [],
        "scai_pca_explained_variance_ratio": prep.explained_variance_ratio_(),
        "use_scai_pca": use_scai_pca,
        "residual_policy": "infusion_residual_within_scai_12h fit on train fold only (dropped from final X)",
        "demographic_training_policy": "race_cd and ethnicity_cd excluded; use subgroup MAE only",
        "rolling_feature_mode": "v1_static_12h",
        "xgb_best_iteration": int(reg.best_iteration) if reg.best_iteration is not None else int(n_estimators),
        "xgb_params": {k: v for k, v in xgb_kw.items() if k != "early_stopping_rounds"},
        "xgb_feature_weights": {prep.feature_names_[i]: float(fw[i]) for i in range(len(fw))},
        "performance_targets": PERFORMANCE_TARGETS,
        "use_feature_weights": bool(use_feature_weights),
        "metrics_train": _metrics(y_train_h.to_numpy(), reg.predict(X_tr)),
        "metrics_val": _metrics(y_val_h.to_numpy(), reg.predict(X_va)),
        "metrics_test": _metrics(y_test_h.to_numpy(), reg.predict(X_te)),
    }
    meta["targets_met_test"] = _targets_met(meta["metrics_test"])
    meta["targets_met_all_test"] = all(meta["targets_met_test"].values())
    meta["performance_targets_note"] = (
        "v1 uses ≤12h snapshot features only. Enable rolling aggregates in los_duckdb_features "
        "when ROLLING_FEATURE_MODE='rolling' in los_feature_policy."
    )

    joblib.dump(pipe, model_dir / BUNDLE_NAME)
    (model_dir / "performance_targets.json").write_text(
        json.dumps(
            {
                "targets": PERFORMANCE_TARGETS,
                "metrics_test": meta["metrics_test"],
                "targets_met_test": meta["targets_met_test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Train LOS (length of stay) XGBoost regressor")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Default: <data-dir>/los_model/",
    )
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--xgb-params-json", type=Path, default=None)
    ap.add_argument(
        "--no-feature-weights",
        action="store_true",
        help="Train with uniform feature weights (ignores plan weights).",
    )
    ap.add_argument(
        "--use-scai-pca",
        action="store_true",
        help="Enable optional SCAI PCA (off by default; manifest uses raw SCAI columns).",
    )
    args = ap.parse_args()

    overrides = None
    if args.xgb_params_json is not None:
        overrides = json.loads(_resolve(args.xgb_params_json).read_text(encoding="utf-8"))

    model_dir = _resolve(args.model_dir) if args.model_dir else _resolve(args.data_dir) / "los_model"
    meta = train_los_model(
        args.data_dir,
        model_dir=model_dir,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        xgb_overrides=overrides,
        use_feature_weights=not args.no_feature_weights,
        use_scai_pca=args.use_scai_pca,
    )
    print(
        json.dumps(
            {
                "metrics_test": meta["metrics_test"],
                "targets_met_test": meta["targets_met_test"],
                "performance_targets": meta["performance_targets"],
                "n_features_raw": meta["feature_manifest_n_raw"],
                "n_features_model": len(meta["feature_columns_model"]),
            },
            indent=2,
        )
    )
    print(f"Wrote {model_dir / BUNDLE_NAME}")
    print(f"Wrote {model_dir / META_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
