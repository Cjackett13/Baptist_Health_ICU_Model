#!/usr/bin/env python3
"""
Hyperparameter search for LOS XGBoost (optimizes validation MAE in hours).

Writes ``<model-dir>/xgb_los_best_hyperparams.json`` then optionally retrains.

  python3 -m models.length_of_stay.tune --data-dir data --trials 40
  python3 -m models.length_of_stay.tune --data-dir data --trials 40 --retrain
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from models.length_of_stay.config import (
    DATA_DIR_DEFAULT,
    HYPERPARAMS_NAME,
    LABEL_EVAL,
    LABEL_TRAIN,
    PERFORMANCE_TARGETS,
    REPO_ROOT,
)
from models.length_of_stay.features import load_training_feature_spec
from models.length_of_stay.train import (
    DEFAULT_XGB_PARAMS,
    _base_columns_for_training,
    _feature_matrix,
    _metrics,
    _resolve,
    load_training_frame,
    split_by_manifest,
    train_los_model,
)
from models.length_of_stay.transforms import LosTrainPreprocessor

SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [4, 5, 6, 7, 8, 9],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "subsample": [0.75, 0.85, 0.92, 1.0],
    "colsample_bytree": [0.6, 0.72, 0.85, 1.0],
    "colsample_bylevel": [0.7, 0.86, 1.0],
    "reg_lambda": [0.3, 0.8, 1.2, 2.5],
    "reg_alpha": [0.0, 0.04, 0.2, 0.8],
    "min_child_weight": [0.5, 1.0, 3.0, 8.0],
    "gamma": [0.0, 0.03, 0.1, 0.3],
    "n_estimators": [600, 1000, 1500, 2000],
}


def _prepare_xy(data_dir: Path, *, use_feature_weights: bool):
    model_dir = _resolve(data_dir) / "los_model"
    included, weight_map = load_training_feature_spec(
        model_dir, data_dir, use_plan_weights=use_feature_weights
    )
    base_cols = _base_columns_for_training(included, use_scai_pca=False)
    df = load_training_frame(data_dir)
    tr, va, te = split_by_manifest(df, data_dir)

    def xy(split_df):
        X = _feature_matrix(split_df, included)
        y = X.pop(LABEL_TRAIN)
        y_h = X.pop(LABEL_EVAL)
        return X, y, y_h

    X_tr, y_tr, y_tr_h = xy(tr)
    X_va, y_va, y_va_h = xy(va)
    X_te, y_te, y_te_h = xy(te)

    prep = LosTrainPreprocessor(
        base_feature_columns=base_cols,
        weight_by_column=weight_map if use_feature_weights else {},
        scai_pca_n_components=0,
    )
    return (
        prep.fit_transform(X_tr),
        y_tr,
        prep.transform(X_va),
        y_va,
        y_va_h.to_numpy(),
        prep.transform(X_te),
        y_te_h.to_numpy(),
        prep,
    )


def _eval_params(
    X_tr,
    y_tr,
    X_va,
    y_va,
    y_va_h: np.ndarray,
    *,
    params: dict[str, Any],
    early_stopping_rounds: int,
) -> tuple[float, float, float, int]:
    kw = {
        **DEFAULT_XGB_PARAMS,
        **params,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "early_stopping_rounds": early_stopping_rounds,
    }
    reg = XGBRegressor(**kw)
    reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_log = reg.predict(X_va)
    pred_h = np.expm1(np.clip(pred_log, 0, None))
    mae = float(mean_absolute_error(y_va_h, pred_h))
    rmse = float(np.sqrt(mean_squared_error(y_va_h, pred_h)))
    r2 = float(r2_score(y_va_h, pred_h))
    best_it = int(reg.best_iteration) if reg.best_iteration is not None else int(params["n_estimators"])
    return mae, rmse, r2, best_it


def tune_los(
    data_dir: Path,
    *,
    model_dir: Path | None = None,
    trials: int = 40,
    seed: int = 42,
    early_stopping_rounds: int = 50,
    use_feature_weights: bool = False,
) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir or (data_dir / "los_model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_va, y_va, y_va_h, X_te, y_te_h, prep = _prepare_xy(
        data_dir, use_feature_weights=use_feature_weights
    )

    rng = random.Random(seed)
    best: dict[str, Any] | None = None

    for t in range(trials):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        mae, rmse, r2, best_it = _eval_params(
            X_tr,
            y_tr,
            X_va,
            y_va,
            y_va_h,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
        row = {
            "trial": t,
            "val_mae_hours": mae,
            "val_rmse_hours": rmse,
            "val_r2_hours": r2,
            "xgb_best_iteration": best_it,
            "xgb_params": params,
        }
        if best is None or mae < best["val_mae_hours"]:
            best = row

    assert best is not None
    best_params = {**DEFAULT_XGB_PARAMS, **dict(best["xgb_params"])}
    best_params["n_estimators"] = max(int(best["xgb_best_iteration"]) + 1, 50)

    reg = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
        **best_params,
    )
    reg.fit(X_tr, y_tr, verbose=False)
    test_metrics = _metrics(y_te_h, reg.predict(X_te))

    payload = {
        "performance_targets": PERFORMANCE_TARGETS,
        "use_feature_weights": use_feature_weights,
        "trials": trials,
        "seed": seed,
        "best_trial": best["trial"],
        "val_mae_hours": best["val_mae_hours"],
        "val_rmse_hours": best["val_rmse_hours"],
        "val_r2_hours": best["val_r2_hours"],
        "xgb_best_iteration": best["xgb_best_iteration"],
        "xgb_params": {k: v for k, v in best_params.items() if k != "early_stopping_rounds"},
        "metrics_test_after_tune": test_metrics,
    }
    (model_dir / HYPERPARAMS_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune LOS XGBoost hyperparameters (val MAE)")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--use-feature-weights", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="Run full train.py after tuning")
    args = ap.parse_args()

    model_dir = _resolve(args.model_dir) if args.model_dir else _resolve(args.data_dir) / "los_model"
    payload = tune_los(
        args.data_dir,
        model_dir=model_dir,
        trials=args.trials,
        seed=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
        use_feature_weights=args.use_feature_weights,
    )
    print(json.dumps(payload, indent=2))
    print(f"Wrote {model_dir / HYPERPARAMS_NAME}")

    if args.retrain:
        meta = train_los_model(
            args.data_dir,
            model_dir=model_dir,
            n_estimators=int(payload["xgb_params"].get("n_estimators", 800)),
            early_stopping_rounds=args.early_stopping_rounds,
            use_feature_weights=args.use_feature_weights,
            use_scai_pca=False,
        )
        print(json.dumps({"metrics_test": meta["metrics_test"], "targets_met_test": meta["targets_met_test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
