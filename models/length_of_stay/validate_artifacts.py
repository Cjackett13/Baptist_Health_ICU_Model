#!/usr/bin/env python3
"""
Post-retrain validation for LOS artifacts (manifest alignment, leakage guards, weights).

  python3 -m models.length_of_stay.validate_artifacts --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.length_of_stay.config import BUNDLE_NAME, DATA_DIR_DEFAULT, META_NAME, REPO_ROOT
from models.length_of_stay.features import (
    EXPECTED_RAW_FEATURE_COUNT,
    FORBIDDEN_MANIFEST_VIOLATIONS,
    load_training_feature_spec,
    manifest_feature_columns,
    raw_matrix_for_pipeline,
)
from models.length_of_stay.train import load_training_frame, split_by_manifest
from models.length_of_stay.transforms import LosTrainPreprocessor, RESIDUAL_OUT_COL


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _check_manifest_meta(meta: dict[str, Any]) -> dict[str, Any]:
    raw = meta.get("feature_columns_raw_included", [])
    model_cols = meta.get("feature_columns_model", [])
    manifest = manifest_feature_columns()
    violations = sorted(set(raw) & FORBIDDEN_MANIFEST_VIOLATIONS)
    return {
        "pass": (
            len(raw) == EXPECTED_RAW_FEATURE_COUNT
            and set(raw) == set(manifest)
            and not violations
            and meta.get("feature_manifest_n_raw") == EXPECTED_RAW_FEATURE_COUNT
        ),
        "n_raw_features": len(raw),
        "expected_raw_features": EXPECTED_RAW_FEATURE_COUNT,
        "n_model_matrix_features": len(model_cols),
        "expected_model_matrix_min": EXPECTED_RAW_FEATURE_COUNT,
        "manifest_match": set(raw) == set(manifest),
        "forbidden_in_raw": violations,
        "use_scai_pca": meta.get("use_scai_pca"),
        "demographic_policy": meta.get("demographic_training_policy"),
    }


def _check_preprocessor_train_only_leakage(
    data_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    included, weight_map = load_training_feature_spec(model_dir, data_dir)
    df = load_training_frame(data_dir)
    tr, _, te = split_by_manifest(df, data_dir)

    def _x(split: pd.DataFrame) -> pd.DataFrame:
        X = raw_matrix_for_pipeline(split, included)
        return X

    prep = LosTrainPreprocessor(
        base_feature_columns=included,
        weight_by_column=weight_map,
        scai_pca_n_components=0,
    )
    X_tr = _x(tr)
    X_te = _x(te.head(200))

    prep.fit(X_tr)
    gm_after_fit = float(prep.global_mean_)
    bins_after_fit = dict(prep.bin_means_)

    # Transform test — must not mutate stored train statistics
    _ = prep.transform(X_te)
    gm_after_transform = float(prep.global_mean_)
    bins_after_transform = dict(prep.bin_means_)

    # Adversarial: if fit=True on test, mean would shift
    prep_leaky = LosTrainPreprocessor(base_feature_columns=included, scai_pca_n_components=0)
    prep_leaky.fit(X_tr)
    gm_train = float(prep_leaky.global_mean_)
    prep_leaky._engineer(X_te, fit=True)
    gm_refit_on_test = float(prep_leaky.global_mean_)

    return {
        "pass": (
            gm_after_fit == gm_after_transform
            and bins_after_fit == bins_after_transform
            and gm_train != gm_refit_on_test
        ),
        "global_mean_after_fit": gm_after_fit,
        "global_mean_after_transform_test": gm_after_transform,
        "global_mean_if_refit_on_test": gm_refit_on_test,
        "bin_means_unchanged_after_transform": bins_after_fit == bins_after_transform,
        "residual_out_col_dropped_from_model": RESIDUAL_OUT_COL
        not in (joblib.load(model_dir / BUNDLE_NAME).named_steps["prep"].feature_names_),
    }


def _check_feature_weights_sanity(meta: dict[str, Any]) -> dict[str, Any]:
    weights = meta.get("xgb_feature_weights", {})
    high_weight_cols = [k for k, v in weights.items() if float(v) >= 5.0]
    return {
        "pass": len(high_weight_cols) >= 3,
        "n_weighted_features": len(weights),
        "high_weight_features": high_weight_cols,
        "note": "XGB feature_weights scale splits; confirm top-weight cols match clinical drivers in SHAP audit.",
    }


def run_validate_artifacts(data_dir: Path, model_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    model_dir = _resolve(model_dir)
    meta_path = model_dir / META_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "meta_path": str(meta_path),
        "metrics_test": meta.get("metrics_test"),
        "targets_met_test": meta.get("targets_met_test"),
        "manifest_alignment": _check_manifest_meta(meta),
        "preprocessor_train_only": _check_preprocessor_train_only_leakage(data_dir, model_dir),
        "feature_weights_sanity": _check_feature_weights_sanity(meta),
    }
    report["all_pass"] = all(
        report[k].get("pass") for k in ("manifest_alignment", "preprocessor_train_only", "feature_weights_sanity")
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate LOS retrained artifacts")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--write-json", type=Path, default=None)
    args = ap.parse_args()
    model_dir = _resolve(args.model_dir) if args.model_dir else _resolve(args.data_dir) / "los_model"
    report = run_validate_artifacts(args.data_dir, model_dir)
    out = args.write_json or (model_dir / "validation" / "los_artifact_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
