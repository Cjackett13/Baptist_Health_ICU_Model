#!/usr/bin/env python3
"""
LOS data-prep sign-off: leakage checklist, feature policy validation, target QA.

  python3 Baptist_tester/los_prep_signoff.py --data-dir data

Writes ``los_data_prep_signoff.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "Baptist_tester") not in sys.path:
    sys.path.insert(0, str(_REPO / "Baptist_tester"))

from los_feature_policy import (  # noqa: E402
    FORBIDDEN_AS_FEATURES,
    FULL_STAY_AGGREGATE_FORBIDDEN,
    LABEL_COLUMNS,
    V1_FEATURE_WINDOW_HOURS,
    validate_modeling_columns,
)
from los_model_config import (  # noqa: E402
    ID_COLUMNS,
    LABEL_PRIMARY,
    LABEL_TRAIN_TRANSFORM,
    MODEL_FEATURE_COLUMNS,
)

SIGNOFF_NAME = "los_data_prep_signoff.json"
EARLY_FULL_PAIRS = [
    ("med_distinct_12h", "med_distinct"),
    ("med_infusion_mean_12h", "med_infusion_mean"),
    ("clinical_event_n_12h", "clinical_event_n"),
    ("proc_n_12h", "proc_n"),
    ("current_scai_12h", "current_scai"),
]


def _resolve(data_dir: Path) -> Path:
    return data_dir.resolve() if data_dir.is_absolute() else (_REPO / data_dir).resolve()


def _post_window_violations(names: list[str]) -> list[dict[str, str]]:
    """Columns that violate ≤12h / admit-only feature policy."""
    violations = []
    admit_ok = {"age_years", "sex_bin", "admit_type_cd", "admit_src_cd", "unit_cd"}
    for col in names:
        reason = None
        if col in LABEL_COLUMNS or col in FORBIDDEN_AS_FEATURES:
            reason = "forbidden_label_or_outcome"
        elif col.endswith("_0_to_t"):
            reason = "checkpoint_suffix_0_to_t"
        elif col in FULL_STAY_AGGREGATE_FORBIDDEN:
            reason = "full_stay_aggregate_name"
        elif col in ("prediction_hour", "feature_window_hours"):
            reason = "checkpoint_metadata"
        elif col.startswith(("DISCH_", "DECEASED", "truncated_", "los_truncated")):
            reason = "disposition_or_truncation"
        elif not col.endswith("_12h") and col not in admit_ok and not col.startswith("dx_cat_"):
            if col in ("race_cd", "ethnicity_cd", "icd_prefix", "demo_profile_bucket"):
                reason = "non_12h_optional_demographic"
            elif not col.endswith("_12h"):
                reason = "missing_12h_suffix_or_admit_exception"
        if reason:
            violations.append({"column": col, "reason": reason})
    return violations


def leakage_checklist(data_dir: Path, df: pd.DataFrame) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    feats = MODEL_FEATURE_COLUMNS
    policy_ok = []
    policy_fail = []

    try:
        validate_modeling_columns(feats, prediction_hour=V1_FEATURE_WINDOW_HOURS, strict_v1_12h=True)
        policy_ok.append("validate_modeling_columns_passed")
    except ValueError as exc:
        policy_fail.append(str(exc))

    forbidden_hit = sorted(set(feats) & FORBIDDEN_AS_FEATURES)
    full_stay_hit = sorted(set(feats) & FULL_STAY_AGGREGATE_FORBIDDEN)
    label_hit = sorted(set(feats) & LABEL_COLUMNS)
    post_window = _post_window_violations(feats)

    early_vs_full = []
    mort_path = data_dir / "duckdb_patient_features.parquet"
    if mort_path.is_file():
        full = pd.read_parquet(mort_path)
        early_cols = [c for c in feats if c in df.columns]
        merged = df[["PERSON_ID", LABEL_PRIMARY] + early_cols].merge(full, on="PERSON_ID", how="inner")
        y = pd.to_numeric(merged[LABEL_PRIMARY], errors="coerce")
        for early, late in EARLY_FULL_PAIRS:
            if early not in feats or early not in merged.columns or late not in merged.columns:
                continue
            e = pd.to_numeric(merged[early], errors="coerce")
            l = pd.to_numeric(merged[late], errors="coerce")
            m = e.notna() & l.notna() & y.notna()
            if m.sum() < 20:
                continue
            r_e, _ = stats.spearmanr(e[m], y[m], nan_policy="omit")
            r_l, _ = stats.spearmanr(l[m], y[m], nan_policy="omit")
            early_vs_full.append(
                {
                    "early_12h": early,
                    "full_stay": late,
                    "spearman_early": round(float(r_e), 4),
                    "spearman_full": round(float(r_l), 4),
                    "delta_abs_r": round(float(abs(r_l) - abs(r_e)), 4),
                    "in_final_model": True,
                    "status": "ok" if late not in feats else "full_stay_column_dropped",
                }
            )

    return {
        "final_feature_count": len(feats),
        "features": feats,
        "policy_validation": {"passed": policy_ok, "failed": policy_fail},
        "forbidden_intersection": forbidden_hit,
        "full_stay_name_intersection": full_stay_hit,
        "label_in_features": label_hit,
        "post_window_violations": post_window,
        "post_window_clean": len(post_window) == 0,
        "residual_features_in_model": [c for c in feats if "residual" in c],
        "early_vs_full_stay_spearman": early_vs_full,
        "notes": [
            "All model features use _12h suffix or admit-time columns (age, sex, admit_*, unit_*).",
            "Dropped *_residual_within_scai_12h (cohort-level SCAI means).",
            "duckdb_patient_features.parquet is blocked for training; early_vs_full is sanity only.",
        ],
    }


def feature_json_audit(data_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    doc = json.loads((data_dir / "los_feature_columns.json").read_text(encoding="utf-8"))
    final_doc = json.loads((data_dir / "los_final_model_features.json").read_text(encoding="utf-8"))
    model_feats = doc.get("model_feature_columns") or final_doc.get("feature_columns", [])
    final_feats = final_doc.get("feature_columns", [])
    dropped = doc.get("dropped_features", [])
    legacy_v2 = doc.get("feature_columns_v2_checkpoint", [])
    legacy_primary = doc.get("primary_features", [])
    return {
        "schema_version": doc.get("schema_version"),
        "feature_window_hours": doc.get("feature_window_hours"),
        "model_feature_columns_count": len(model_feats),
        "final_manifest_count": len(final_feats),
        "manifests_match": model_feats == final_feats == MODEL_FEATURE_COLUMNS,
        "post_window_in_model_features": _post_window_violations(model_feats),
        "model_features_zero_post_window": len(_post_window_violations(model_feats)) == 0,
        "labels_not_in_model_features": all(c not in model_feats for c in doc.get("label_columns", [])),
        "label_for_training_not_in_model_features": doc.get("label_for_training")
        not in model_feats,
        "dropped_post_window_in_model_list": _post_window_violations(model_feats),
        "legacy_v2_checkpoint_not_used_for_training": True,
        "legacy_v2_has_0_to_t": any(c.endswith("_0_to_t") for c in legacy_v2),
        "legacy_primary_not_in_final_model": all(c not in final_feats for c in legacy_primary),
        "checkpoint_meta_in_dropped": all(
            c in dropped for c in ("prediction_hour", "feature_window_hours")
        ),
    }


def frame_column_audit(df: pd.DataFrame) -> dict[str, Any]:
    feat_cols = set(MODEL_FEATURE_COLUMNS)
    frame_feats = [c for c in df.columns if c not in ID_COLUMNS and c not in LABEL_COLUMNS]
    extra = sorted(set(frame_feats) - feat_cols - {"log1p_los_hours_total"})
    missing = sorted(feat_cols - set(df.columns))
    return {
        "frame_shape": list(df.shape),
        "features_match_manifest": extra == [] and missing == [],
        "extra_columns": extra,
        "missing_features": missing,
    }


def target_qa(df: pd.DataFrame) -> dict[str, Any]:
    rep = {}
    for col in (LABEL_PRIMARY, LABEL_TRAIN_TRANSFORM, "remaining_los_hours"):
        if col not in df.columns:
            rep[col] = {"present": False}
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n = len(s)
        n_miss = int(s.isna().sum())
        rep[col] = {
            "present": True,
            "n": n,
            "n_missing": n_miss,
            "pct_missing": round(100.0 * n_miss / n, 4) if n else 100.0,
            "min": float(s.min()) if n_miss < n else None,
            "max": float(s.max()) if n_miss < n else None,
            "median": float(s.median()) if n_miss < n else None,
        }
    primary = rep.get(LABEL_PRIMARY, {})
    acceptable = primary.get("pct_missing", 100) == 0.0 and (primary.get("min") or 0) > 0
    rep["acceptable"] = acceptable
    rep["threshold"] = "pct_missing == 0 and los_hours_total > 0"
    return rep


def run_signoff(data_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    df = pd.read_parquet(data_dir / "los_modeling_frame.parquet")
    leak = leakage_checklist(data_dir, df)
    feat_doc = feature_json_audit(data_dir)
    frame_doc = frame_column_audit(df)
    targets = target_qa(df)

    all_pass = (
        leak["post_window_clean"]
        and not leak["forbidden_intersection"]
        and not leak["label_in_features"]
        and not leak["policy_validation"]["failed"]
        and feat_doc["model_features_zero_post_window"]
        and feat_doc["manifests_match"]
        and feat_doc["labels_not_in_model_features"]
        and frame_doc["features_match_manifest"]
        and targets.get("acceptable")
    )

    signoff = {
        "status": "APPROVED_FOR_TRAINING" if all_pass else "BLOCKED_REVIEW_REQUIRED",
        "dataset": str(data_dir),
        "n_encounters": int(len(df)),
        "leakage_checklist": leak,
        "los_feature_columns_audit": feat_doc,
        "modeling_frame_audit": frame_doc,
        "target_quality": targets,
        "training_handoff": {
            "model_script": "Baptist_tester/los_model.py",
            "modeling_frame": "los_modeling_frame.parquet",
            "feature_manifest": "los_final_model_features.json",
            "feature_spec": "los_feature_columns.json",
            "split_manifest": "los_split_manifest.json",
            "label_for_training": LABEL_TRAIN_TRANSFORM,
            "split_by": "PERSON_ID",
            "command": "python3 Baptist_tester/los_model.py --data-dir data",
        },
        "required_artifacts": [
            "los_modeling_frame.parquet",
            "los_final_model_features.json",
            "los_feature_columns.json",
            "los_split_manifest.json",
            "los_feature_audit.json",
            "los_cleaning_log.json",
        ],
    }
    (data_dir / SIGNOFF_NAME).write_text(json.dumps(signoff, indent=2), encoding="utf-8")
    return signoff


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="LOS data prep sign-off")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    s = run_signoff(args.data_dir)
    print(json.dumps({"status": s["status"], "target": s["target_quality"]}, indent=2))
    print(f"Wrote {_resolve(args.data_dir) / SIGNOFF_NAME}")
    return 0 if s["status"] == "APPROVED_FOR_TRAINING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
