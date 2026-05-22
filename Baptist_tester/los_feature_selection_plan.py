#!/usr/bin/env python3
"""
Build LOS training feature plan: correlation vs ``los_hours_total``, include/exclude, XGBoost weights.

Writes ``data/los_training_feature_plan.json`` (and optional markdown).
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

PLAN_NAME = "los_training_feature_plan.json"
TARGET = "los_hours_total"
# |Spearman| below this → exclude unless clinically required
ABS_RHO_EXCLUDE = 0.02
# User policy: demographics at floor; composite bucket instead of age/sex/weight
DEMO_FEATURE_WEIGHT = 0.06  # just above 0 for race, ethnicity, demo_profile_bucket
FORCE_EXCLUDE_FEATURES: frozenset[str] = frozenset(
    {
        "age_years",
        "sex_bin",
        "admit_los_index_h",  # removed: near-duplicate of synthetic LOS assignment formula
        # weight is encoded inside demo_profile_bucket (age_bin × weight_bin × sex_bin)
    }
)
# Near-duplicate pairs: keep first, drop second
DROP_DUPLICATES = {
    "infusion_mean_12h": "duplicate_of med_infusion_mean_12h",
    "proc_distinct_nom_12h": "duplicate_of proc_n_12h",
    # med_admin_rows_12h kept — extra signal vs LOS (|ρ|≈0.24) despite correlation with med_distinct
    "unit_cicu": "redundant_with unit_cd one-hot",
    "unit_cvicu": "redundant_with unit_cd one-hot",
    "unit_micu": "constant_in_cohort",
    "admit_type_emergency": "redundant_with admit_type_cd one-hot",
    "admit_type_elective": "redundant_with admit_type_cd one-hot",
    "admit_src_ed": "redundant_with admit_src_cd one-hot",
    "admit_src_outside_hospital": "redundant_with admit_src_cd one-hot",
    "dx_cat_other": "constant_or_uninformative",
    "prediction_hour": "checkpoint_metadata",
    "feature_window_hours": "checkpoint_metadata",
}
LEAKAGE_NOTE = (
    "infusion_residual_within_scai_12h uses cohort SCAI means; optional at moderate weight "
    "or exclude for strict leakage policy."
)

# Always include for LOS v2 synthetic cohort (12h trajectory coupled to stay length)
FORCE_INCLUDE_FEATURES: frozenset[str] = frozenset(
    {
        "scai_first_12h",
        "scai_last_12h",
        "scai_mean_12h",
        "scai_std_12h",
        "scai_slope_12h",
        "med_admin_rows_12h",
        "proc_distinct_nom_12h",
        "infusion_residual_within_scai_12h",
    }
)


def _resolve(data_dir: Path) -> Path:
    return data_dir.resolve() if data_dir.is_absolute() else (_REPO / data_dir).resolve()


def _correlate_series(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    if pd.api.types.is_numeric_dtype(x):
        xv = pd.to_numeric(x, errors="coerce")
    else:
        codes = pd.Categorical(x.astype("string")).codes
        xv = pd.Series(codes, index=x.index, dtype=float).where(codes >= 0, np.nan)
    m = xv.notna() & y.notna()
    n = int(m.sum())
    if n < 30:
        return None, None, n
    try:
        rho, _ = stats.spearmanr(xv[m], y[m], nan_policy="omit")
        pear, _ = stats.pearsonr(xv[m], y[m])
    except Exception:
        return None, None, n
    if rho != rho or pear != pear:
        return None, None, n
    return float(rho), float(pear), n


def _weight_from_abs_rho(abs_rho: float, *, leakage: bool = False) -> float:
    """Map |Spearman| to XGBoost ``feature_weights`` (higher ρ → much higher weight)."""
    if leakage:
        return min(4.0, _weight_from_abs_rho(abs_rho, leakage=False))
    if abs_rho >= 0.30:
        return 10.0
    if abs_rho >= 0.25:
        return 8.0
    if abs_rho >= 0.20:
        return 6.5
    if abs_rho >= 0.15:
        return 4.5
    if abs_rho >= 0.10:
        return 3.0
    if abs_rho >= 0.05:
        return 1.2
    if abs_rho >= ABS_RHO_EXCLUDE:
        return 0.35
    return 0.15


def _tier(abs_rho: float | None) -> str:
    if abs_rho is None:
        return "constant"
    a = abs_rho
    if a >= 0.30:
        return "very_high"
    if a >= 0.20:
        return "high"
    if a >= 0.10:
        return "moderate"
    if a >= 0.05:
        return "low"
    if a >= ABS_RHO_EXCLUDE:
        return "minimal"
    return "negligible"


def _reason_include(abs_rho: float | None, feature: str) -> str:
    if feature.endswith("_cd") and (abs_rho or 0) < 0.05:
        return "Admit/demographic context; weak linear ρ but kept for one-hot signal."
    if (abs_rho or 0) >= 0.20:
        return "Strong early-window signal for LOS; up-weight heavily."
    if (abs_rho or 0) >= 0.10:
        return "Meaningful clinical association with LOS."
    if (abs_rho or 0) >= ABS_RHO_EXCLUDE:
        return "Weak but non-zero association; include at low weight."
    return "Below correlation threshold."


def build_plan(data_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    feat = pd.read_parquet(data_dir / "los_modeling_features.parquet")
    labels = pd.read_parquet(data_dir / "los_encounter_labels.parquet")[["ENCOUNTER_ID", TARGET]]
    df = feat.merge(labels, on="ENCOUNTER_ID")
    y = pd.to_numeric(df[TARGET], errors="coerce")

    col_groups = json.loads((data_dir / "los_modeling_column_groups.json").read_text(encoding="utf-8"))
    all_features = list(col_groups["feature_columns"])

    duck_audit: dict[str, Any] = {}
    duck_path = data_dir / "duckdb_los_feature_audit.json"
    if duck_path.is_file():
        duck_audit = json.loads(duck_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for feature in sorted(all_features):
        rho, pear, n_valid = _correlate_series(df[feature], y)
        abs_rho = abs(rho) if rho is not None else None
        dup_reason = DROP_DUPLICATES.get(feature)
        leakage = feature.endswith("_residual_within_scai_12h")

        include = True
        exclude_reason = None
        if feature in FORCE_EXCLUDE_FEATURES:
            include = False
            if feature == "admit_los_index_h":
                exclude_reason = "removed: synthetic LOS assignment index (leakage / not deployable)"
            elif feature in ("age_years", "sex_bin"):
                exclude_reason = "user_policy: use demo_profile_bucket instead of age/sex/weight"
            else:
                exclude_reason = "user_policy: excluded from LOS model"
        elif dup_reason and feature not in FORCE_INCLUDE_FEATURES:
            include = False
            exclude_reason = dup_reason
        elif feature in FORCE_INCLUDE_FEATURES:
            include = True
            exclude_reason = None
        elif abs_rho is None:
            include = False
            exclude_reason = "constant_or_all_missing"
        elif abs_rho < ABS_RHO_EXCLUDE and feature not in (
            "admit_type_cd",
            "admit_src_cd",
            "unit_cd",
            "race_cd",
            "ethnicity_cd",
            "demo_profile_bucket",
        ):
            include = False
            exclude_reason = f"|spearman| < {ABS_RHO_EXCLUDE} (practically zero)"
        elif feature == "icd_prefix":
            include = False
            exclude_reason = "|spearman| ≈ 0; high cardinality noise"
        elif feature == "med_residual_within_scai_12h":
            include = False
            exclude_reason = "leakage_risk + |spearman| < 0.05"

        weight = None
        weight_rationale = None
        if include:
            if feature in ("race_cd", "ethnicity_cd", "demo_profile_bucket"):
                w = DEMO_FEATURE_WEIGHT
                weight = round(w, 2)
                if feature == "demo_profile_bucket":
                    weight_rationale = (
                        "Composite age×weight×sex bucket; very low weight per user policy "
                        "(replaces age_years, sex_bin, and raw weight)."
                    )
                else:
                    weight_rationale = (
                        "Demographic one-hot; minimal influence (just above 0) per user policy."
                    )
            else:
                w = _weight_from_abs_rho(abs_rho or 0.0, leakage=leakage)
                weight = round(w, 2)
                weight_rationale = _reason_include(abs_rho, feature)
                if leakage:
                    weight_rationale += " Capped weight (leakage-sensitive residual)."

        rows.append(
            {
                "feature": feature,
                "spearman_rho": round(rho, 4) if rho is not None else None,
                "pearson_r": round(pear, 4) if pear is not None else None,
                "abs_spearman": round(abs_rho, 4) if abs_rho is not None else None,
                "tier": _tier(abs_rho),
                "n_valid": n_valid,
                "include_in_model": include,
                "exclude_reason": exclude_reason,
                "planned_xgb_feature_weight": weight,
                "weight_rationale": weight_rationale,
                "mutual_information_exploratory": duck_audit.get("mutual_information_full_cohort", {}).get(
                    feature
                ),
            }
        )

    rows.sort(key=lambda r: (-(r["abs_spearman"] or 0), r["feature"]))
    included = [r for r in rows if r["include_in_model"]]
    excluded = [r for r in rows if not r["include_in_model"]]

    return {
        "target": TARGET,
        "label_for_training": "log1p_los_hours_total",
        "n_encounters": int(len(df)),
        "correlation_note": (
            "Spearman/Pearson on full cohort (exploratory). Final weights applied on train fold only; "
            "order matches expanded design matrix after one-hot."
        ),
        "weight_policy": {
            "basis": "abs_spearman_vs_los_hours_total",
            "very_high_gte": 0.30,
            "high_gte": 0.20,
            "moderate_gte": 0.10,
            "low_gte": 0.05,
            "minimal_gte": ABS_RHO_EXCLUDE,
            "exclude_below": ABS_RHO_EXCLUDE,
            "weight_range": "0.06–10.0 (residuals capped at 4.0 if included)",
            "demographic_floor_weight": DEMO_FEATURE_WEIGHT,
        },
        "user_demographic_policy": {
            "exclude": sorted(FORCE_EXCLUDE_FEATURES),
            "include_demo_profile_bucket": True,
            "demo_profile_weight": DEMO_FEATURE_WEIGHT,
            "race_ethnicity_weight": DEMO_FEATURE_WEIGHT,
            "note": "age_years and sex_bin dropped; weight only via demo_profile_bucket.",
        },
        "features_all": rows,
        "features_included": included,
        "features_excluded": excluded,
        "included_count": len(included),
        "excluded_count": len(excluded),
        "leakage_note": LEAKAGE_NOTE,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="LOS feature selection plan for XGBoost")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    plan = build_plan(args.data_dir)
    out = _resolve(args.data_dir) / "los_model" / PLAN_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({plan['included_count']} included, {plan['excluded_count']} excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
