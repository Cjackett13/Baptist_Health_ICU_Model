"""
Encounter-grain modeling frame: join DuckDB sidecar to label table, apply cohort rules,
split ID / label / feature columns, and document imputation + missingness.

Outputs under ``data_dir``:
  - ``los_encounter_labels.parquet``
  - ``los_modeling_features.parquet``
  - ``los_modeling_column_groups.json``
  - ``los_imputation_policy.json``
  - ``los_feature_missingness.json``
  - ``los_cohort_excluded.parquet`` (encounters failing inclusion rules)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from los_data_cleaning import clean_bundle  # noqa: E402
from los_duckdb_features import FEATURE_COLUMNS_LOS, merge_los_duckdb_features  # noqa: E402
from los_feature_policy import (  # noqa: E402
    LABEL_REMAINING,
    LABEL_TOTAL,
    V1_FEATURE_WINDOW_HOURS,
    validate_modeling_columns,
)

ENCOUNTER_LABELS_NAME = "los_encounter_labels.parquet"
MODELING_FEATURES_NAME = "los_modeling_features.parquet"
COLUMN_GROUPS_NAME = "los_modeling_column_groups.json"
IMPUTATION_POLICY_NAME = "los_imputation_policy.json"
MISSINGNESS_NAME = "los_feature_missingness.json"
EXCLUDED_NAME = "los_cohort_excluded.parquet"

ID_COLUMNS = ["ENCOUNTER_ID", "PERSON_ID"]
LABEL_COLUMNS = [LABEL_TOTAL, LABEL_REMAINING]
LABEL_TRANSFORM_AUDIT = ["los_hours", "log1p_los_hours_total", "log1p_remaining_los_hours"]
CHECKPOINT_META = ["prediction_hour", "feature_window_hours"]
AUDIT_ONLY_COLUMNS = [
    "REG_DT_TM",
    "DISCH_DT_TM",
    "DISCH_DISPOSITION_CD",
    "los_hours_reconciled",
    "los_reconcile_ok",
    "los_reconcile_delta_h",
    "los_truncated",
    "truncated_death",
    "truncated_hospice",
    "truncated_ama",
    "truncated_disposition_cd",
    "principal_icd_12h",
    "admit_type_cd",
    "admit_src_cd",
    "unit_cd",
    "age_bin",
    "weight_bin",
    "weight_kg_est",
]

# String / coded features (not in DuckDB numeric list)
INCLUSION_RULES: dict[str, Any] = {
    "grain": "one_row_per_encounter_id",
    "age_years_min": 18,
    "encounter_type": "INPATIENT",
    "unit_codes": ["CICU", "CVICU", "MICU"],
    "require_valid_admit_discharge": True,
    "los_hours_min_exclusive": 0,
    "los_hours_max": 14 * 24,
    "exclude_dispositions": ["AMA"],
}

EXTRA_FEATURE_COLUMNS = [
    "age_years",
    "sex_bin",
    "race_cd",
    "ethnicity_cd",
    "icd_prefix",
    "admit_type_cd",
    "admit_src_cd",
    "unit_cd",
]


def _load_and_clean_bundle(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    names = [
        "person",
        "encounter",
        "diagnosis",
        "clinical_event",
        "medication_admin",
        "procedure_event",
        "scai_stage_hourly",
        "code_value",
    ]
    raw = {n: pd.read_parquet(_resolve(data_dir) / f"{n}.parquet") for n in names}
    return clean_bundle(raw)


def _age_at_admit(person: pd.DataFrame, enc: pd.DataFrame) -> pd.Series:
    p = person[["PERSON_ID", "BIRTH_DT_TM"]].copy()
    e = enc[["PERSON_ID", "REG_DT_TM"]].copy()
    m = e.merge(p, on="PERSON_ID", how="left")
    birth = pd.to_datetime(m["BIRTH_DT_TM"], errors="coerce")
    admit = pd.to_datetime(m["REG_DT_TM"], errors="coerce")
    return ((admit - birth).dt.days / 365.25).astype(float)


def reconcile_los_hours(enc: pd.DataFrame) -> pd.DataFrame:
    out = enc.copy()
    reg = pd.to_datetime(out["REG_DT_TM"], errors="coerce")
    disch = pd.to_datetime(out["DISCH_DT_TM"], errors="coerce")
    out["los_hours_calc"] = (disch - reg).dt.total_seconds() / 3600.0
    stored = pd.to_numeric(out["LOS_HOURS"], errors="coerce")
    out["los_reconcile_delta_h"] = (stored - out["los_hours_calc"]).abs()
    ok = out["los_reconcile_delta_h"].le(1.0) & out["los_hours_calc"].notna()
    out["los_reconcile_ok"] = ok
    out["los_hours_reconciled"] = np.where(
        ok,
        out["los_hours_calc"],
        np.where(stored.notna(), stored, out["los_hours_calc"]),
    )
    return out


def apply_inclusion_mask(enc: pd.DataFrame, person: pd.DataFrame) -> pd.Series:
    age = _age_at_admit(person, enc)
    enc = enc.copy()
    enc["_age_years"] = age.reindex(enc.index).values
    unit = enc["UNIT_CD"].astype("string").str.upper().str.strip()
    etype = enc["ENCNTR_TYPE_CD"].astype("string").str.upper().str.strip()
    disp = enc["DISCH_DISPOSITION_CD"].astype("string").str.upper().str.strip()
    los = pd.to_numeric(enc["los_hours_reconciled"], errors="coerce")
    reg = pd.to_datetime(enc["REG_DT_TM"], errors="coerce")
    disch = pd.to_datetime(enc["DISCH_DT_TM"], errors="coerce")
    rules = INCLUSION_RULES
    mask = (
        enc["_age_years"].ge(rules["age_years_min"]).fillna(False)
        & etype.eq(rules["encounter_type"]).fillna(False)
        & unit.isin(rules["unit_codes"]).fillna(False)
        & reg.notna()
        & disch.notna()
        & los.gt(rules["los_hours_min_exclusive"]).fillna(False)
        & los.le(rules["los_hours_max"]).fillna(False)
        & ~disp.isin(rules["exclude_dispositions"]).fillna(True)
    )
    return mask.astype(bool)


def flag_truncated_los(enc: pd.DataFrame, person: pd.DataFrame) -> pd.DataFrame:
    disp = enc["DISCH_DISPOSITION_CD"].astype("string").str.upper().str.strip()
    died = person.set_index("PERSON_ID")["DECEASED_DT_TM"].reindex(enc["PERSON_ID"]).notna()
    out = pd.DataFrame(index=enc.index)
    out["truncated_death"] = (disp.eq("EXPIRED").fillna(False).to_numpy() | died.fillna(False).to_numpy())
    out["truncated_hospice"] = disp.eq("HOSPICE").fillna(False).to_numpy()
    out["truncated_ama"] = disp.eq("AMA").fillna(False).to_numpy()
    out["los_truncated"] = out["truncated_death"] | out["truncated_hospice"] | out["truncated_ama"]
    out["truncated_disposition_cd"] = disp.where(out["los_truncated"], pd.NA)
    return out


def _principal_dx_category(dx: pd.DataFrame) -> pd.DataFrame:
    d = dx.copy()
    d["NOMENCLATURE_CD"] = d["NOMENCLATURE_CD"].astype("string")
    pr = d[d["DIAG_PRIORITY"] == 1].copy()
    if pr.empty:
        pr = d.sort_values("DIAG_PRIORITY").groupby("PERSON_ID", as_index=False).first()
    pr["icd_prefix"] = pr["NOMENCLATURE_CD"].str.slice(0, 3)
    return pr[["PERSON_ID", "icd_prefix"]].drop_duplicates("PERSON_ID")


def _resolve(data_dir: Path) -> Path:
    root = Path(__file__).resolve().parent.parent
    return data_dir.resolve() if data_dir.is_absolute() else (root / data_dir).resolve()


def build_modeling_feature_list() -> list[str]:
    raw = list(
        dict.fromkeys(
            FEATURE_COLUMNS_LOS
            + EXTRA_FEATURE_COLUMNS
            + list(CHECKPOINT_META)
        )
    )
    return validate_modeling_columns(raw, prediction_hour=V1_FEATURE_WINDOW_HOURS, strict_v1_12h=True)


def build_encounter_label_table(
    enc: pd.DataFrame,
    person: pd.DataFrame,
    *,
    prediction_hour: int = V1_FEATURE_WINDOW_HOURS,
) -> pd.DataFrame:
    """One row per encounter with reconciled LOS labels (before cohort filter)."""
    enc = reconcile_los_hours(enc)
    trunc = flag_truncated_los(enc, person)
    out = enc[
        [
            "ENCOUNTER_ID",
            "PERSON_ID",
            "REG_DT_TM",
            "DISCH_DT_TM",
            "ADMIT_TYPE_CD",
            "ADMIT_SRC_CD",
            "UNIT_CD",
            "DISCH_DISPOSITION_CD",
            "los_hours_reconciled",
            "los_reconcile_ok",
        ]
    ].copy()
    out["los_reconcile_delta_h"] = enc["los_reconcile_delta_h"].values
    for c in trunc.columns:
        out[c] = trunc[c].values
    out[LABEL_TOTAL] = pd.to_numeric(out["los_hours_reconciled"], errors="coerce")
    out["prediction_hour"] = int(prediction_hour)
    out["feature_window_hours"] = int(prediction_hour)
    out[LABEL_REMAINING] = np.maximum(
        0.0, out[LABEL_TOTAL] - float(prediction_hour)
    )
    out["log1p_los_hours_total"] = np.log1p(out[LABEL_TOTAL].clip(lower=0))
    out["log1p_remaining_los_hours"] = np.log1p(out[LABEL_REMAINING].clip(lower=0))
    out["los_hours"] = out[LABEL_TOTAL]
    return out


def _exclusion_reasons(enc: pd.DataFrame, person: pd.DataFrame) -> pd.DataFrame:
    """Per-encounter boolean flags for each cohort rule (for excluded audit)."""
    enc = enc.copy()
    age = _age_at_admit(person, enc)
    unit = enc["UNIT_CD"].astype("string").str.upper().str.strip()
    etype = enc["ENCNTR_TYPE_CD"].astype("string").str.upper().str.strip()
    disp = enc["DISCH_DISPOSITION_CD"].astype("string").str.upper().str.strip()
    los = pd.to_numeric(enc.get("los_hours_reconciled", enc.get("LOS_HOURS")), errors="coerce")
    reg = pd.to_datetime(enc["REG_DT_TM"], errors="coerce")
    disch = pd.to_datetime(enc["DISCH_DT_TM"], errors="coerce")
    rules = INCLUSION_RULES
    return pd.DataFrame(
        {
            "ENCOUNTER_ID": enc["ENCOUNTER_ID"].values,
            "PERSON_ID": enc["PERSON_ID"].values,
            "fail_age": (~age.ge(rules["age_years_min"])).fillna(True).astype(bool).values,
            "fail_encounter_type": (~etype.eq(rules["encounter_type"])).fillna(True).astype(bool).values,
            "fail_unit": (~unit.isin(rules["unit_codes"])).fillna(True).astype(bool).values,
            "fail_admit_discharge": (reg.isna() | disch.isna()).astype(bool).values,
            "fail_los_range": (
                ~los.gt(rules["los_hours_min_exclusive"]).fillna(True)
                | ~los.le(rules["los_hours_max"]).fillna(True)
            ).astype(bool).values,
            "fail_disposition": disp.isin(rules["exclude_dispositions"]).fillna(False).astype(bool).values,
        }
    )


def build_imputation_policy(feature_columns: list[str]) -> dict[str, Any]:
    """Mark imputation intent per column; training applies train-fold stats only."""
    duckdb_zero_fill = set(FEATURE_COLUMNS_LOS)
    columns: dict[str, dict[str, str]] = {}
    for col in feature_columns:
        if col in ("race_cd", "ethnicity_cd", "icd_prefix", "admit_type_cd", "admit_src_cd", "unit_cd"):
            columns[col] = {
                "dtype": "string",
                "prep_stage": "none_or_sentinel_cleaning",
                "train_imputation": "mode",
                "note": "Fit mode on train fold only; holdout/test use train statistics.",
            }
        elif col in CHECKPOINT_META:
            columns[col] = {
                "dtype": "int",
                "prep_stage": "fixed_v1_snapshot",
                "train_imputation": "none",
            }
        elif col in duckdb_zero_fill:
            columns[col] = {
                "dtype": "float",
                "prep_stage": "duckdb_coalesce_zero",
                "train_imputation": "median_if_missing",
                "note": "DuckDB COALESCE(...,0); if NaN remains after join, median on train fold.",
            }
        else:
            columns[col] = {
                "dtype": "float",
                "prep_stage": "as_observed",
                "train_imputation": "median_if_missing",
            }
    return {
        "policy_version": 1,
        "grain": "encounter",
        "training_scope": "train_fold_only",
        "not_applied_in_prep": True,
        "bundle_cleaning_note": (
            "Table cleaning may median-impute medication_admin / scai_stage_hourly only; "
            "modeling imputation is deferred to training."
        ),
        "columns": columns,
    }


def compute_feature_missingness(features: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    n = len(features)
    per_col: dict[str, dict[str, float | int]] = {}
    for col in feature_columns:
        if col not in features.columns:
            per_col[col] = {"n_missing": n, "pct_missing": 100.0}
            continue
        s = features[col]
        if pd.api.types.is_numeric_dtype(s):
            n_miss = int(s.isna().sum())
        else:
            n_miss = int(s.isna().sum() + (s.astype("string").str.strip() == "").sum())
        per_col[col] = {
            "n_missing": n_miss,
            "pct_missing": round(100.0 * n_miss / n, 4) if n else 0.0,
        }
    return {
        "n_encounters": n,
        "feature_columns": feature_columns,
        "per_column": per_col,
        "any_missing": {k: v for k, v in per_col.items() if v["n_missing"] > 0},
    }


def build_encounter_modeling_frames(
    data_dir: Path,
    *,
    tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
  Join sidecar to encounter labels, drop cohort failures, return (labels, features, excluded).
    """
    data_dir = _resolve(data_dir)
    if tables is None:
        tables, _ = _load_and_clean_bundle(data_dir)

    enc = reconcile_los_hours(tables["encounter"])
    person = tables["person"]
    dx = tables["diagnosis"]

    labels_all = build_encounter_label_table(enc, person)
    reasons = _exclusion_reasons(enc, person)
    incl = apply_inclusion_mask(enc, person).reindex(labels_all.index, fill_value=False).fillna(False).astype(bool)
    excluded_ids = labels_all.loc[~incl, "ENCOUNTER_ID"]
    excluded = labels_all[labels_all["ENCOUNTER_ID"].isin(excluded_ids)].merge(
        reasons[reasons["ENCOUNTER_ID"].isin(excluded_ids)],
        on=["ENCOUNTER_ID", "PERSON_ID"],
        how="left",
    )
    labels_all = labels_all.loc[incl].copy()

    labels_all = merge_los_duckdb_features(labels_all, data_dir)
    person_demo = person[["PERSON_ID", "RACE_CD", "ETHNICITY_CD"]].copy()
    person_demo["race_cd"] = person_demo["RACE_CD"].astype("string").str.upper().str.strip()
    person_demo["ethnicity_cd"] = person_demo["ETHNICITY_CD"].astype("string").str.upper().str.strip()
    labels_all = labels_all.merge(person_demo[["PERSON_ID", "race_cd", "ethnicity_cd"]], on="PERSON_ID", how="left")
    if "principal_icd_12h" in labels_all.columns:
        labels_all["icd_prefix"] = labels_all["principal_icd_12h"].astype("string").str.slice(0, 3)
    else:
        labels_all = labels_all.merge(_principal_dx_category(dx), on="PERSON_ID", how="left")
    labels_all["admit_type_cd"] = labels_all["ADMIT_TYPE_CD"].astype("string").str.upper().str.strip()
    labels_all["admit_src_cd"] = labels_all["ADMIT_SRC_CD"].astype("string").str.upper().str.strip()
    labels_all["unit_cd"] = labels_all["UNIT_CD"].astype("string").str.upper().str.strip()

    feature_columns = build_modeling_feature_list()
    id_cols = [c for c in ID_COLUMNS if c in labels_all.columns]
    label_cols = [c for c in LABEL_COLUMNS + LABEL_TRANSFORM_AUDIT if c in labels_all.columns]
    label_export = labels_all[id_cols + label_cols + [c for c in CHECKPOINT_META if c in labels_all.columns]].copy()
    feat_export = labels_all[id_cols + [c for c in feature_columns if c in labels_all.columns]].copy()

    return label_export, feat_export, excluded


def write_modeling_artifacts(
    data_dir: Path,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    excluded: pd.DataFrame,
    *,
    enc_raw_n: int,
) -> None:
    data_dir = _resolve(data_dir)
    feature_columns = [c for c in features.columns if c not in ID_COLUMNS]

    groups = {
        "grain": "encounter",
        "modeling_unit": "ENCOUNTER_ID",
        "split_key": "PERSON_ID",
        "id_columns": ID_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "label_transform_audit_only": LABEL_TRANSFORM_AUDIT,
        "feature_columns": feature_columns,
        "checkpoint_meta_columns": CHECKPOINT_META,
        "audit_only_columns": AUDIT_ONLY_COLUMNS,
        "counts": {
            "encounters_raw": enc_raw_n,
            "encounters_included": int(len(labels)),
            "encounters_excluded": int(enc_raw_n - len(labels)),
        },
    }
    (data_dir / COLUMN_GROUPS_NAME).write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (data_dir / IMPUTATION_POLICY_NAME).write_text(
        json.dumps(build_imputation_policy(feature_columns), indent=2),
        encoding="utf-8",
    )
    (data_dir / MISSINGNESS_NAME).write_text(
        json.dumps(compute_feature_missingness(features, feature_columns), indent=2),
        encoding="utf-8",
    )

    labels.to_parquet(data_dir / ENCOUNTER_LABELS_NAME, index=False)
    features.to_parquet(data_dir / MODELING_FEATURES_NAME, index=False)
    excluded.to_parquet(data_dir / EXCLUDED_NAME, index=False)


def materialize_encounter_modeling(data_dir: Path, *, tables: dict[str, pd.DataFrame] | None = None) -> Path:
    data_dir = _resolve(data_dir)
    enc_n = len(tables["encounter"]) if tables is not None else len(pd.read_parquet(data_dir / "encounter.parquet"))
    labels, features, excluded = build_encounter_modeling_frames(data_dir, tables=tables)
    write_modeling_artifacts(data_dir, labels, features, excluded, enc_raw_n=enc_n)
    return data_dir / MODELING_FEATURES_NAME


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build encounter-grain LOS modeling ID/label/feature tables.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    out = materialize_encounter_modeling(args.data_dir)
    print(f"Wrote {out}")
    print(f"Column groups: {_resolve(args.data_dir) / COLUMN_GROUPS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
