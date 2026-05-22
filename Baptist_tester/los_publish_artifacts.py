#!/usr/bin/env python3
"""
Publish LOS modeling artifacts under ``data/``:

  los_feature_columns.json, los_final_model_features.json,
  los_feature_audit.json, los_cleaning_log.json,
  los_modeling_frame.parquet, los_split_manifest.json,
  los_data_dictionary_notes.md, los_eda/ (via los_baptist_style_eda.py)

  python3 Baptist_tester/los_publish_artifacts.py --data-dir data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "Baptist_tester") not in sys.path:
    sys.path.insert(0, str(_REPO / "Baptist_tester"))

from los_model_config import (  # noqa: E402
    DROPPED_FEATURES,
    ID_COLUMNS,
    LABEL_PRIMARY,
    LABEL_SECONDARY,
    LABEL_TRAIN_TRANSFORM,
    MODEL_FEATURE_COLUMNS,
    SPLIT_RATIOS,
    SPLIT_SEED,
    build_feature_columns_doc,
)

FINAL_FEATURES_NAME = "los_final_model_features.json"
FEATURE_COLUMNS_NAME = "los_feature_columns.json"
FEATURE_AUDIT_NAME = "los_feature_audit.json"
CLEANING_LOG_NAME = "los_cleaning_log.json"
MODELING_FRAME_NAME = "los_modeling_frame.parquet"
SPLIT_MANIFEST_NAME = "los_split_manifest.json"
DICT_NOTES_NAME = "los_data_dictionary_notes.md"


def _resolve(data_dir: Path) -> Path:
    return data_dir.resolve() if data_dir.is_absolute() else (_REPO / data_dir).resolve()


def load_merged_frame(data_dir: Path) -> pd.DataFrame:
    data_dir = _resolve(data_dir)
    labels = pd.read_parquet(data_dir / "los_encounter_labels.parquet")
    feat = pd.read_parquet(data_dir / "los_modeling_features.parquet")
    import pyarrow.parquet as pq

    pf_path = data_dir / "los_patient_frame.parquet"
    use_pf = ["ENCOUNTER_ID", "admit_type_cd", "admit_src_cd", "unit_cd"]
    names = pq.read_schema(pf_path).names
    pf = pd.read_parquet(pf_path, columns=[c for c in use_pf if c in names])
    df = labels.merge(feat, on=["ENCOUNTER_ID", "PERSON_ID"], how="inner")
    df = df.merge(pf, on="ENCOUNTER_ID", how="left", suffixes=("", "_pf"))
    for c in ("admit_type_cd", "admit_src_cd", "unit_cd"):
        if f"{c}_pf" in df.columns:
            df[c] = df[c].fillna(df[f"{c}_pf"])
            df = df.drop(columns=[f"{c}_pf"])
    if LABEL_TRAIN_TRANSFORM not in df.columns:
        df[LABEL_TRAIN_TRANSFORM] = np.log1p(pd.to_numeric(df[LABEL_PRIMARY], errors="coerce").clip(lower=0))
    return df


def build_feature_audit(data_dir: Path, df: pd.DataFrame) -> dict[str, Any]:
    """Spearman + Pearson + MI vs los_hours_total for all modeling + dropped columns."""
    from sklearn.feature_selection import mutual_info_regression

    data_dir = _resolve(data_dir)
    y = pd.to_numeric(df[LABEL_PRIMARY], errors="coerce")
    audit_cols = sorted(
        set(MODEL_FEATURE_COLUMNS)
        | set(DROPPED_FEATURES)
        | set(json.loads((data_dir / "los_modeling_column_groups.json").read_text())["feature_columns"])
    )
    rows = []
    Xm_cols = []
    for col in audit_cols:
        if col not in df.columns:
            rows.append({"feature": col, "in_frame": False})
            continue
        s = df[col]
        x_num = pd.to_numeric(s, errors="coerce")
        if x_num.notna().sum() >= max(10, int(0.5 * len(s))):
            x = x_num
        else:
            x = pd.Series(pd.Categorical(s.astype("string")).codes, index=s.index)
        m = x.notna() & y.notna()
        sp, pe, mi = np.nan, np.nan, None
        if m.sum() >= 10 and x[m].nunique() > 1:
            try:
                sp, _ = stats.spearmanr(x[m], y[m], nan_policy="omit")
                pe = float(np.corrcoef(x[m].astype(float), y[m].astype(float))[0, 1])
            except Exception:
                pass
            if col in MODEL_FEATURE_COLUMNS:
                Xm_cols.append(col)
                med = float(x[m].median())
                x_fill = x[m].fillna(med)
                try:
                    mi_vals = mutual_info_regression(
                        x_fill.values.reshape(-1, 1), y[m].values, random_state=SPLIT_SEED
                    )
                    mi = float(mi_vals[0])
                except Exception:
                    mi = None
        rows.append(
            {
                "feature": col,
                "in_model": col in MODEL_FEATURE_COLUMNS,
                "dropped": col in DROPPED_FEATURES,
                "spearman_rho": float(sp) if np.isfinite(sp) else None,
                "pearson_r": float(pe) if np.isfinite(pe) else None,
                "mutual_information": (
                    float(mi) if mi is not None and np.isfinite(float(mi)) else None
                ),
            }
        )
    rows.sort(key=lambda r: abs(r.get("spearman_rho") or 0), reverse=True)
    duck_path = data_dir / "duckdb_los_feature_audit.json"
    duck_audit = json.loads(duck_path.read_text()) if duck_path.is_file() else {}
    return {
        "target": LABEL_PRIMARY,
        "n_encounters": int(len(df)),
        "model_features": MODEL_FEATURE_COLUMNS,
        "per_feature": rows,
        "model_features_only": [r for r in rows if r.get("in_model")],
        "duckdb_exploratory_audit_file": "duckdb_los_feature_audit.json",
        "duckdb_mi_full_cohort": duck_audit.get("mutual_information_full_cohort", {}),
        "note": "MI in per_feature is univariate on model columns; duckdb file is full-cohort exploratory.",
    }


def build_cleaning_log(data_dir: Path) -> dict[str, Any]:
    data_dir = _resolve(data_dir)
    raw = json.loads((data_dir / "los_data_cleaning_audit.json").read_text(encoding="utf-8"))
    totals = raw.get("totals", {})
    cleaned_manifest = data_dir / "los_cleaned_bundle_manifest.json"
    return {
        "source_file": "los_data_cleaning_audit.json",
        "summary": {
            **totals,
            "sentinels_to_missing_total": totals.get("sentinels_to_missing"),
            "numeric_median_imputation": raw.get("numeric_median_imputation", {}),
        },
        "imputation_policy": raw.get("imputation_policy"),
        "datetime_policy": raw.get("datetime_policy"),
        "raw_parquet_policy": raw.get("raw_parquet_policy"),
        "rules_applied": raw.get("rules_applied", []),
        "per_table": raw.get("per_table"),
        "table_rules": raw.get("table_rules_audit"),
        "cleaned_bundle_manifest": (
            cleaned_manifest.name if cleaned_manifest.is_file() else None
        ),
        "corrections": [
            "String sentinels (NULL, N/A, etc.) → pd.NA on object/string/category/_CD/_TXT columns",
            "ICD-10 validation on diagnosis.NOMENCLATURE_CD",
            "EVENT_CD validation on clinical_event",
            "LOS reconcile: prefer datetime diff when |stored − delta| > 1h",
            "LOS_HOURS outside (0, 336] nulled on encounter",
            "Unparseable datetimes → NaT (rows kept)",
            "Out-of-range RESULT_VAL nulled (not median-filled) on clinical_event",
            "Median impute only medication_admin / scai_stage_hourly numerics in bundle clean",
            "Optional cleaned parquets: data/cleaned/ (see los_cleaned_bundle_manifest.json)",
            "Cohort exclusion (AMA, age, unit, etc.) applied in los_modeling_frame.py — not row drops in raw parquets",
        ],
    }


def build_split_manifest(df: pd.DataFrame) -> dict[str, Any]:
    """Patient-level 60/20/20 split — IDs only, no model fit."""
    persons = df.drop_duplicates("PERSON_ID")[["PERSON_ID"]].copy()
    groups = persons["PERSON_ID"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=1 - SPLIT_RATIOS["train"], random_state=SPLIT_SEED)
    train_i, temp_i = next(gss1.split(persons, groups=groups))
    temp_p = persons.iloc[temp_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SPLIT_SEED)
    val_rel, test_rel = next(gss2.split(temp_p, groups=temp_p["PERSON_ID"].values))
    val_i = temp_i[val_rel]
    test_i = temp_i[test_rel]
    train_ids = sorted(persons.iloc[train_i]["PERSON_ID"].astype(int).tolist())
    val_ids = sorted(persons.iloc[val_i]["PERSON_ID"].astype(int).tolist())
    test_ids = sorted(persons.iloc[test_i]["PERSON_ID"].astype(int).tolist())
    enc_train = df[df["PERSON_ID"].isin(train_ids)]["ENCOUNTER_ID"].astype(int).tolist()
    enc_val = df[df["PERSON_ID"].isin(val_ids)]["ENCOUNTER_ID"].astype(int).tolist()
    enc_test = df[df["PERSON_ID"].isin(test_ids)]["ENCOUNTER_ID"].astype(int).tolist()
    return {
        "policy": "GroupShuffleSplit on PERSON_ID (60% train / 20% val / 20% test)",
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "no_model_fit": True,
        "train_person_ids": train_ids,
        "val_person_ids": val_ids,
        "test_person_ids": test_ids,
        "train_encounter_ids": enc_train,
        "val_encounter_ids": enc_val,
        "test_encounter_ids": enc_test,
        "counts": {
            "persons_train": len(train_ids),
            "persons_val": len(val_ids),
            "persons_test": len(test_ids),
            "encounters_train": len(enc_train),
            "encounters_val": len(enc_val),
            "encounters_test": len(enc_test),
        },
    }


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = ID_COLUMNS + [LABEL_PRIMARY, LABEL_SECONDARY, LABEL_TRAIN_TRANSFORM] + MODEL_FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling frame: {missing}")
    return df[cols].copy()


def write_dictionary_notes(data_dir: Path) -> Path:
    data_dir = _resolve(data_dir)
    src = _REPO / "Baptist_tester" / "__pycache__" / "data_dictionary.md"
    base = src.read_text(encoding="utf-8") if src.is_file() else ""
    appendix = f"""
---

## LOS model appendix (Dataset B, v1 ≤12h snapshot)

**Grain:** one row per `ENCOUNTER_ID` (5000 encounters = 5000 persons in current cohort).

**Labels:** `{LABEL_PRIMARY}` (raw hours), `{LABEL_SECONDARY}` (hours remaining at 12h), train on `{LABEL_TRAIN_TRANSFORM}`.

**Model features ({len(MODEL_FEATURE_COLUMNS)} raw columns before one-hot):**

{chr(10).join('- `' + c + '`' for c in MODEL_FEATURE_COLUMNS)}

**Transforms:** see `los_feature_columns.json` and `los_final_model_features.json`.

**Split:** `los_split_manifest.json` — patient-level train/val/test IDs only (seed {SPLIT_SEED}).

**Artifacts:** `los_modeling_frame.parquet`, `los_feature_audit.json`, `los_cleaning_log.json`, `eda/los_eda_*.html`.
"""
    out = data_dir / DICT_NOTES_NAME
    out.write_text(base + appendix, encoding="utf-8")
    return out


def publish_all(data_dir: Path, *, run_eda: bool = True) -> dict[str, Path]:
    data_dir = _resolve(data_dir)
    df = load_merged_frame(data_dir)
    paths: dict[str, Path] = {}

    # los_feature_columns.json
    p = data_dir / FEATURE_COLUMNS_NAME
    p.write_text(json.dumps(build_feature_columns_doc(), indent=2), encoding="utf-8")
    paths["feature_columns"] = p

    # los_final_model_features.json — model only
    p = data_dir / FINAL_FEATURES_NAME
    p.write_text(
        json.dumps(
            {
                "description": "Only features used in LOS v1 model (before one-hot expansion).",
                "label_for_training": LABEL_TRAIN_TRANSFORM,
                "label_columns_raw": [LABEL_PRIMARY, LABEL_SECONDARY],
                "feature_columns": MODEL_FEATURE_COLUMNS,
                "numeric": [c for c in MODEL_FEATURE_COLUMNS if c not in ("admit_type_cd", "admit_src_cd", "unit_cd")],
                "categorical_one_hot": ["admit_type_cd", "admit_src_cd", "unit_cd"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["final_model_features"] = p

    p = data_dir / FEATURE_AUDIT_NAME
    p.write_text(json.dumps(build_feature_audit(data_dir, df), indent=2), encoding="utf-8")
    paths["feature_audit"] = p

    p = data_dir / CLEANING_LOG_NAME
    p.write_text(json.dumps(build_cleaning_log(data_dir), indent=2), encoding="utf-8")
    paths["cleaning_log"] = p

    mf = build_modeling_frame(df)
    p = data_dir / MODELING_FRAME_NAME
    mf.to_parquet(p, index=False)
    paths["modeling_frame"] = p

    p = data_dir / SPLIT_MANIFEST_NAME
    p.write_text(json.dumps(build_split_manifest(df), indent=2), encoding="utf-8")
    paths["split_manifest"] = p

    paths["dictionary_notes"] = write_dictionary_notes(data_dir)

    if run_eda:
        from los_baptist_style_eda import run_baptist_style_eda

        paths["eda_index"] = run_baptist_style_eda(data_dir, df=mf)

    return paths


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--skip-eda", action="store_true")
    args = ap.parse_args()
    paths = publish_all(args.data_dir, run_eda=not args.skip_eda)
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
