#!/usr/bin/env python3
"""
Length-of-stay (LOS) cohort cleaning and patient-level feature frame.

**Current prep (v1):** one row per ``PERSON_ID``; features through first 12h; target ``los_hours``.

**Planned modeling (v2):** one row per ``(PERSON_ID, prediction_hour)`` every 24h from admit;
features censored at ``prediction_hour``; target ``remaining_los_hours``. See
``Baptist_tester/los_prediction_policy.json``.

  python3 Baptist_tester/los_data_prep.py
  python3 Baptist_tester/los_data_prep.py --data-dir Baptist_tester/synth_cs_data --force

Writes under ``--data-dir``:
  - ``los_cohort_manifest.json`` — inclusion rules, target policy, row counts
  - ``los_patient_frame.parquet`` — modeling frame (v1 admit snapshot)
  - ``los_data_quality_report.json`` — sentinel cleanup, reconciliation, truncation flags
  - ``los_admit_features.parquet`` — DuckDB aggregates through 12h (v1 only)
  - ``los_feature_columns.json`` — tiered feature manifest (update when adding columns)
  - ``los_prediction_policy.json`` — 24h checkpoint + forbidden-field rules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_MODELS = _REPO / "Frontend" / "lib" / "models"
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from mortality_demographics import build_demo_profile_legend  # noqa: E402

from los_feature_policy import (  # noqa: E402
    FIRST_PREDICTION_HOUR,
    LABEL_REMAINING,
    LABEL_TOTAL,
    PREDICTION_CADENCE_HOURS,
    V1_FEATURE_WINDOW_HOURS,
    assert_frame_separated,
    build_policy_dict,
    validate_modeling_columns,
)

INCLUSION_RULES: dict[str, Any] = {
    "grain": "one_row_per_person_id",
    "age_years_min": 18,
    "encounter_type": "INPATIENT",
    "unit_codes": ["CICU", "CVICU", "MICU"],
    "require_valid_admit_discharge": True,
    "los_hours_min_exclusive": 0,
    "los_hours_max": 14 * 24,
    "exclude_dispositions": ["AMA"],
    "note": (
        "Cardiogenic-shock ICU cohort. EXPIRED/HOSPICE kept but flagged as truncated; "
        "not dropped by default."
    ),
}

TARGET_POLICY: dict[str, Any] = {
    "labels": {
        LABEL_TOTAL: "Total ICU stay hours (reconciled); same on all checkpoints.",
        LABEL_REMAINING: "max(0, los_hours_total - prediction_hour).",
    },
    "transform_primary": "raw_hours",
    "transform_eda_columns": ["log1p_los_hours_total", "log1p_remaining_los_hours"],
    "rationale": "Train/predict both total and remaining LOS; neither label in feature matrix.",
    "reconcile": "prefer_datetime_diff_hours_when_within_1h_else_los_hours",
}

STRING_SENTINELS: frozenset[str] = frozenset(
    {"", " ", "NULL", "null", "N/A", "n/a", "NA", "NONE", "UNK", "UNKNOWN", "-99", "999"}
)

COHORT_MANIFEST_NAME = "los_cohort_manifest.json"
QUALITY_REPORT_NAME = "los_data_quality_report.json"
PATIENT_FRAME_NAME = "los_patient_frame.parquet"
FEATURE_COLUMNS_NAME = "los_feature_columns.json"
ADMIT_FEATURES_NAME = "los_admit_features.parquet"
PREDICTION_POLICY_NAME = "los_prediction_policy.json"
CHECKPOINT_FRAME_NAME = "los_checkpoint_frame.parquet"
EARLY_WINDOW_DOC_NAME = "los_early_window_per_table.json"


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (_REPO / p).resolve()


def _normalize_object_series(s: pd.Series) -> pd.Series:
    out = s.astype("string")
    out = out.str.strip()
    upper = out.str.upper()
    mask = upper.isin(STRING_SENTINELS) | out.isna()
    return out.mask(mask, pd.NA)


def _clean_table_strings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, int] = {}
    out = df.copy()
    for c in out.select_dtypes(include=["object", "string"]).columns:
        before_na = int(out[c].isna().sum())
        cleaned = _normalize_object_series(out[c])
        out[c] = cleaned
        n_replaced = int(cleaned.isna().sum()) - before_na
        if n_replaced > 0:
            counts[c] = n_replaced
    return out, counts


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
    return (
        enc["_age_years"].ge(rules["age_years_min"])
        & etype.eq(rules["encounter_type"])
        & unit.isin(rules["unit_codes"])
        & reg.notna()
        & disch.notna()
        & los.gt(rules["los_hours_min_exclusive"])
        & los.le(rules["los_hours_max"])
        & ~disp.isin(rules["exclude_dispositions"])
    )


def flag_truncated_los(enc: pd.DataFrame, person: pd.DataFrame) -> pd.DataFrame:
    disp = enc["DISCH_DISPOSITION_CD"].astype("string").str.upper().str.strip()
    died = person.set_index("PERSON_ID")["DECEASED_DT_TM"].reindex(enc["PERSON_ID"]).notna()
    out = pd.DataFrame(index=enc.index)
    out["truncated_death"] = disp.eq("EXPIRED") | died.to_numpy()
    out["truncated_hospice"] = disp.eq("HOSPICE")
    out["truncated_ama"] = disp.eq("AMA")
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


def materialize_admit_features(data_dir: Path) -> Path:
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("Install duckdb: pip install duckdb") from exc

    data_dir = data_dir.resolve()

    def q(p: Path) -> str:
        return str(p.resolve()).replace("'", "''")

    out_p = data_dir / ADMIT_FEATURES_NAME
    sql = f"""
    PRAGMA threads=4;
    CREATE OR REPLACE VIEW enc AS SELECT * FROM read_parquet('{q(data_dir / "encounter.parquet")}');
    CREATE OR REPLACE VIEW person AS SELECT * FROM read_parquet('{q(data_dir / "person.parquet")}');
    CREATE OR REPLACE VIEW med AS SELECT * FROM read_parquet('{q(data_dir / "medication_admin.parquet")}');
    CREATE OR REPLACE VIEW proc AS SELECT * FROM read_parquet('{q(data_dir / "procedure_event.parquet")}');
    CREATE OR REPLACE VIEW scai AS SELECT * FROM read_parquet('{q(data_dir / "scai_stage_hourly.parquet")}');
    CREATE OR REPLACE VIEW cevt AS SELECT * FROM read_parquet('{q(data_dir / "clinical_event.parquet")}');

    CREATE OR REPLACE VIEW enc_ts AS
    SELECT e.*, TRY_CAST(e.REG_DT_TM AS TIMESTAMP) AS admit_ts FROM enc e;

    CREATE OR REPLACE VIEW med_early AS
    SELECT m.PERSON_ID,
      COUNT(DISTINCT m.MEDICATION_CD)::DOUBLE AS med_distinct_12h,
      AVG(TRY_CAST(m.INFUSION_RATE AS DOUBLE)) AS med_infusion_mean_12h,
      COUNT(*)::DOUBLE AS med_admin_rows_12h
    FROM med m JOIN enc_ts e ON e.ENCOUNTER_ID = m.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(m.ADMIN_START_DT_TM AS TIMESTAMP) - e.admit_ts))/3600.0 <= 12
    GROUP BY 1;

    CREATE OR REPLACE VIEW proc_early AS
    SELECT pr.PERSON_ID, COUNT(*)::DOUBLE AS proc_n_12h
    FROM proc pr JOIN enc_ts e ON e.ENCOUNTER_ID = pr.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(pr.PROC_START_DT_TM AS TIMESTAMP) - e.admit_ts))/3600.0 <= 12
    GROUP BY 1;

    CREATE OR REPLACE VIEW scai_early AS
    SELECT e.PERSON_ID,
      MIN(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)) AS scai_first_12h,
      MAX(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)) AS scai_last_12h,
      AVG(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)) AS scai_mean_12h,
      STDDEV_SAMP(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)) AS scai_std_12h
    FROM scai sh JOIN enc_ts e ON e.ENCOUNTER_ID = sh.ENCOUNTER_ID
    WHERE TRY_CAST(sh.HOUR_FROM_ADMIT AS DOUBLE) <= 12
    GROUP BY 1;

    CREATE OR REPLACE VIEW ce_early AS
    SELECT c.PERSON_ID, COUNT(*)::DOUBLE AS clinical_event_n_12h
    FROM cevt c JOIN enc_ts e ON e.ENCOUNTER_ID = c.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(c.EVENT_END_DT_TM AS TIMESTAMP) - e.admit_ts))/3600.0 <= 12
    GROUP BY 1;

    CREATE OR REPLACE VIEW demo AS
    SELECT e.PERSON_ID,
      date_diff('year', TRY_CAST(p.BIRTH_DT_TM AS TIMESTAMP), TRY_CAST(e.REG_DT_TM AS TIMESTAMP))::DOUBLE AS age_years,
      CASE WHEN UPPER(TRIM(COALESCE(p.SEX_CD,''))) IN ('M','MALE') THEN 1 ELSE 0 END::INT AS sex_bin,
      UPPER(TRIM(COALESCE(p.RACE_CD,''))) AS race_cd,
      UPPER(TRIM(COALESCE(p.ETHNICITY_CD,''))) AS ethnicity_cd
    FROM enc_ts e JOIN person p ON p.PERSON_ID = e.PERSON_ID;

    COPY (
      SELECT d.PERSON_ID, d.age_years, d.sex_bin, d.race_cd, d.ethnicity_cd,
        COALESCE(me.med_distinct_12h,0) AS med_distinct_12h,
        COALESCE(me.med_infusion_mean_12h,0) AS med_infusion_mean_12h,
        COALESCE(pe.proc_n_12h,0) AS proc_n_12h,
        COALESCE(se.scai_first_12h,0) AS scai_first_12h,
        COALESCE(se.scai_last_12h,0) AS scai_last_12h,
        COALESCE(se.scai_mean_12h,0) AS scai_mean_12h,
        COALESCE(se.scai_std_12h,0) AS scai_std_12h,
        (COALESCE(se.scai_last_12h,0)-COALESCE(se.scai_first_12h,0))::DOUBLE AS scai_slope_12h,
        COALESCE(ce.clinical_event_n_12h,0) AS clinical_event_n_12h,
        COALESCE(me.med_admin_rows_12h,0)/NULLIF(COALESCE(ce.clinical_event_n_12h,0)+1.0,0) AS med_per_clinical_event_12h
      FROM demo d
      LEFT JOIN med_early me ON me.PERSON_ID = d.PERSON_ID
      LEFT JOIN proc_early pe ON pe.PERSON_ID = d.PERSON_ID
      LEFT JOIN scai_early se ON se.PERSON_ID = d.PERSON_ID
      LEFT JOIN ce_early ce ON ce.PERSON_ID = d.PERSON_ID
    ) TO '{q(out_p)}' (FORMAT PARQUET);
    """
    con = duckdb.connect(database=":memory:")
    con.execute(sql)
    con.close()
    return out_p


def materialize_checkpoint_frame(data_dir: Path) -> Path:
    """
    v2 grain: (PERSON_ID, prediction_hour) every 24h while patient still in ICU.
    Features censored at prediction_hour; labels los_hours_total + remaining_los_hours.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("Install duckdb: pip install duckdb") from exc

    data_dir = data_dir.resolve()
    cadence = PREDICTION_CADENCE_HOURS
    first_h = FIRST_PREDICTION_HOUR

    def q(p: Path) -> str:
        return str(p.resolve()).replace("'", "''")

    out_p = data_dir / CHECKPOINT_FRAME_NAME
    sql = f"""
    PRAGMA threads=4;
    CREATE OR REPLACE VIEW enc AS SELECT * FROM read_parquet('{q(data_dir / "encounter.parquet")}');
    CREATE OR REPLACE VIEW person AS SELECT * FROM read_parquet('{q(data_dir / "person.parquet")}');
    CREATE OR REPLACE VIEW med AS SELECT * FROM read_parquet('{q(data_dir / "medication_admin.parquet")}');
    CREATE OR REPLACE VIEW proc AS SELECT * FROM read_parquet('{q(data_dir / "procedure_event.parquet")}');
    CREATE OR REPLACE VIEW scai AS SELECT * FROM read_parquet('{q(data_dir / "scai_stage_hourly.parquet")}');
    CREATE OR REPLACE VIEW cevt AS SELECT * FROM read_parquet('{q(data_dir / "clinical_event.parquet")}');
    CREATE OR REPLACE VIEW dx AS SELECT * FROM read_parquet('{q(data_dir / "diagnosis.parquet")}');

    CREATE OR REPLACE VIEW enc_base AS
    SELECT
      e.PERSON_ID,
      e.ENCOUNTER_ID,
      TRY_CAST(e.REG_DT_TM AS TIMESTAMP) AS admit_ts,
      UPPER(TRIM(COALESCE(e.ENCNTR_TYPE_CD,''))) AS encntr_type_cd,
      UPPER(TRIM(COALESCE(e.UNIT_CD,''))) AS unit_cd,
      UPPER(TRIM(COALESCE(e.ADMIT_TYPE_CD,''))) AS admit_type_cd,
      UPPER(TRIM(COALESCE(e.ADMIT_SRC_CD,''))) AS admit_src_cd,
      TRY_CAST(e.DISCH_DT_TM AS TIMESTAMP) AS disch_ts,
      TRY_CAST(e.LOS_HOURS AS DOUBLE) AS los_stored,
      EXTRACT(EPOCH FROM (TRY_CAST(e.DISCH_DT_TM AS TIMESTAMP) - TRY_CAST(e.REG_DT_TM AS TIMESTAMP))) / 3600.0 AS los_calc,
      CASE
        WHEN ABS(TRY_CAST(e.LOS_HOURS AS DOUBLE) - EXTRACT(EPOCH FROM (
          TRY_CAST(e.DISCH_DT_TM AS TIMESTAMP) - TRY_CAST(e.REG_DT_TM AS TIMESTAMP)
        )) / 3600.0) <= 1.0
        THEN EXTRACT(EPOCH FROM (
          TRY_CAST(e.DISCH_DT_TM AS TIMESTAMP) - TRY_CAST(e.REG_DT_TM AS TIMESTAMP)
        )) / 3600.0
        ELSE COALESCE(TRY_CAST(e.LOS_HOURS AS DOUBLE), EXTRACT(EPOCH FROM (
          TRY_CAST(e.DISCH_DT_TM AS TIMESTAMP) - TRY_CAST(e.REG_DT_TM AS TIMESTAMP)
        )) / 3600.0)
      END::DOUBLE AS los_hours_total
    FROM enc e;

    CREATE OR REPLACE VIEW enc_ok AS
    SELECT b.*,
      date_diff('year', TRY_CAST(p.BIRTH_DT_TM AS TIMESTAMP), b.admit_ts)::DOUBLE AS age_years,
      CASE WHEN UPPER(TRIM(COALESCE(p.SEX_CD,''))) IN ('M','MALE') THEN 1 ELSE 0 END::INT AS sex_bin,
      UPPER(TRIM(COALESCE(p.RACE_CD,''))) AS race_cd,
      UPPER(TRIM(COALESCE(p.ETHNICITY_CD,''))) AS ethnicity_cd
    FROM enc_base b
    JOIN person p ON p.PERSON_ID = b.PERSON_ID
    WHERE b.admit_ts IS NOT NULL AND b.disch_ts IS NOT NULL
      AND b.los_hours_total > 0 AND b.los_hours_total <= {14 * 24}
      AND b.encntr_type_cd = 'INPATIENT'
      AND b.unit_cd IN ('CICU','CVICU','MICU')
      AND date_diff('year', TRY_CAST(p.BIRTH_DT_TM AS TIMESTAMP), b.admit_ts) >= 18;

    CREATE OR REPLACE VIEW hours AS
    SELECT UNNEST(generate_series({first_h}::BIGINT, {14 * 24}::BIGINT, {cadence}::BIGINT))::INT AS prediction_hour;

    CREATE OR REPLACE VIEW ckpt AS
    SELECT
      e.PERSON_ID,
      e.ENCOUNTER_ID,
      h.prediction_hour,
      h.prediction_hour::INT AS feature_window_hours,
      e.los_hours_total,
      GREATEST(0.0, e.los_hours_total - h.prediction_hour)::DOUBLE AS remaining_los_hours,
      e.age_years, e.sex_bin, e.race_cd, e.ethnicity_cd,
      e.admit_type_cd, e.admit_src_cd, e.unit_cd
    FROM enc_ok e
    CROSS JOIN hours h
    WHERE h.prediction_hour < e.los_hours_total;

    CREATE OR REPLACE VIEW med_cap AS
    SELECT
      c.PERSON_ID, c.prediction_hour,
      COUNT(DISTINCT m.MEDICATION_CD)::DOUBLE AS med_distinct_0_to_t,
      AVG(TRY_CAST(m.INFUSION_RATE AS DOUBLE)) AS med_infusion_mean_0_to_t,
      COUNT(*)::DOUBLE AS med_admin_rows_0_to_t
    FROM ckpt c
    JOIN enc_ok e ON e.PERSON_ID = c.PERSON_ID
    JOIN med m ON m.ENCOUNTER_ID = e.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(m.ADMIN_START_DT_TM AS TIMESTAMP) - e.admit_ts)) / 3600.0
          <= c.prediction_hour
    GROUP BY 1, 2;

    CREATE OR REPLACE VIEW proc_cap AS
    SELECT c.PERSON_ID, c.prediction_hour, COUNT(*)::DOUBLE AS proc_n_0_to_t
    FROM ckpt c
    JOIN enc_ok e ON e.PERSON_ID = c.PERSON_ID
    JOIN proc pr ON pr.ENCOUNTER_ID = e.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(pr.PROC_START_DT_TM AS TIMESTAMP) - e.admit_ts)) / 3600.0
          <= c.prediction_hour
    GROUP BY 1, 2;

    CREATE OR REPLACE VIEW scai_cap AS
    SELECT
      c.PERSON_ID, c.prediction_hour,
      MIN(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)) AS scai_first_0_to_t,
      MAX(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)) AS scai_last_0_to_t,
      AVG(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)) AS scai_mean_0_to_t,
      COALESCE(STDDEV_SAMP(TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE)), 0)::DOUBLE AS scai_std_0_to_t,
      AVG(CASE WHEN TRY_CAST(sh.SCAI_STAGE_NUM AS DOUBLE) >= 3 THEN 1.0 ELSE 0.0 END) AS scai_prop_ge3_0_to_t
    FROM ckpt c
    JOIN enc_ok e ON e.PERSON_ID = c.PERSON_ID
    JOIN scai sh ON sh.ENCOUNTER_ID = e.ENCOUNTER_ID
    WHERE TRY_CAST(sh.HOUR_FROM_ADMIT AS DOUBLE) <= c.prediction_hour
    GROUP BY 1, 2;

    CREATE OR REPLACE VIEW ce_cap AS
    SELECT
      c.PERSON_ID, c.prediction_hour,
      COUNT(*)::DOUBLE AS clinical_event_n_0_to_t,
      AVG(TRY_CAST(ce.RESULT_VAL AS DOUBLE)) FILTER (WHERE ce.EVENT_CD = 'MAP') AS mean_MAP_0_to_t,
      AVG(TRY_CAST(ce.RESULT_VAL AS DOUBLE)) FILTER (WHERE ce.EVENT_CD = 'HR') AS mean_HR_0_to_t,
      AVG(TRY_CAST(ce.RESULT_VAL AS DOUBLE)) FILTER (WHERE ce.EVENT_CD = 'LACTATE') AS mean_LACTATE_0_to_t
    FROM ckpt c
    JOIN enc_ok e ON e.PERSON_ID = c.PERSON_ID
    JOIN cevt ce ON ce.ENCOUNTER_ID = e.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(ce.EVENT_END_DT_TM AS TIMESTAMP) - e.admit_ts)) / 3600.0
          <= c.prediction_hour
    GROUP BY 1, 2;

    CREATE OR REPLACE VIEW dx_cap AS
    SELECT c.PERSON_ID, c.prediction_hour, COUNT(*)::DOUBLE AS n_diagnoses_0_to_t
    FROM ckpt c
    JOIN enc_ok e ON e.PERSON_ID = c.PERSON_ID
    JOIN dx d ON d.ENCOUNTER_ID = e.ENCOUNTER_ID
    WHERE TRY_CAST(d.DIAGNOSIS_DT_TM AS TIMESTAMP) <= e.admit_ts + (c.prediction_hour * INTERVAL '1 hour')
    GROUP BY 1, 2;

    COPY (
      SELECT
        c.PERSON_ID,
        c.prediction_hour,
        c.feature_window_hours,
        c.los_hours_total,
        c.remaining_los_hours,
        c.age_years, c.sex_bin, c.race_cd, c.ethnicity_cd,
        c.admit_type_cd, c.admit_src_cd, c.unit_cd,
        COALESCE(mc.med_distinct_0_to_t, 0) AS med_distinct_0_to_t,
        COALESCE(mc.med_infusion_mean_0_to_t, 0) AS med_infusion_mean_0_to_t,
        COALESCE(pc.proc_n_0_to_t, 0) AS proc_n_0_to_t,
        COALESCE(sc.scai_first_0_to_t, 0) AS scai_first_0_to_t,
        COALESCE(sc.scai_last_0_to_t, 0) AS scai_last_0_to_t,
        COALESCE(sc.scai_mean_0_to_t, 0) AS scai_mean_0_to_t,
        COALESCE(sc.scai_std_0_to_t, 0) AS scai_std_0_to_t,
        (COALESCE(sc.scai_last_0_to_t, 0) - COALESCE(sc.scai_first_0_to_t, 0))::DOUBLE AS scai_slope_0_to_t,
        COALESCE(sc.scai_prop_ge3_0_to_t, 0) AS scai_prop_ge3_0_to_t,
        COALESCE(cc.clinical_event_n_0_to_t, 0) AS clinical_event_n_0_to_t,
        COALESCE(cc.mean_MAP_0_to_t, 0) AS mean_MAP_0_to_t,
        COALESCE(cc.mean_HR_0_to_t, 0) AS mean_HR_0_to_t,
        COALESCE(cc.mean_LACTATE_0_to_t, 0) AS mean_LACTATE_0_to_t,
        COALESCE(mc.med_admin_rows_0_to_t, 0) / NULLIF(COALESCE(cc.clinical_event_n_0_to_t, 0) + 1.0, 0)
          AS med_per_clinical_event_0_to_t,
        COALESCE(dc.n_diagnoses_0_to_t, 0) AS n_diagnoses_0_to_t
      FROM ckpt c
      LEFT JOIN med_cap mc ON mc.PERSON_ID = c.PERSON_ID AND mc.prediction_hour = c.prediction_hour
      LEFT JOIN proc_cap pc ON pc.PERSON_ID = c.PERSON_ID AND pc.prediction_hour = c.prediction_hour
      LEFT JOIN scai_cap sc ON sc.PERSON_ID = c.PERSON_ID AND sc.prediction_hour = c.prediction_hour
      LEFT JOIN ce_cap cc ON cc.PERSON_ID = c.PERSON_ID AND cc.prediction_hour = c.prediction_hour
      LEFT JOIN dx_cap dc ON dc.PERSON_ID = c.PERSON_ID AND dc.prediction_hour = c.prediction_hour
      ORDER BY c.PERSON_ID, c.prediction_hour
    ) TO '{q(out_p)}' (FORMAT PARQUET);
    """
    con = duckdb.connect(database=":memory:")
    con.execute(sql)
    con.close()
    return out_p


def build_los_patient_frame(data_dir: Path, *, force_admit_features: bool = False) -> pd.DataFrame:
    data_dir = _resolve(data_dir)
    enc_raw = pd.read_parquet(data_dir / "encounter.parquet")
    person_raw = pd.read_parquet(data_dir / "person.parquet")
    dx_raw = pd.read_parquet(data_dir / "diagnosis.parquet")

    enc, _ = _clean_table_strings(enc_raw)
    person, _ = _clean_table_strings(person_raw)
    dx, _ = _clean_table_strings(dx_raw)

    enc = reconcile_los_hours(enc)
    incl = apply_inclusion_mask(enc, person)
    trunc = flag_truncated_los(enc, person)

    enc_cohort = enc.loc[incl].copy()
    trunc_cohort = trunc.loc[incl].copy()

    admit_path = data_dir / ADMIT_FEATURES_NAME
    if force_admit_features or not admit_path.is_file():
        materialize_admit_features(data_dir)
    admit = pd.read_parquet(admit_path)
    dx_cat = _principal_dx_category(dx)

    base = enc_cohort[
        [
            "PERSON_ID",
            "ENCOUNTER_ID",
            "REG_DT_TM",
            "DISCH_DT_TM",
            "ADMIT_TYPE_CD",
            "ADMIT_SRC_CD",
            "UNIT_CD",
            "DISCH_DISPOSITION_CD",
            "los_hours_reconciled",
            "los_reconcile_ok",
        ]
    ].merge(admit, on="PERSON_ID", how="left")
    base = base.merge(dx_cat, on="PERSON_ID", how="left")
    for c in trunc_cohort.columns:
        base[c] = trunc_cohort[c].values

    base["los_hours_total"] = pd.to_numeric(base["los_hours_reconciled"], errors="coerce")
    base["prediction_hour"] = V1_FEATURE_WINDOW_HOURS
    base["feature_window_hours"] = V1_FEATURE_WINDOW_HOURS
    base["remaining_los_hours"] = np.maximum(
        0.0, base["los_hours_total"] - base["prediction_hour"].astype(float)
    )
    base["log1p_los_hours_total"] = np.log1p(base["los_hours_total"].clip(lower=0))
    base["log1p_remaining_los_hours"] = np.log1p(base["remaining_los_hours"].clip(lower=0))
    # Legacy alias for EDA scripts
    base["los_hours"] = base["los_hours_total"]
    base["admit_type_cd"] = base["ADMIT_TYPE_CD"].astype("string").str.upper().str.strip()
    base["admit_src_cd"] = base["ADMIT_SRC_CD"].astype("string").str.upper().str.strip()
    base["unit_cd"] = base["UNIT_CD"].astype("string").str.upper().str.strip()
    return base


def write_artifacts(
    data_dir: Path,
    frame: pd.DataFrame,
    *,
    enc_raw_n: int,
    sentinel_counts: dict[str, dict[str, int]],
    reconcile_summary: dict[str, Any],
) -> None:
    data_dir = _resolve(data_dir)
    manifest = {
        "inclusion_rules": INCLUSION_RULES,
        "target_policy": TARGET_POLICY,
        "counts": {
            "encounters_raw": enc_raw_n,
            "patients_included": int(len(frame)),
            "patients_excluded": int(enc_raw_n - len(frame)),
            "los_truncated_flagged": int(frame["los_truncated"].sum()),
        },
        "outputs": {
            "patient_frame": PATIENT_FRAME_NAME,
            "checkpoint_frame": CHECKPOINT_FRAME_NAME,
            "admit_features": ADMIT_FEATURES_NAME,
            "feature_columns": FEATURE_COLUMNS_NAME,
            "prediction_policy": PREDICTION_POLICY_NAME,
            "early_window_doc": EARLY_WINDOW_DOC_NAME,
        },
        "prediction_schedule_v2": {
            "cadence_hours": 24,
            "grain": "one_row_per_person_per_prediction_hour",
            "see": PREDICTION_POLICY_NAME,
        },
    }
    (data_dir / COHORT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # v2 checkpoint features (censored at prediction_hour) — edit tiers when adding columns
    primary_features = [
        "med_infusion_mean_0_to_t",
        "scai_mean_0_to_t",
        "scai_last_0_to_t",
        "mean_MAP_0_to_t",
        "mean_LACTATE_0_to_t",
        "proc_n_0_to_t",
        "med_distinct_0_to_t",
        "clinical_event_n_0_to_t",
        "age_years",
        "scai_slope_0_to_t",
        "scai_prop_ge3_0_to_t",
        "remaining_los_hours",
    ]
    # remaining_los_hours is a LABEL — remove from primary before validate
    primary_features = [c for c in primary_features if c != LABEL_REMAINING]

    supporting_features = [
        "sex_bin",
        "scai_first_0_to_t",
        "scai_std_0_to_t",
        "med_per_clinical_event_0_to_t",
        "mean_HR_0_to_t",
        "n_diagnoses_0_to_t",
        "admit_type_cd",
        "admit_src_cd",
        "unit_cd",
        "race_cd",
        "ethnicity_cd",
        "prediction_hour",
        "feature_window_hours",
    ]
    v1_snapshot_features = [
        "med_infusion_mean_12h",
        "scai_mean_12h",
        "scai_last_12h",
        "proc_n_12h",
        "med_distinct_12h",
        "clinical_event_n_12h",
        "scai_slope_12h",
        "scai_first_12h",
        "scai_std_12h",
        "med_per_clinical_event_12h",
        "icd_prefix",
    ]
    feature_cols_v2 = primary_features + [
        c for c in supporting_features if c not in primary_features
    ]
    feature_cols_v1 = list(
        dict.fromkeys(
            v1_snapshot_features
            + [
                "age_years",
                "sex_bin",
                "admit_type_cd",
                "admit_src_cd",
                "unit_cd",
                "icd_prefix",
                "race_cd",
                "ethnicity_cd",
            ]
        )
    )

    feature_cols_v2 = validate_modeling_columns(feature_cols_v2)
    feature_cols_v1 = validate_modeling_columns(feature_cols_v1, prediction_hour=V1_FEATURE_WINDOW_HOURS, strict_v1_12h=True)
    assert_frame_separated(frame.columns.tolist(), feature_cols_v2)
    assert_frame_separated(frame.columns.tolist(), feature_cols_v1)

    policy = build_policy_dict()
    (data_dir / PREDICTION_POLICY_NAME).write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (data_dir / EARLY_WINDOW_DOC_NAME).write_text(
        json.dumps(policy["early_window_per_table"], indent=2), encoding="utf-8"
    )

    (data_dir / FEATURE_COLUMNS_NAME).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "labels": {
                    LABEL_TOTAL: "Total ICU LOS (hours); blocked from X.",
                    LABEL_REMAINING: "Hours left at prediction_hour; blocked from X.",
                },
                "label_columns": [LABEL_TOTAL, LABEL_REMAINING],
                "feature_columns_v2_checkpoint": feature_cols_v2,
                "feature_columns_v1_snapshot": feature_cols_v1,
                "primary_features": primary_features,
                "supporting_features": supporting_features,
                "blocked_label_derived": sorted(policy["forbidden_as_features"]["label_derived"]),
                "blocked_full_stay_aggregates": sorted(policy["forbidden_as_features"]["full_stay_aggregates"]),
                "blocked_combined": sorted(policy["forbidden_as_features"]["combined"]),
                "categorical_columns": [
                    "admit_type_cd",
                    "admit_src_cd",
                    "unit_cd",
                    "race_cd",
                    "ethnicity_cd",
                ],
                "flag_columns_audit_only": [
                    "los_truncated",
                    "truncated_death",
                    "truncated_hospice",
                    "truncated_disposition_cd",
                ],
                "log1p_columns": ["clinical_event_n_0_to_t", "med_per_clinical_event_0_to_t"],
                "sqrt_columns": ["med_distinct_0_to_t", "med_infusion_mean_0_to_t", "proc_n_0_to_t"],
                "prediction_schedule": policy["prediction_schedule"],
                "early_window_per_table_file": EARLY_WINDOW_DOC_NAME,
                "policy_file": PREDICTION_POLICY_NAME,
                "checkpoint_frame": CHECKPOINT_FRAME_NAME,
                "when_adding_features": "Update primary/supporting lists; run validate_modeling_columns; never use blocked_* columns.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    quality = {
        "string_sentinels_normalized": sorted(STRING_SENTINELS),
        "sentinel_replacements": sentinel_counts,
        "los_reconciliation": reconcile_summary,
        "truncation_flags": {
            "truncated_death": int(frame["truncated_death"].sum()),
            "truncated_hospice": int(frame["truncated_hospice"].sum()),
            "los_truncated_any": int(frame["los_truncated"].sum()),
        },
        "los_hours_summary": frame["los_hours"].describe().to_dict(),
    }
    (data_dir / QUALITY_REPORT_NAME).write_text(
        json.dumps(quality, indent=2, default=str), encoding="utf-8"
    )
    frame.to_parquet(data_dir / PATIENT_FRAME_NAME, index=False)

    legend_path = data_dir / "demo_profile_bucket_legend.json"
    if not legend_path.is_file():
        legend_path.write_text(json.dumps(build_demo_profile_legend(), indent=2), encoding="utf-8")


def run_prep(data_dir: Path, *, force: bool = False) -> Path:
    data_dir = _resolve(data_dir)
    enc_raw = pd.read_parquet(data_dir / "encounter.parquet")
    person_raw = pd.read_parquet(data_dir / "person.parquet")
    dx_raw = pd.read_parquet(data_dir / "diagnosis.parquet")

    _, enc_sent = _clean_table_strings(enc_raw)
    _, person_sent = _clean_table_strings(person_raw)
    _, dx_sent = _clean_table_strings(dx_raw)

    enc_rec = reconcile_los_hours(enc_raw)
    reconcile_summary = {
        "n_total": int(len(enc_rec)),
        "n_within_1h": int(enc_rec["los_reconcile_ok"].sum()),
        "n_mismatch_gt_1h": int((~enc_rec["los_reconcile_ok"]).sum()),
        "max_abs_delta_h": float(enc_rec["los_reconcile_delta_h"].max()),
        "median_abs_delta_h": float(enc_rec["los_reconcile_delta_h"].median()),
    }

    frame = build_los_patient_frame(data_dir, force_admit_features=force)
    materialize_checkpoint_frame(data_dir)
    write_artifacts(
        data_dir,
        frame,
        enc_raw_n=len(enc_raw),
        sentinel_counts={"encounter": enc_sent, "person": person_sent, "diagnosis": dx_sent},
        reconcile_summary=reconcile_summary,
    )
    return data_dir / PATIENT_FRAME_NAME


def main() -> int:
    ap = argparse.ArgumentParser(description="LOS cohort cleaning and patient frame.")
    ap.add_argument("--data-dir", type=Path, default=Path("Baptist_tester/synth_cs_data"))
    ap.add_argument("--force", action="store_true", help="Rebuild admit-features parquet")
    args = ap.parse_args()
    out = run_prep(args.data_dir, force=args.force)
    print(f"Wrote {out}")
    print(f"Manifest: {_resolve(args.data_dir) / COHORT_MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
