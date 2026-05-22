"""
DuckDB-driven patient-level features for LOS modeling (early window ≤12h).

Writes ``duckdb_los_features.parquet`` and exploratory ``duckdb_los_feature_audit.json``
(MI / Pearson vs ``los_hours_total`` on full cohort — not for train-only selection).

Adapted from ``Frontend/lib/models/mortality_duckdb_features.py``; label is LOS hours,
not ``died``. All aggregates use hours-from-admit ≤ 12 (suffix ``_12h``) in v1.

For rolling checkpoints (24h, 48h, …), set ``ROLLING_FEATURE_MODE='rolling'`` in
``los_feature_policy`` and parameterize ``EARLY_WINDOW_HOURS`` / SQL suffix via
``aggregate_suffix_for_prediction_hour()``.
"""

from __future__ import annotations

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

from mortality_demographics import LEGEND_JSON_NAME, build_demo_profile_legend  # noqa: E402

EARLY_WINDOW_HOURS = 12
SIDE_CAR_NAME = "duckdb_los_features.parquet"
AUDIT_JSON_NAME = "duckdb_los_feature_audit.json"

# Model-facing numeric columns (exclude PERSON_ID, los_hours_total, audit-only bands)
FEATURE_COLUMNS_LOS: list[str] = [
    "demo_profile_bucket",
    "med_distinct_12h",
    "med_infusion_mean_12h",
    "med_admin_rows_12h",
    "proc_n_12h",
    "proc_distinct_nom_12h",
    "scai_first_12h",
    "scai_last_12h",
    "scai_mean_12h",
    "scai_std_12h",
    "scai_slope_12h",
    "scai_prop_ge3_12h",
    "current_scai_12h",
    "clinical_event_n_12h",
    "med_per_clinical_event_12h",
    "infusion_mean_12h",
    "proc_n_bucket_12h",
    "med_distinct_bucket_12h",
    "med_residual_within_scai_12h",
    "infusion_residual_within_scai_12h",
    "proc_residual_within_scai_12h",
    "admit_type_emergency",
    "admit_type_elective",
    "admit_src_ed",
    "admit_src_outside_hospital",
    "unit_cicu",
    "unit_cvicu",
    "unit_micu",
    "dx_cat_acute_mi",
    "dx_cat_adhf",
    "dx_cat_arrhythmia_arrest",
    "dx_cat_myocarditis_cmp",
    "dx_cat_aortic_valve",
    "dx_cat_post_cardiotomy",
    "dx_cat_other",
    "n_diagnoses_12h",
]


def _q(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


def register_bundle(con: Any, data_dir: Path, tables: dict[str, pd.DataFrame] | None) -> None:
    """Register parquet bundle or in-memory cleaned tables on a DuckDB connection."""
    if tables is not None:
        con.register("enc", tables["encounter"])
        con.register("person", tables["person"])
        con.register("dx", tables["diagnosis"])
        con.register("med", tables["medication_admin"])
        con.register("proc", tables["procedure_event"])
        con.register("scai", tables["scai_stage_hourly"])
        con.register("cevt", tables["clinical_event"])
        return

    data_dir = data_dir.resolve()
    con.execute(f"CREATE OR REPLACE VIEW enc AS SELECT * FROM read_parquet('{_q(data_dir / 'encounter.parquet')}')")
    con.execute(f"CREATE OR REPLACE VIEW person AS SELECT * FROM read_parquet('{_q(data_dir / 'person.parquet')}')")
    con.execute(f"CREATE OR REPLACE VIEW dx AS SELECT * FROM read_parquet('{_q(data_dir / 'diagnosis.parquet')}')")
    con.execute(
        f"CREATE OR REPLACE VIEW med AS SELECT * FROM read_parquet('{_q(data_dir / 'medication_admin.parquet')}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW proc AS SELECT * FROM read_parquet('{_q(data_dir / 'procedure_event.parquet')}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW scai AS SELECT * FROM read_parquet('{_q(data_dir / 'scai_stage_hourly.parquet')}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW cevt AS SELECT * FROM read_parquet('{_q(data_dir / 'clinical_event.parquet')}')"
    )


def _build_los_features_sql(out_p: Path) -> str:
    h = EARLY_WINDOW_HOURS
    return f"""
    PRAGMA threads=4;

    CREATE OR REPLACE VIEW enc_ts AS
    SELECT
      e.*,
      TRY_CAST(e.REG_DT_TM AS TIMESTAMP) AS admit_ts,
      TRY_CAST(e.DISCH_DT_TM AS TIMESTAMP) AS disch_ts,
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

    CREATE OR REPLACE VIEW labels AS
    SELECT
      e.ENCOUNTER_ID,
      e.PERSON_ID,
      e.los_hours_total,
      date_diff('year', TRY_CAST(p.BIRTH_DT_TM AS TIMESTAMP), e.admit_ts)::DOUBLE AS age_years,
      CASE WHEN UPPER(TRIM(COALESCE(p.SEX_CD, ''))) IN ('M', 'MALE') THEN 1 ELSE 0 END::INT AS sex_bin,
      (
        CASE WHEN UPPER(TRIM(COALESCE(p.SEX_CD, ''))) IN ('M', 'MALE') THEN 82.0 ELSE 70.0 END
        + MOD(ABS(HASH(CAST(p.PERSON_ID AS VARCHAR))), 31)
      )::DOUBLE AS weight_kg_est,
      UPPER(TRIM(COALESCE(e.ADMIT_TYPE_CD, ''))) AS admit_type_cd,
      UPPER(TRIM(COALESCE(e.ADMIT_SRC_CD, ''))) AS admit_src_cd,
      UPPER(TRIM(COALESCE(e.UNIT_CD, ''))) AS unit_cd
    FROM enc_ts e
    JOIN person p ON p.PERSON_ID = e.PERSON_ID;

    CREATE OR REPLACE VIEW demo_bins AS
    SELECT
      ENCOUNTER_ID,
      PERSON_ID,
      los_hours_total,
      age_years,
      sex_bin,
      weight_kg_est,
      admit_type_cd,
      admit_src_cd,
      unit_cd,
      CASE WHEN age_years < 60 THEN 0 WHEN age_years < 70 THEN 1 WHEN age_years < 80 THEN 2 ELSE 3 END::INT AS age_bin,
      CASE WHEN weight_kg_est < 65 THEN 0 WHEN weight_kg_est <= 85 THEN 1 ELSE 2 END::INT AS weight_bin
    FROM labels;

    CREATE OR REPLACE VIEW demo_profile AS
    SELECT
      ENCOUNTER_ID,
      PERSON_ID,
      los_hours_total,
      age_years,
      weight_kg_est,
      sex_bin,
      age_bin,
      weight_bin,
      admit_type_cd,
      admit_src_cd,
      unit_cd,
      (age_bin * 6 + weight_bin * 2 + sex_bin)::INT AS demo_profile_bucket,
      CASE WHEN admit_type_cd = 'EMERGENCY' THEN 1 ELSE 0 END::INT AS admit_type_emergency,
      CASE WHEN admit_type_cd = 'ELECTIVE' THEN 1 ELSE 0 END::INT AS admit_type_elective,
      CASE WHEN admit_src_cd = 'ED' THEN 1 ELSE 0 END::INT AS admit_src_ed,
      CASE WHEN admit_src_cd = 'OUTSIDE_HOSPITAL' THEN 1 ELSE 0 END::INT AS admit_src_outside_hospital,
      CASE WHEN unit_cd = 'CICU' THEN 1 ELSE 0 END::INT AS unit_cicu,
      CASE WHEN unit_cd = 'CVICU' THEN 1 ELSE 0 END::INT AS unit_cvicu,
      CASE WHEN unit_cd = 'MICU' THEN 1 ELSE 0 END::INT AS unit_micu
    FROM demo_bins;

    CREATE OR REPLACE VIEW med_time AS
    SELECT
      m.PERSON_ID,
      TRY_CAST(m.INFUSION_RATE AS DOUBLE) AS ir,
      EXTRACT(EPOCH FROM (TRY_CAST(m.ADMIN_START_DT_TM AS TIMESTAMP) - e.admit_ts)) / 3600.0 AS hrs
    FROM med m
    JOIN enc_ts e ON e.ENCOUNTER_ID = m.ENCOUNTER_ID AND e.PERSON_ID = m.PERSON_ID
    WHERE TRY_CAST(m.INFUSION_RATE AS DOUBLE) IS NOT NULL;

    CREATE OR REPLACE VIEW med_early AS
    SELECT
      m.PERSON_ID,
      COUNT(DISTINCT m.MEDICATION_CD)::DOUBLE AS med_distinct_12h,
      AVG(TRY_CAST(m.INFUSION_RATE AS DOUBLE)) AS med_infusion_mean_12h,
      COUNT(*)::DOUBLE AS med_admin_rows_12h
    FROM med m
    JOIN enc_ts e ON e.ENCOUNTER_ID = m.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(m.ADMIN_START_DT_TM AS TIMESTAMP) - e.admit_ts)) / 3600.0 <= {h}
    GROUP BY 1;

    CREATE OR REPLACE VIEW med_inf_12h AS
    SELECT PERSON_ID, AVG(ir) AS infusion_mean_12h
    FROM med_time
    WHERE hrs IS NOT NULL AND hrs <= {h}
    GROUP BY 1;

    CREATE OR REPLACE VIEW proc_early AS
    SELECT
      pr.PERSON_ID,
      COUNT(*)::DOUBLE AS proc_n_12h,
      COUNT(DISTINCT pr.NOMENCLATURE_CD)::DOUBLE AS proc_distinct_nom_12h
    FROM proc pr
    JOIN enc_ts e ON e.ENCOUNTER_ID = pr.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(pr.PROC_START_DT_TM AS TIMESTAMP) - e.admit_ts)) / 3600.0 <= {h}
    GROUP BY 1;

    CREATE OR REPLACE VIEW scai_enc AS
    SELECT s.*, e.PERSON_ID AS pid, e.admit_ts
    FROM scai s
    JOIN enc_ts e ON e.ENCOUNTER_ID = s.ENCOUNTER_ID;

    CREATE OR REPLACE VIEW scai_stage AS
    SELECT
      pid AS PERSON_ID,
      TRY_CAST(SCAI_STAGE_NUM AS DOUBLE) AS sn,
      TRY_CAST(HOUR_FROM_ADMIT AS DOUBLE) AS hr,
      TRY_CAST(EVENT_DT_TM AS TIMESTAMP) AS ev
    FROM scai_enc
    WHERE TRY_CAST(HOUR_FROM_ADMIT AS DOUBLE) <= {h};

    CREATE OR REPLACE VIEW current_scai_12h_tbl AS
    SELECT PERSON_ID, sn AS current_scai_12h
    FROM (
      SELECT PERSON_ID, sn,
             ROW_NUMBER() OVER (PARTITION BY PERSON_ID ORDER BY ev DESC NULLS LAST, hr DESC NULLS LAST) AS rn
      FROM scai_stage
    ) t
    WHERE rn = 1;

    CREATE OR REPLACE VIEW scai_traj AS
    SELECT
      PERSON_ID,
      MIN(sn) AS scai_first_12h,
      MAX(sn) AS scai_last_12h,
      AVG(sn) AS scai_mean_12h,
      COALESCE(STDDEV_SAMP(sn), 0)::DOUBLE AS scai_std_12h,
      AVG(CASE WHEN sn >= 3 THEN 1.0 ELSE 0.0 END) AS scai_prop_ge3_12h
    FROM scai_stage
    GROUP BY 1;

    CREATE OR REPLACE VIEW clinical_early AS
    SELECT c.PERSON_ID, COUNT(*)::DOUBLE AS clinical_event_n_12h
    FROM cevt c
    JOIN enc_ts e ON e.ENCOUNTER_ID = c.ENCOUNTER_ID
    WHERE EXTRACT(EPOCH FROM (TRY_CAST(c.EVENT_END_DT_TM AS TIMESTAMP) - e.admit_ts)) / 3600.0 <= {h}
    GROUP BY 1;

    CREATE OR REPLACE VIEW dx_early AS
    SELECT
      e.PERSON_ID,
      COUNT(*)::DOUBLE AS n_diagnoses_12h,
      MIN_BY(UPPER(TRIM(COALESCE(d.NOMENCLATURE_CD, ''))), TRY_CAST(d.DIAG_PRIORITY AS INT)) AS principal_icd_12h
    FROM dx d
    JOIN enc_ts e ON e.ENCOUNTER_ID = d.ENCOUNTER_ID
    WHERE TRY_CAST(d.DIAGNOSIS_DT_TM AS TIMESTAMP) <= e.admit_ts + INTERVAL '{h} hours'
    GROUP BY 1;

    CREATE OR REPLACE VIEW dx_flags AS
    SELECT
      PERSON_ID,
      principal_icd_12h,
      CASE WHEN principal_icd_12h LIKE 'I21%' THEN 1 ELSE 0 END::INT AS dx_cat_acute_mi,
      CASE WHEN principal_icd_12h LIKE 'I50%' THEN 1 ELSE 0 END::INT AS dx_cat_adhf,
      CASE WHEN principal_icd_12h LIKE 'I46%' OR principal_icd_12h LIKE 'I49%' THEN 1 ELSE 0 END::INT AS dx_cat_arrhythmia_arrest,
      CASE WHEN principal_icd_12h LIKE 'I40%' OR principal_icd_12h LIKE 'I42%' THEN 1 ELSE 0 END::INT AS dx_cat_myocarditis_cmp,
      CASE WHEN principal_icd_12h LIKE 'I71%' OR principal_icd_12h LIKE 'I35%' THEN 1 ELSE 0 END::INT AS dx_cat_aortic_valve,
      CASE WHEN principal_icd_12h LIKE 'Z95%' THEN 1 ELSE 0 END::INT AS dx_cat_post_cardiotomy,
      CASE
        WHEN principal_icd_12h IS NULL OR principal_icd_12h = '' THEN 1
        WHEN principal_icd_12h LIKE 'I21%' OR principal_icd_12h LIKE 'I50%'
          OR principal_icd_12h LIKE 'I46%' OR principal_icd_12h LIKE 'I49%'
          OR principal_icd_12h LIKE 'I40%' OR principal_icd_12h LIKE 'I42%'
          OR principal_icd_12h LIKE 'I71%' OR principal_icd_12h LIKE 'I35%'
          OR principal_icd_12h LIKE 'Z95%' THEN 0
        ELSE 1
      END::INT AS dx_cat_other
    FROM dx_early;

    CREATE OR REPLACE VIEW base_join AS
    SELECT
      dp.ENCOUNTER_ID,
      dp.PERSON_ID,
      dp.los_hours_total,
      dp.demo_profile_bucket,
      dp.age_years,
      dp.weight_kg_est,
      dp.sex_bin,
      dp.age_bin,
      dp.weight_bin,
      dp.admit_type_cd,
      dp.admit_src_cd,
      dp.unit_cd,
      dp.admit_type_emergency,
      dp.admit_type_elective,
      dp.admit_src_ed,
      dp.admit_src_outside_hospital,
      dp.unit_cicu,
      dp.unit_cvicu,
      dp.unit_micu,
      COALESCE(me.med_distinct_12h, 0) AS med_distinct_12h,
      COALESCE(me.med_infusion_mean_12h, 0) AS med_infusion_mean_12h,
      COALESCE(me.med_admin_rows_12h, 0) AS med_admin_rows_12h,
      COALESCE(mi.infusion_mean_12h, me.med_infusion_mean_12h, 0) AS infusion_mean_12h,
      COALESCE(pe.proc_n_12h, 0) AS proc_n_12h,
      COALESCE(pe.proc_distinct_nom_12h, 0) AS proc_distinct_nom_12h,
      COALESCE(st.scai_first_12h, cs.current_scai_12h, 0) AS scai_first_12h,
      COALESCE(st.scai_last_12h, cs.current_scai_12h, 0) AS scai_last_12h,
      COALESCE(st.scai_mean_12h, cs.current_scai_12h, 0) AS scai_mean_12h,
      COALESCE(st.scai_std_12h, 0) AS scai_std_12h,
      COALESCE(st.scai_prop_ge3_12h, 0) AS scai_prop_ge3_12h,
      COALESCE(cs.current_scai_12h, st.scai_last_12h, 0) AS current_scai_12h,
      COALESCE(ce.clinical_event_n_12h, 0) AS clinical_event_n_12h,
      COALESCE(dx.n_diagnoses_12h, 0) AS n_diagnoses_12h,
      df.principal_icd_12h,
      COALESCE(df.dx_cat_acute_mi, 0) AS dx_cat_acute_mi,
      COALESCE(df.dx_cat_adhf, 0) AS dx_cat_adhf,
      COALESCE(df.dx_cat_arrhythmia_arrest, 0) AS dx_cat_arrhythmia_arrest,
      COALESCE(df.dx_cat_myocarditis_cmp, 0) AS dx_cat_myocarditis_cmp,
      COALESCE(df.dx_cat_aortic_valve, 0) AS dx_cat_aortic_valve,
      COALESCE(df.dx_cat_post_cardiotomy, 0) AS dx_cat_post_cardiotomy,
      COALESCE(df.dx_cat_other, 1) AS dx_cat_other
    FROM demo_profile dp
    LEFT JOIN med_early me ON me.PERSON_ID = dp.PERSON_ID
    LEFT JOIN med_inf_12h mi ON mi.PERSON_ID = dp.PERSON_ID
    LEFT JOIN proc_early pe ON pe.PERSON_ID = dp.PERSON_ID
    LEFT JOIN scai_traj st ON st.PERSON_ID = dp.PERSON_ID
    LEFT JOIN current_scai_12h_tbl cs ON cs.PERSON_ID = dp.PERSON_ID
    LEFT JOIN clinical_early ce ON ce.PERSON_ID = dp.PERSON_ID
    LEFT JOIN dx_early dx ON dx.PERSON_ID = dp.PERSON_ID
    LEFT JOIN dx_flags df ON df.PERSON_ID = dp.PERSON_ID;

    CREATE OR REPLACE VIEW enriched AS
    SELECT
      ENCOUNTER_ID,
      PERSON_ID,
      los_hours_total,
      demo_profile_bucket,
      age_years,
      weight_kg_est,
      sex_bin,
      age_bin,
      weight_bin,
      admit_type_cd,
      admit_src_cd,
      unit_cd,
      admit_type_emergency,
      admit_type_elective,
      admit_src_ed,
      admit_src_outside_hospital,
      unit_cicu,
      unit_cvicu,
      unit_micu,
      med_distinct_12h,
      med_infusion_mean_12h,
      med_admin_rows_12h,
      infusion_mean_12h,
      proc_n_12h,
      proc_distinct_nom_12h,
      scai_first_12h,
      scai_last_12h,
      scai_mean_12h,
      scai_std_12h,
      scai_prop_ge3_12h,
      current_scai_12h,
      (COALESCE(scai_last_12h, current_scai_12h) - COALESCE(scai_first_12h, current_scai_12h))::DOUBLE AS scai_slope_12h,
      clinical_event_n_12h,
      (med_admin_rows_12h / NULLIF(clinical_event_n_12h + 1.0, 0))::DOUBLE AS med_per_clinical_event_12h,
      n_diagnoses_12h,
      principal_icd_12h,
      dx_cat_acute_mi,
      dx_cat_adhf,
      dx_cat_arrhythmia_arrest,
      dx_cat_myocarditis_cmp,
      dx_cat_aortic_valve,
      dx_cat_post_cardiotomy,
      dx_cat_other,
      CASE WHEN proc_n_12h <= 0 THEN 0.0 WHEN proc_n_12h <= 3 THEN 1.0 ELSE 2.0 END AS proc_n_bucket_12h,
      CASE WHEN med_distinct_12h <= 2 THEN 0.0 WHEN med_distinct_12h <= 5 THEN 1.0 ELSE 2.0 END AS med_distinct_bucket_12h,
      med_distinct_12h - AVG(med_distinct_12h) OVER (PARTITION BY CAST(ROUND(current_scai_12h) AS INT)) AS med_residual_within_scai_12h,
      med_infusion_mean_12h - AVG(med_infusion_mean_12h) OVER (PARTITION BY CAST(ROUND(current_scai_12h) AS INT)) AS infusion_residual_within_scai_12h,
      proc_n_12h - AVG(proc_n_12h) OVER (PARTITION BY CAST(ROUND(current_scai_12h) AS INT)) AS proc_residual_within_scai_12h
    FROM base_join;

    COPY enriched TO '{_q(out_p)}' (FORMAT PARQUET);
    """


def materialize_duckdb_los_features(
    data_dir: Path,
    *,
    tables: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Run DuckDB SQL and write ``duckdb_los_features.parquet``."""
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("Install duckdb: pip install duckdb") from exc

    data_dir = data_dir.resolve()
    out_p = data_dir / SIDE_CAR_NAME
    con = duckdb.connect(database=":memory:")
    register_bundle(con, data_dir, tables)
    con.execute(_build_los_features_sql(out_p))
    con.close()

    legend_path = data_dir / LEGEND_JSON_NAME
    if not legend_path.is_file():
        legend_path.write_text(json.dumps(build_demo_profile_legend(), indent=2), encoding="utf-8")
    return out_p


def run_stratified_los_audit(data_dir: Path) -> dict[str, Any]:
    """Within rounded ``current_scai_12h``, mean LOS and early-window feature means."""
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("Install duckdb: pip install duckdb") from exc

    data_dir = data_dir.resolve()
    p_side = _q(data_dir / SIDE_CAR_NAME)
    sql = f"""
    CREATE OR REPLACE VIEW e AS SELECT * FROM read_parquet('{p_side}');
    SELECT
      CAST(ROUND(current_scai_12h) AS INT) AS scai_bin,
      COUNT(*)::INT AS n,
      AVG(los_hours_total) AS mean_los_hours,
      AVG(med_distinct_12h) AS mean_med_distinct_12h,
      AVG(med_infusion_mean_12h) AS mean_infusion_12h,
      AVG(proc_n_12h) AS mean_proc_n_12h,
      AVG(clinical_event_n_12h) AS mean_clinical_event_n_12h
    FROM e
    GROUP BY 1
    ORDER BY 1;
    """
    con = duckdb.connect(database=":memory:")
    df = con.execute(sql).fetchdf()
    con.close()
    return {"stratified_means_by_scai_bin_12h": df.to_dict(orient="records")}


def _mutual_info_vs_los(X: pd.DataFrame, y: pd.Series, feature_cols: list[str], *, random_state: int = 0) -> dict[str, float]:
    from sklearn.feature_selection import mutual_info_regression

    Xm = X[feature_cols].copy()
    for c in feature_cols:
        med = Xm[c].median()
        Xm[c] = pd.to_numeric(Xm[c], errors="coerce").fillna(0.0 if pd.isna(med) else float(med))
    y_num = pd.to_numeric(y, errors="coerce")
    mask = y_num.notna()
    mi = mutual_info_regression(Xm.loc[mask], y_num.loc[mask], random_state=random_state, discrete_features=False)
    return {feature_cols[i]: float(mi[i]) for i in range(len(feature_cols))}


def _pearson_vs_los(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> dict[str, float]:
    y_num = pd.to_numeric(y, errors="coerce")
    out: dict[str, float] = {}
    for c in feature_cols:
        x = pd.to_numeric(X[c], errors="coerce")
        pair = pd.concat([x, y_num], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].std() == 0:
            out[c] = 0.0
            continue
        out[c] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return out


def write_los_feature_audit_json(
    data_dir: Path,
    *,
    mi_full_cohort: dict[str, float] | None = None,
    pearson_full_cohort: dict[str, float] | None = None,
    stratified_audit: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
) -> Path:
    data_dir = data_dir.resolve()
    audit = {
        "sidecar_parquet": SIDE_CAR_NAME,
        "early_window_hours": EARLY_WINDOW_HOURS,
        "label_column": "los_hours_total",
        "feature_columns_audited": feature_columns or FEATURE_COLUMNS_LOS,
        "note_mi_full_cohort": (
            "mutual_information_full_cohort and pearson_vs_los_hours_total are exploratory "
            "(fit on all patients before split); use train-only metrics when training LOS models."
        ),
        "mutual_information_full_cohort": mi_full_cohort or {},
        "pearson_vs_los_hours_total": pearson_full_cohort or {},
        "stratified_los_audit": stratified_audit or {},
        "residual_warning": (
            "`*_residual_within_scai_12h` use windowed means over all rows in the sidecar; "
            "recompute within train fold before production modeling."
        ),
    }
    out = data_dir / AUDIT_JSON_NAME
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


def build_exploratory_audit(data_dir: Path, *, random_state: int = 0) -> Path:
    """MI + Pearson vs ``los_hours_total`` on full sidecar cohort."""
    data_dir = data_dir.resolve()
    side = pd.read_parquet(data_dir / SIDE_CAR_NAME)
    cols = [c for c in FEATURE_COLUMNS_LOS if c in side.columns]
    y = side["los_hours_total"]
    mi = _mutual_info_vs_los(side, y, cols, random_state=random_state)
    pearson = _pearson_vs_los(side, y, cols)
    return write_los_feature_audit_json(
        data_dir,
        mi_full_cohort=mi,
        pearson_full_cohort=pearson,
        stratified_audit=run_stratified_los_audit(data_dir),
        feature_columns=cols,
    )


def ensure_duckdb_los_sidecar(
    data_dir: Path, *, force: bool = False, tables: dict[str, pd.DataFrame] | None = None
) -> Path:
    data_dir = data_dir.resolve()
    path = data_dir / SIDE_CAR_NAME
    if force or not path.is_file():
        materialize_duckdb_los_features(data_dir, tables=tables)
    return path


def merge_los_duckdb_features(
    base: pd.DataFrame,
    data_dir: Path,
    *,
    force: bool = False,
    tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Left-join sidecar onto ``base`` (``ENCOUNTER_ID`` or ``PERSON_ID``). Drops duplicate labels."""
    ensure_duckdb_los_sidecar(data_dir, force=force, tables=tables)
    side = pd.read_parquet(data_dir / SIDE_CAR_NAME)
    drop = [c for c in side.columns if c in base.columns and c not in ("PERSON_ID", "ENCOUNTER_ID")]
    if "los_hours_total" in base.columns and "los_hours_total" in side.columns:
        drop.append("los_hours_total")
    side = side.drop(columns=[c for c in drop if c in side.columns], errors="ignore")
    on = "ENCOUNTER_ID" if "ENCOUNTER_ID" in base.columns and "ENCOUNTER_ID" in side.columns else "PERSON_ID"
    return base.merge(side, on=on, how="left", suffixes=("", "_duck"))


def export_admit_features_subset(data_dir: Path, *, tables: dict[str, pd.DataFrame] | None = None) -> Path:
    """Write ``los_admit_features.parquet`` (legacy v1 subset) from the DuckDB sidecar."""
    data_dir = data_dir.resolve()
    ensure_duckdb_los_sidecar(data_dir, tables=tables)
    side = pd.read_parquet(data_dir / SIDE_CAR_NAME)
    subset_cols = [
        "PERSON_ID",
        "age_years",
        "sex_bin",
        "med_distinct_12h",
        "med_infusion_mean_12h",
        "proc_n_12h",
        "scai_first_12h",
        "scai_last_12h",
        "scai_mean_12h",
        "scai_std_12h",
        "scai_slope_12h",
        "clinical_event_n_12h",
        "med_per_clinical_event_12h",
    ]
    # race/ethnicity not in sidecar — keep from person merge in patient frame
    out = data_dir / "los_admit_features.parquet"
    side[[c for c in subset_cols if c in side.columns]].to_parquet(out, index=False)
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Materialize DuckDB LOS feature sidecar + audit.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    data_dir = args.data_dir.resolve() if args.data_dir.is_absolute() else (_REPO / args.data_dir).resolve()
    materialize_duckdb_los_features(data_dir)
    export_admit_features_subset(data_dir)
    audit = build_exploratory_audit(data_dir)
    print(f"Wrote {data_dir / SIDE_CAR_NAME}")
    print(f"Audit: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
