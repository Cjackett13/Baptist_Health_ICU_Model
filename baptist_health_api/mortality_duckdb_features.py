"""
DuckDB-driven patient-level features for mortality modeling.

Builds a sidecar parquet (``duckdb_patient_features.parquet``) with:
  - ``demo_profile_bucket``: ordinal 0…23 = age band × weight band × sex (see ``mortality_demographics.py``)
  - ``age_years`` / ``weight_kg_est`` / band columns (audit only; not all are model inputs)
  - Early-window (≤12h) SCAI trajectory (``scai_slope_12h``, ``scai_max``, ``scai_increase_count``)
  - Fraction of hourly SCAI at stage D–E (``scai_prop_ge3``, numeric ``sn >= 3``)
  - Infusion: early−overall difference and above-mean flag (avoids ratio≈1.0 COALESCE artifacts)
  - Clinical velocity: ``clinical_events_last_4h``, rates per ICU hour (med/proc/events)
  - Demographics: ``demo_profile_bucket``, ``frailty_index`` (age/weight/sex combined)
  - Base med/proc/SCAI fields for residuals (``*_residual_within_scai`` computed train-fold in ``mortality_model.prepare_train_test_features``)
  - Ordinal med/procedure count buckets

Requires: ``pip install duckdb``

Synthetic bundle paths default to ``Baptist_tester/synth_cs_data/*.parquet``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mortality_demographics import LEGEND_JSON_NAME, build_demo_profile_legend

SIDE_CAR_NAME = "duckdb_patient_features.parquet"
AUDIT_JSON_NAME = "duckdb_mortality_feature_audit.json"


def _q(p: Path) -> str:
    """Single-quoted path for DuckDB read_parquet."""
    return str(p.resolve()).replace("'", "''")


def materialize_duckdb_patient_features(data_dir: Path) -> Path:
    """
    Run DuckDB SQL over the parquet bundle and write ``duckdb_patient_features.parquet``.

    Returns path to the written parquet.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install duckdb: pip install duckdb") from exc

    data_dir = data_dir.resolve()
    p_enc = _q(data_dir / "encounter.parquet")
    p_person = _q(data_dir / "person.parquet")
    p_med = _q(data_dir / "medication_admin.parquet")
    p_proc = _q(data_dir / "procedure_event.parquet")
    p_scai = _q(data_dir / "scai_stage_hourly.parquet")
    p_ce = _q(data_dir / "clinical_event.parquet")
    out_p = data_dir / SIDE_CAR_NAME

    sql = f"""
    PRAGMA threads=4;

    CREATE OR REPLACE VIEW enc AS SELECT * FROM read_parquet('{p_enc}');
    CREATE OR REPLACE VIEW person AS SELECT * FROM read_parquet('{p_person}');
    CREATE OR REPLACE VIEW med AS SELECT * FROM read_parquet('{p_med}');
    CREATE OR REPLACE VIEW proc AS SELECT * FROM read_parquet('{p_proc}');
    CREATE OR REPLACE VIEW scai AS SELECT * FROM read_parquet('{p_scai}');
    CREATE OR REPLACE VIEW cevt AS SELECT * FROM read_parquet('{p_ce}');

    CREATE OR REPLACE VIEW labels AS
    SELECT
      e.PERSON_ID,
      (p.DECEASED_DT_TM IS NOT NULL)::INT AS died,
      date_diff(
        'year',
        TRY_CAST(p.BIRTH_DT_TM AS TIMESTAMP),
        TRY_CAST(e.REG_DT_TM AS TIMESTAMP)
      )::DOUBLE AS age_years,
      CASE
        WHEN UPPER(TRIM(COALESCE(p.SEX_CD, ''))) IN ('M', 'MALE') THEN 1
        ELSE 0
      END::INT AS sex_bin,
      (
        CASE
          WHEN UPPER(TRIM(COALESCE(p.SEX_CD, ''))) IN ('M', 'MALE') THEN 82.0
          ELSE 70.0
        END
        + MOD(ABS(HASH(CAST(p.PERSON_ID AS VARCHAR))), 31)
      )::DOUBLE AS weight_kg_est
    FROM enc e
    JOIN person p ON p.PERSON_ID = e.PERSON_ID;

    CREATE OR REPLACE VIEW demo_bins AS
    SELECT
      PERSON_ID,
      died,
      age_years,
      sex_bin,
      weight_kg_est,
      CASE
        WHEN age_years < 60 THEN 0
        WHEN age_years < 70 THEN 1
        WHEN age_years < 80 THEN 2
        ELSE 3
      END::INT AS age_bin,
      CASE
        WHEN weight_kg_est < 65 THEN 0
        WHEN weight_kg_est <= 85 THEN 1
        ELSE 2
      END::INT AS weight_bin
    FROM labels;

    CREATE OR REPLACE VIEW demo_profile AS
    SELECT
      PERSON_ID,
      died,
      age_years,
      weight_kg_est,
      sex_bin,
      age_bin,
      weight_bin,
      (age_bin * 6 + weight_bin * 2 + sex_bin)::INT AS demo_profile_bucket
    FROM demo_bins;

    CREATE OR REPLACE VIEW med_agg AS
    SELECT
      m.PERSON_ID,
      COUNT(DISTINCT m.MEDICATION_CD)::DOUBLE AS med_distinct,
      AVG(TRY_CAST(m.INFUSION_RATE AS DOUBLE)) AS med_infusion_mean,
      COUNT(*)::DOUBLE AS med_admin_rows
    FROM med m
    GROUP BY 1;

    CREATE OR REPLACE VIEW med_time AS
    SELECT
      m.PERSON_ID,
      TRY_CAST(m.INFUSION_RATE AS DOUBLE) AS ir,
      EXTRACT(EPOCH FROM (
        TRY_CAST(m.ADMIN_START_DT_TM AS TIMESTAMP) - TRY_CAST(e.REG_DT_TM AS TIMESTAMP)
      )) / 3600.0 AS hrs
    FROM med m
    JOIN enc e ON e.ENCOUNTER_ID = m.ENCOUNTER_ID AND e.PERSON_ID = m.PERSON_ID
    WHERE TRY_CAST(m.INFUSION_RATE AS DOUBLE) IS NOT NULL;

    CREATE OR REPLACE VIEW med_early AS
    SELECT
      PERSON_ID,
      AVG(ir) FILTER (WHERE hrs IS NOT NULL AND hrs <= 12) AS inf_mean_12h,
      AVG(ir) AS inf_mean_all
    FROM med_time
    GROUP BY 1;

    CREATE OR REPLACE VIEW proc_agg AS
    SELECT
      pr.PERSON_ID,
      COUNT(*)::DOUBLE AS proc_n,
      COUNT(DISTINCT pr.NOMENCLATURE_CD)::DOUBLE AS proc_distinct_nom
    FROM proc pr
    GROUP BY 1;

    CREATE OR REPLACE VIEW scai_enc AS
    SELECT s.*, e.PERSON_ID AS pid
    FROM scai s
    JOIN enc e ON e.ENCOUNTER_ID = s.ENCOUNTER_ID;

    CREATE OR REPLACE VIEW scai_stage AS
    SELECT
      pid AS PERSON_ID,
      TRY_CAST(SCAI_STAGE_NUM AS DOUBLE) AS sn,
      TRY_CAST(HOUR_FROM_ADMIT AS DOUBLE) AS hr,
      TRY_CAST(EVENT_DT_TM AS TIMESTAMP) AS ev
    FROM scai_enc;

    CREATE OR REPLACE VIEW current_scai_tbl AS
    SELECT PERSON_ID, sn AS current_scai
    FROM (
      SELECT PERSON_ID, sn,
             ROW_NUMBER() OVER (
               PARTITION BY PERSON_ID ORDER BY ev DESC NULLS LAST, hr DESC NULLS LAST
             ) AS rn
      FROM scai_stage
    ) t
    WHERE rn = 1;

    CREATE OR REPLACE VIEW scai_prop AS
    SELECT
      PERSON_ID,
      AVG(CASE WHEN sn >= 3 THEN 1.0 ELSE 0.0 END) AS scai_prop_ge3,
      STDDEV_SAMP(sn) AS scai_std
    FROM scai_stage
    GROUP BY 1;

    CREATE OR REPLACE VIEW scai_early AS
    SELECT
      PERSON_ID,
      ARG_MIN(sn, ev) FILTER (WHERE hr IS NOT NULL AND hr <= 12) AS scai_first_12h,
      ARG_MAX(sn, ev) FILTER (WHERE hr IS NOT NULL AND hr <= 12) AS scai_last_12h
    FROM scai_stage
    GROUP BY 1;

    CREATE OR REPLACE VIEW scai_traj AS
    SELECT
      PERSON_ID,
      sn,
      LAG(sn) OVER (
        PARTITION BY PERSON_ID ORDER BY ev NULLS LAST, hr NULLS LAST
      ) AS prev_sn
    FROM scai_stage;

    CREATE OR REPLACE VIEW scai_vol AS
    SELECT
      PERSON_ID,
      MAX(sn)::DOUBLE AS scai_max,
      SUM(
        CASE WHEN prev_sn IS NOT NULL AND sn > prev_sn THEN 1 ELSE 0 END
      )::DOUBLE AS scai_increase_count
    FROM scai_traj
    GROUP BY 1;

    CREATE OR REPLACE VIEW enc_los AS
    SELECT
      PERSON_ID,
      MAX(COALESCE(TRY_CAST(LOS_HOURS AS DOUBLE), 24.0))::DOUBLE AS los_hours
    FROM enc
    GROUP BY 1;

    CREATE OR REPLACE VIEW ce_time AS
    SELECT
      c.PERSON_ID,
      EXTRACT(EPOCH FROM (
        TRY_CAST(c.EVENT_END_DT_TM AS TIMESTAMP) - TRY_CAST(e.REG_DT_TM AS TIMESTAMP)
      )) / 3600.0 AS hrs
    FROM cevt c
    JOIN enc e ON e.ENCOUNTER_ID = c.ENCOUNTER_ID AND e.PERSON_ID = c.PERSON_ID
    WHERE TRY_CAST(c.EVENT_END_DT_TM AS TIMESTAMP) IS NOT NULL;

    CREATE OR REPLACE VIEW clinical_n AS
    SELECT
      PERSON_ID,
      COUNT(*)::DOUBLE AS clinical_event_n,
      COUNT(*) FILTER (WHERE hrs IS NOT NULL AND hrs >= 0 AND hrs <= 4)::DOUBLE AS clinical_events_last_4h
    FROM ce_time
    GROUP BY 1;

    CREATE OR REPLACE VIEW base_join AS
    SELECT
      dp.PERSON_ID,
      dp.died,
      dp.demo_profile_bucket,
      dp.age_years,
      dp.weight_kg_est,
      dp.sex_bin,
      dp.age_bin,
      dp.weight_bin,
      COALESCE(ma.med_distinct, 0) AS med_distinct,
      COALESCE(ma.med_infusion_mean, 0) AS med_infusion_mean,
      COALESCE(pa.proc_n, 0) AS proc_n,
      COALESCE(pa.proc_distinct_nom, 0) AS proc_distinct_nom,
      COALESCE(cs.current_scai, 0) AS current_scai,
      COALESCE(sp.scai_prop_ge3, 0) AS scai_prop_ge3,
      COALESCE(sp.scai_std, 0) AS scai_std,
      COALESCE(se.scai_first_12h, cs.current_scai) AS scai_first_12h,
      COALESCE(se.scai_last_12h, cs.current_scai) AS scai_last_12h,
      COALESCE(sv.scai_max, cs.current_scai) AS scai_max,
      COALESCE(sv.scai_increase_count, 0) AS scai_increase_count,
      COALESCE(me.inf_mean_12h, ma.med_infusion_mean) AS inf_mean_12h,
      COALESCE(me.inf_mean_all, ma.med_infusion_mean) AS inf_mean_all,
      COALESCE(cn.clinical_event_n, 0) AS clinical_event_n,
      COALESCE(cn.clinical_events_last_4h, 0) AS clinical_events_last_4h,
      COALESCE(ma.med_admin_rows, 0) AS med_admin_rows,
      COALESCE(el.los_hours, 24.0) AS los_hours
    FROM demo_profile dp
    LEFT JOIN med_agg ma ON ma.PERSON_ID = dp.PERSON_ID
    LEFT JOIN med_early me ON me.PERSON_ID = dp.PERSON_ID
    LEFT JOIN proc_agg pa ON pa.PERSON_ID = dp.PERSON_ID
    LEFT JOIN current_scai_tbl cs ON cs.PERSON_ID = dp.PERSON_ID
    LEFT JOIN scai_prop sp ON sp.PERSON_ID = dp.PERSON_ID
    LEFT JOIN scai_early se ON se.PERSON_ID = dp.PERSON_ID
    LEFT JOIN scai_vol sv ON sv.PERSON_ID = dp.PERSON_ID
    LEFT JOIN clinical_n cn ON cn.PERSON_ID = dp.PERSON_ID
    LEFT JOIN enc_los el ON el.PERSON_ID = dp.PERSON_ID;

    CREATE OR REPLACE VIEW enriched AS
    SELECT
      PERSON_ID,
      died,
      demo_profile_bucket,
      age_years,
      weight_kg_est,
      sex_bin,
      age_bin,
      weight_bin,
      med_distinct,
      med_infusion_mean,
      proc_n,
      proc_distinct_nom,
      current_scai,
      scai_prop_ge3,
      COALESCE(scai_std, 0)::DOUBLE AS scai_std,
      scai_max,
      scai_increase_count,
      (COALESCE(scai_last_12h, current_scai) - COALESCE(scai_first_12h, current_scai))::DOUBLE AS scai_slope_12h,
      (COALESCE(inf_mean_12h, 0) - COALESCE(inf_mean_all, 0))::DOUBLE AS inf_early_minus_all,
      CASE
        WHEN inf_mean_all IS NOT NULL AND inf_mean_12h IS NOT NULL
             AND inf_mean_12h > inf_mean_all * 1.02 THEN 1
        ELSE 0
      END::INT AS infusion_early_above_mean,
      CASE WHEN inf_mean_all IS NULL OR inf_mean_all = 0 THEN 1.0
           ELSE COALESCE(inf_mean_12h, inf_mean_all) / inf_mean_all END AS infusion_early_over_all_ratio,
      clinical_event_n,
      clinical_events_last_4h,
      (clinical_event_n / NULLIF(los_hours, 1.0))::DOUBLE AS clinical_events_per_icu_hour,
      (med_admin_rows / NULLIF(los_hours, 1.0))::DOUBLE AS med_admin_per_icu_hour,
      (proc_n / NULLIF(los_hours, 1.0))::DOUBLE AS proc_per_icu_hour,
      (med_admin_rows / NULLIF(clinical_event_n + 1.0, 0))::DOUBLE AS med_per_clinical_event,
      (age_years / GREATEST(weight_kg_est, 45.0) * (1.0 + 0.12 * sex_bin))::DOUBLE AS frailty_index,
      CASE
        WHEN proc_n <= 0 THEN 0.0
        WHEN proc_n <= 3 THEN 1.0
        ELSE 2.0
      END AS proc_n_bucket,
      CASE
        WHEN med_distinct <= 2 THEN 0.0
        WHEN med_distinct <= 5 THEN 1.0
        ELSE 2.0
      END AS med_distinct_bucket,
      (med_infusion_mean * scai_prop_ge3)::DOUBLE AS infusion_x_scai_severity
    FROM base_join;

    COPY enriched TO '{_q(out_p)}' (FORMAT PARQUET);
    """

    con = duckdb.connect(database=":memory:")
    con.execute(sql)
    con.close()
    legend_path = data_dir / LEGEND_JSON_NAME
    legend_path.write_text(json.dumps(build_demo_profile_legend(), indent=2), encoding="utf-8")
    return out_p


def audit_infusion_feature_distribution(data_dir: Path) -> dict:
    """Check ``infusion_early_over_all_ratio`` spike at 1.0 vs new early-minus-all features."""
    path = data_dir.resolve() / SIDE_CAR_NAME
    if not path.is_file():
        return {}
    df = pd.read_parquet(path)
    out: dict[str, object] = {}
    if "infusion_early_over_all_ratio" in df.columns:
        r = pd.to_numeric(df["infusion_early_over_all_ratio"], errors="coerce")
        out["infusion_early_over_all_ratio"] = {
            "n": int(r.notna().sum()),
            "pct_eq_1": float((r == 1.0).mean()),
            "pct_eq_1_02": float((r.between(0.98, 1.02)).mean()),
            "median": float(r.median()) if r.notna().any() else None,
        }
    if "inf_early_minus_all" in df.columns:
        d = pd.to_numeric(df["inf_early_minus_all"], errors="coerce")
        out["inf_early_minus_all"] = {
            "median": float(d.median()) if d.notna().any() else None,
            "pct_positive": float((d > 0).mean()),
        }
    if "infusion_early_above_mean" in df.columns:
        f = pd.to_numeric(df["infusion_early_above_mean"], errors="coerce").fillna(0)
        out["infusion_early_above_mean"] = {"pct_one": float((f >= 1).mean())}
    return out


def run_stratified_residual_audit(data_dir: Path) -> dict:
    """
    DuckDB: within each rounded SCAI stage, compare mean(feature) | died=1 vs died=0.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install duckdb: pip install duckdb") from exc

    data_dir = data_dir.resolve()
    p_side = _q(data_dir / SIDE_CAR_NAME)
    sql = f"""
    CREATE OR REPLACE VIEW e AS SELECT * FROM read_parquet('{p_side}');
    SELECT
      CAST(ROUND(current_scai) AS INT) AS scai_bin,
      COUNT(*)::INT AS n,
      AVG(CASE WHEN died = 1 THEN med_distinct END) AS mean_med_died,
      AVG(CASE WHEN died = 0 THEN med_distinct END) AS mean_med_surv,
      AVG(CASE WHEN died = 1 THEN med_infusion_mean END) AS mean_inf_died,
      AVG(CASE WHEN died = 0 THEN med_infusion_mean END) AS mean_inf_surv,
      AVG(CASE WHEN died = 1 THEN clinical_events_per_icu_hour END) AS mean_ce_rate_died,
      AVG(CASE WHEN died = 0 THEN clinical_events_per_icu_hour END) AS mean_ce_rate_surv,
      AVG(CASE WHEN died = 1 THEN clinical_events_last_4h END) AS mean_ce_4h_died,
      AVG(CASE WHEN died = 0 THEN clinical_events_last_4h END) AS mean_ce_4h_surv,
      AVG(CASE WHEN died = 1 THEN scai_slope_12h END) AS mean_slope_died,
      AVG(CASE WHEN died = 0 THEN scai_slope_12h END) AS mean_slope_surv,
      AVG(CASE WHEN died = 1 THEN scai_max END) AS mean_scai_max_died,
      AVG(CASE WHEN died = 0 THEN scai_max END) AS mean_scai_max_surv
    FROM e
    GROUP BY 1
    ORDER BY 1;
    """
    con = duckdb.connect(database=":memory:")
    df = con.execute(sql).fetchdf()
    con.close()
    return {"stratified_means_by_scai_bin": df.to_dict(orient="records")}


def write_feature_audit_json(
    data_dir: Path,
    *,
    mi_full_cohort: dict[str, float] | None = None,
    stratified_audit: dict | None = None,
    infusion_audit: dict | None = None,
) -> Path:
    """Write exploratory audit JSON next to parquets."""
    data_dir = data_dir.resolve()
    audit = {
        "sidecar_parquet": SIDE_CAR_NAME,
        "note_mi_full_cohort": (
            "mutual_information_full_cohort is exploratory (fit on all patients before split); "
            "training uses train-only metrics in xgb_mortality_model_meta.json."
        ),
        "mutual_information_full_cohort": mi_full_cohort or {},
        "stratified_residual_audit": stratified_audit or {},
        "infusion_feature_audit": infusion_audit or {},
    }
    out = data_dir / AUDIT_JSON_NAME
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


def ensure_duckdb_sidecar(data_dir: Path, *, force: bool = False) -> Path:
    """Materialize sidecar if missing or ``force``."""
    data_dir = data_dir.resolve()
    path = data_dir / SIDE_CAR_NAME
    if force or not path.is_file():
        materialize_duckdb_patient_features(data_dir)
    return path


def merge_duckdb_features(base: pd.DataFrame, data_dir: Path, *, force_duckdb: bool = False) -> pd.DataFrame:
    """Left-join DuckDB sidecar columns onto ``base`` (expects PERSON_ID)."""
    ensure_duckdb_sidecar(data_dir, force=force_duckdb)
    side = pd.read_parquet(data_dir / SIDE_CAR_NAME)
    # Avoid duplicate died / overlap
    drop_cols = [c for c in side.columns if c in ("died",) or c in base.columns and c != "PERSON_ID"]
    side = side.drop(columns=[c for c in drop_cols if c in side.columns], errors="ignore")
    out = base.merge(side, on="PERSON_ID", how="left", suffixes=("", "_duck"))
    return out
