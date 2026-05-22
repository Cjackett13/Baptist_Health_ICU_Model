"""
LOS-only table cleaning for Dataset B (`data/`). Does not modify mortality pipeline.

Steps: trim strings, sentinel → missing (object/string/category columns), canonical
coded vocab, datetime parse (unparseable → NaT, row kept), numeric coercion,
ICD / EVENT_CD validation, table-specific rules; median impute only
``medication_admin`` / ``scai_stage_hourly`` (not encounter, clinical_event, diagnosis).

Optional: ``persist_cleaned_bundle`` writes ``{data_dir}/cleaned/*.parquet`` (raw bundle unchanged).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MISSING_STRINGS: frozenset[str] = frozenset(
    {
        "",
        " ",
        "NULL",
        "NONE",
        "N/A",
        "NA",
        "NAN",
        "NULL",
        "-99",
        "999",
        ".",
        "--",
    }
)

UNKNOWN_STRINGS: frozenset[str] = frozenset({"UNKNOWN", "UNK", "UNKNOWN/UNSPECIFIED"})

SEX_MAP: dict[str, str] = {
    "M": "M",
    "MALE": "M",
    "F": "F",
    "FEMALE": "F",
    "MAN": "M",
    "WOMAN": "F",
}

RACE_MAP: dict[str, str] = {
    "WHITE": "WHITE",
    "BLACK": "BLACK",
    "AFRICAN AMERICAN": "BLACK",
    "ASIAN": "ASIAN",
    "OTHER": "OTHER",
    "UNKNOWN": "UNKNOWN",
}

ETHNICITY_MAP: dict[str, str] = {
    "HISPANIC": "HISPANIC",
    "HISPANIC OR LATINO": "HISPANIC",
    "NON_HISPANIC": "NON_HISPANIC",
    "NON-HISPANIC": "NON_HISPANIC",
    "NOT HISPANIC": "NON_HISPANIC",
    "UNKNOWN": "UNKNOWN",
}

ADMIT_TYPE_MAP: dict[str, str] = {
    "EMERGENCY": "EMERGENCY",
    "EMER": "EMERGENCY",
    "ED": "EMERGENCY",
    "URGENT": "URGENT",
    "ELECTIVE": "ELECTIVE",
    "ELEC": "ELECTIVE",
    "TRANSFER": "TRANSFER",
    "XFER": "TRANSFER",
}

ADMIT_SRC_MAP: dict[str, str] = {
    "ED": "ED",
    "EMERGENCY DEPARTMENT": "ED",
    "OUTSIDE_HOSPITAL": "OUTSIDE_HOSPITAL",
    "OUTSIDE HOSPITAL": "OUTSIDE_HOSPITAL",
    "OSH": "OUTSIDE_HOSPITAL",
    "DIRECT": "DIRECT",
    "OR": "OR",
    "OPERATING ROOM": "OR",
}

UNIT_MAP: dict[str, str] = {
    "CICU": "CICU",
    "CVICU": "CVICU",
    "MICU": "MICU",
    "CCU": "CICU",
    "CARDIAC ICU": "CICU",
}

ENCNTR_TYPE_MAP: dict[str, str] = {
    "INPATIENT": "INPATIENT",
    "IP": "INPATIENT",
    "INP": "INPATIENT",
}

DISPOSITION_MAP: dict[str, str] = {
    "HOME": "HOME",
    "SNF": "SNF",
    "REHAB": "REHAB",
    "LTACH": "LTACH",
    "EXPIRED": "EXPIRED",
    "HOSPICE": "HOSPICE",
    "AMA": "AMA",
}

ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9A-Z]?(\.[0-9A-Z]{1,4})?$")

DATETIME_COLUMNS: dict[str, list[str]] = {
    "person": ["BIRTH_DT_TM", "DECEASED_DT_TM"],
    "encounter": ["REG_DT_TM", "DISCH_DT_TM"],
    "diagnosis": ["DIAGNOSIS_DT_TM"],
    "clinical_event": ["EVENT_END_DT_TM"],
    "medication_admin": ["ADMIN_START_DT_TM", "ADMIN_END_DT_TM"],
    "procedure_event": ["PROC_START_DT_TM", "PROC_END_DT_TM"],
    "scai_stage_hourly": ["EVENT_DT_TM"],
}

NUMERIC_COLUMNS: dict[str, list[str]] = {
    "encounter": ["LOS_HOURS"],
    "clinical_event": ["RESULT_VAL"],
    "medication_admin": ["INFUSION_RATE"],
    "scai_stage_hourly": ["HOUR_FROM_ADMIT", "SCAI_STAGE_NUM"],
    "diagnosis": ["DIAG_PRIORITY"],
}

# Median fill only these tables; encounter / clinical_event / diagnosis keep NaN after rules
NUMERIC_IMPUTE_COLUMNS: dict[str, list[str]] = {
    "medication_admin": ["INFUSION_RATE"],
    "scai_stage_hourly": ["HOUR_FROM_ADMIT", "SCAI_STAGE_NUM"],
}

AGE_HARD_MIN = 18
AGE_HARD_MAX = 120
LOS_HOURS_HARD_MAX = 14 * 24

SCAI_STAGE_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Hard limits for RESULT_VAL by EVENT_CD (align with los_frontend_value_ranges.json)
CLINICAL_RESULT_RANGES: dict[str, tuple[float, float]] = {
    "HR": (20.0, 250.0),
    "SBP": (50.0, 250.0),
    "DBP": (30.0, 150.0),
    "MAP": (30.0, 150.0),
    "RR": (4.0, 60.0),
    "SPO2": (50.0, 100.0),
    "TEMP": (32.0, 42.0),
    "URINE_OUT_HR": (0.0, 2000.0),
    "LACTATE": (0.1, 25.0),
    "CREATININE": (0.2, 20.0),
    "BUN": (2.0, 150.0),
    "PH": (6.8, 7.8),
    "PCO2": (15.0, 90.0),
    "HCO3": (5.0, 45.0),
    "VIS": (0.0, 200.0),
    "CO": (1.0, 15.0),
    "CI": (0.5, 10.0),
    "SCAI_STAGE": (0.0, 4.0),
}

INFUSION_RATE_HARD = (0.0, 50.0)

CODED_COLUMN_MAP: dict[str, dict[str, str]] = {
    "SEX_CD": SEX_MAP,
    "RACE_CD": RACE_MAP,
    "ETHNICITY_CD": ETHNICITY_MAP,
    "ADMIT_TYPE_CD": ADMIT_TYPE_MAP,
    "ADMIT_SRC_CD": ADMIT_SRC_MAP,
    "UNIT_CD": UNIT_MAP,
    "ENCNTR_TYPE_CD": ENCNTR_TYPE_MAP,
    "DISCH_DISPOSITION_CD": DISPOSITION_MAP,
}

CLEANED_BUNDLE_SUBDIR = "cleaned"
CLEANED_MANIFEST_NAME = "los_cleaned_bundle_manifest.json"


def _iter_string_columns(df: pd.DataFrame) -> list[str]:
    """All object/string/category columns plus explicit ``_CD`` / ``_TXT`` fields."""
    cols: set[str] = set()
    for c in df.columns:
        if c in CODED_COLUMN_MAP or c.endswith("_CD") or c.endswith("_TXT"):
            cols.add(c)
            continue
        dt = df[c].dtype
        if (
            pd.api.types.is_object_dtype(dt)
            or pd.api.types.is_string_dtype(dt)
            or pd.api.types.is_categorical_dtype(dt)
        ):
            cols.add(c)
    return sorted(cols)


def _sum_nested_counts(audits: list[dict[str, Any]], key: str) -> int:
    total = 0
    for aud in audits:
        block = aud.get(key)
        if isinstance(block, dict):
            total += sum(int(v) for v in block.values())
        elif isinstance(block, (int, float)):
            total += int(block)
    return total


def aggregate_cleaning_totals(
    per_table: dict[str, Any],
    table_rules: dict[str, Any],
    median_impute: dict[str, Any],
) -> dict[str, Any]:
    """Roll per-table + deep-rule audits into one totals block for JSON logs."""
    table_audits = list(per_table.values()) if isinstance(per_table, dict) else []
    return {
        "sentinels_to_missing": _sum_nested_counts(table_audits, "sentinels_to_missing"),
        "invalid_coded_to_missing": _sum_nested_counts(table_audits, "invalid_coded_to_missing"),
        "datetime_unparseable": _sum_nested_counts(table_audits, "datetime_unparseable"),
        "numeric_coercion_failed": _sum_nested_counts(table_audits, "numeric_coercion_failed"),
        "invalid_icd_to_missing": _sum_nested_counts(table_audits, "invalid_icd_to_missing"),
        "invalid_event_cd_to_missing": _sum_nested_counts(table_audits, "invalid_event_cd_to_missing"),
        "numeric_median_imputed": _sum_nested_counts([median_impute], "numeric_median_imputed"),
        "duplicate_encounter_id_dropped": int(table_rules.get("duplicate_encounter_id_dropped", {}).get("encounter", 0)),
        "los_hours_reconciled_to_datetime": int(
            table_rules.get("los_hours_reconciled_to_datetime", {}).get("encounter", 0)
        ),
        "los_hours_out_of_range_nullified": int(
            table_rules.get("los_hours_out_of_range_nullified", {}).get("encounter", 0)
        ),
        "missing_reg_dt_tm": int(table_rules.get("missing_reg_dt_tm", {}).get("encounter", 0)),
        "missing_disch_dt_tm": int(table_rules.get("missing_disch_dt_tm", {}).get("encounter", 0)),
        "clinical_result_out_of_range_nullified": int(
            table_rules.get("clinical_result_out_of_range_nullified", {}).get("clinical_event", 0)
        ),
        "open_interval_closed_at_discharge": _sum_nested_counts([table_rules], "open_interval_closed_at_discharge"),
        "end_before_start_fixed": _sum_nested_counts([table_rules], "end_before_start_fixed"),
        "age_implausible_birth_nullified": int(table_rules.get("age_implausible_birth_nullified", {}).get("person", 0)),
        "invalid_scai_cd_nullified": int(table_rules.get("invalid_scai_cd_nullified", {}).get("scai_stage_hourly", 0)),
    }


def _norm_token(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    u = s.upper()
    if u in MISSING_STRINGS or u in {"NULL", "N/A", "NA", "NONE"}:
        return None
    return u


def _map_coded(val: Any, vocab: dict[str, str], *, allow_unknown: bool = False) -> str | None:
    u = _norm_token(val)
    if u is None:
        return None
    if u in UNKNOWN_STRINGS:
        return "UNKNOWN" if allow_unknown else None
    if u in vocab:
        return vocab[u]
    # fuzzy: remove spaces/dashes
    compact = u.replace(" ", "_").replace("-", "_")
    if compact in vocab:
        return vocab[compact]
    return None


def _clean_string_series(s: pd.Series, *, column: str, audit: dict) -> pd.Series:
    if pd.api.types.is_categorical_dtype(s):
        s = s.astype("string")
    out = s.astype("string").str.strip()
    upper = out.str.upper()
    n_sent = int(upper.isin(MISSING_STRINGS | {"NULL", "N/A", "NA", "NONE", "NAN"}).sum())
    if n_sent:
        audit.setdefault("sentinels_to_missing", {})[column] = n_sent
    out = out.mask(upper.isin(MISSING_STRINGS | {"NULL", "N/A", "NA", "NONE", "NAN"}), pd.NA)

    if column in CODED_COLUMN_MAP:
        allow_unk = column in ("RACE_CD", "ETHNICITY_CD")
        mapped = out.map(lambda v: _map_coded(v, CODED_COLUMN_MAP[column], allow_unknown=allow_unk))
        n_bad = int((out.notna() & mapped.isna()).sum())
        if n_bad:
            audit.setdefault("invalid_coded_to_missing", {})[column] = n_bad
        out = mapped
    elif column.endswith("_CD"):
        out = out.str.upper()
    return out


def _parse_datetimes(df: pd.DataFrame, table: str, audit: dict) -> pd.DataFrame:
    out = df.copy()
    for col in DATETIME_COLUMNS.get(table, []):
        if col not in out.columns:
            continue
        raw = out[col]
        parsed = pd.to_datetime(raw, errors="coerce", utc=False)
        n_fail = int(raw.notna().sum() - parsed.notna().sum())
        if n_fail:
            audit.setdefault("datetime_unparseable", {})[f"{table}.{col}"] = n_fail
        out[col] = parsed
    return out


def _coerce_numeric(df: pd.DataFrame, table: str, audit: dict) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS.get(table, []):
        if col not in out.columns:
            continue
        raw = out[col]
        coerced = pd.to_numeric(raw, errors="coerce")
        present = raw.notna() & raw.astype("string").str.strip().ne("")
        failed = present & coerced.isna()
        n_fail = int(failed.sum())
        if n_fail:
            audit.setdefault("numeric_coercion_failed", {})[f"{table}.{col}"] = n_fail
        out[col] = coerced
    return out


def _validate_icd(series: pd.Series, audit: dict, *, key: str) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    valid_mask = out.isna() | out.map(lambda v: bool(ICD10_PATTERN.match(str(v))) if v else True)
    n_bad = int((~valid_mask).sum())
    if n_bad:
        audit.setdefault("invalid_icd_to_missing", {})[key] = n_bad
        sample = out[~valid_mask].drop_duplicates().head(15).tolist()
        audit.setdefault("invalid_icd_samples", {})[key] = sample
    return out.where(valid_mask, pd.NA)


def _validate_event_cd(series: pd.Series, valid_codes: set[str], audit: dict) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    valid_mask = out.isna() | out.isin(valid_codes)
    n_bad = int((~valid_mask).sum())
    if n_bad:
        audit.setdefault("invalid_event_cd_to_missing", {})["clinical_event.EVENT_CD"] = n_bad
        audit.setdefault(
            "invalid_event_cd_samples",
            {},
        )["clinical_event.EVENT_CD"] = out[~valid_mask].drop_duplicates().head(15).tolist()
    return out.where(valid_mask, pd.NA)


def _audit_set(audit: dict, key: str, subkey: str, n: int) -> None:
    if n:
        audit.setdefault(key, {})[subkey] = int(audit.get(key, {}).get(subkey, 0)) + int(n)


def impute_numeric_medians(df: pd.DataFrame, table: str, audit: dict) -> pd.DataFrame:
    """Fill numeric NaNs with table-wide median for ``NUMERIC_IMPUTE_COLUMNS`` only."""
    if table not in NUMERIC_IMPUTE_COLUMNS:
        return df
    out = df.copy()
    for col in NUMERIC_IMPUTE_COLUMNS.get(table, []):
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        n_miss = int(s.isna().sum())
        if n_miss == 0:
            continue
        global_med = float(s.median()) if s.notna().any() else 0.0
        if not np.isfinite(global_med):
            global_med = 0.0
        filled = s.fillna(global_med)
        out[col] = filled
        _audit_set(audit, "numeric_median_imputed", f"{table}.{col}", n_miss - int(filled.isna().sum()))
    return out


def clean_person(df: pd.DataFrame, audit: dict) -> pd.DataFrame:
    out = df.copy()
    if "BIRTH_DT_TM" not in out.columns:
        return out
    birth = pd.to_datetime(out["BIRTH_DT_TM"], errors="coerce")
    ref = pd.Timestamp("2025-06-01")
    age = (ref - birth).dt.days / 365.25
    bad = birth.notna() & ((age < AGE_HARD_MIN) | (age > AGE_HARD_MAX))
    _audit_set(audit, "age_implausible_birth_nullified", "person", int(bad.sum()))
    out.loc[bad, "BIRTH_DT_TM"] = pd.NaT
    return out


def clean_encounter(df: pd.DataFrame, person: pd.DataFrame, audit: dict) -> pd.DataFrame:
    out = df.copy()
    if "ENCOUNTER_ID" in out.columns:
        n_dup = int(out.duplicated(subset=["ENCOUNTER_ID"], keep=False).sum())
        out = out.drop_duplicates(subset=["ENCOUNTER_ID"], keep="first")
        _audit_set(audit, "duplicate_encounter_id_dropped", "encounter", n_dup)

    reg = pd.to_datetime(out.get("REG_DT_TM"), errors="coerce")
    disch = pd.to_datetime(out.get("DISCH_DT_TM"), errors="coerce")
    bad_order = reg.notna() & disch.notna() & (disch < reg)
    _audit_set(audit, "discharge_before_admit_nullified", "encounter", int(bad_order.sum()))
    out.loc[bad_order, ["DISCH_DT_TM", "LOS_HOURS"]] = pd.NA

    if "LOS_HOURS" in out.columns and reg.notna().any() and disch.notna().any():
        delta_h = (disch - reg).dt.total_seconds() / 3600.0
        stored = pd.to_numeric(out["LOS_HOURS"], errors="coerce")
        mismatch = stored.notna() & delta_h.notna() & (stored - delta_h).abs().gt(1.0)
        out.loc[mismatch, "LOS_HOURS"] = delta_h.loc[mismatch]
        _audit_set(audit, "los_hours_reconciled_to_datetime", "encounter", int(mismatch.sum()))

    if "LOS_HOURS" in out.columns:
        los = pd.to_numeric(out["LOS_HOURS"], errors="coerce")
        bad_los = los.notna() & ((los <= 0) | (los > LOS_HOURS_HARD_MAX))
        _audit_set(audit, "los_hours_out_of_range_nullified", "encounter", int(bad_los.sum()))
        out.loc[bad_los, "LOS_HOURS"] = np.nan

    _audit_set(audit, "missing_reg_dt_tm", "encounter", int(reg.isna().sum()))
    _audit_set(audit, "missing_disch_dt_tm", "encounter", int(disch.isna().sum()))

    if "DISCH_DISPOSITION_CD" in out.columns and "PERSON_ID" in out.columns:
        disp = out["DISCH_DISPOSITION_CD"].astype("string").str.upper()
        died = (
            person.set_index("PERSON_ID")["DECEASED_DT_TM"]
            .reindex(out["PERSON_ID"].to_numpy())
            .notna()
            .reset_index(drop=True)
        )
        died.index = out.index
        expired = disp == "EXPIRED"
        mismatch_exp = expired & ~died
        mismatch_alive = (~expired) & died & disp.notna()
        _audit_set(audit, "expired_without_death_ts", "encounter", int(mismatch_exp.sum()))
        _audit_set(audit, "death_ts_without_expired_disp", "encounter", int(mismatch_alive.sum()))
        out.loc[mismatch_exp, "DISCH_DISPOSITION_CD"] = pd.NA

    return out


def clean_diagnosis(df: pd.DataFrame, audit: dict) -> pd.DataFrame:
    out = df.copy()
    if "DIAGNOSIS_ID" in out.columns:
        n_dup = int(out.duplicated(subset=["DIAGNOSIS_ID"], keep=False).sum())
        out = out.drop_duplicates(subset=["DIAGNOSIS_ID"], keep="first")
        _audit_set(audit, "duplicate_diagnosis_id_dropped", "diagnosis", n_dup)

    if "DIAG_PRIORITY" in out.columns:
        pr = pd.to_numeric(out["DIAG_PRIORITY"], errors="coerce")
        bad = pr.notna() & (pr < 1)
        out.loc[bad, "DIAG_PRIORITY"] = pd.NA
        _audit_set(audit, "invalid_diag_priority_nullified", "diagnosis", int(bad.sum()))

    keys = [c for c in ("ENCOUNTER_ID", "NOMENCLATURE_CD", "DIAG_PRIORITY") if c in out.columns]
    if len(keys) >= 2:
        n_dup = int(out.duplicated(subset=keys, keep=False).sum())
        out = out.drop_duplicates(subset=keys, keep="first")
        _audit_set(audit, "duplicate_dx_keys_dropped", "diagnosis", n_dup)

    if "ENCOUNTER_ID" in out.columns and "DIAG_PRIORITY" in out.columns:
        princ = out[out["DIAG_PRIORITY"] == 1]
        multi = princ.groupby("ENCOUNTER_ID").filter(lambda g: len(g) > 1)
        if len(multi):
            _audit_set(audit, "multiple_principal_dx_encounters", "diagnosis", multi["ENCOUNTER_ID"].nunique())
            keep = princ.drop_duplicates(subset=["ENCOUNTER_ID"], keep="first")
            drop_idx = princ.index.difference(keep.index)
            out.loc[drop_idx, "DIAG_PRIORITY"] = 2

    return out


def clean_clinical_event(df: pd.DataFrame, audit: dict) -> pd.DataFrame:
    out = df.copy()
    if "EVENT_ID" in out.columns:
        n_dup = int(out.duplicated(subset=["EVENT_ID"], keep=False).sum())
        out = out.drop_duplicates(subset=["EVENT_ID"], keep="first")
        _audit_set(audit, "duplicate_event_id_dropped", "clinical_event", n_dup)

    if "EVENT_CD" in out.columns and "RESULT_VAL" in out.columns:
        n_oob = 0
        vals = pd.to_numeric(out["RESULT_VAL"], errors="coerce")
        for code, (lo, hi) in CLINICAL_RESULT_RANGES.items():
            sel = out["EVENT_CD"].astype("string").str.upper() == code
            oob = sel & vals.notna() & ((vals < lo) | (vals > hi))
            n_oob += int(oob.sum())
            out.loc[oob, "RESULT_VAL"] = np.nan
        _audit_set(audit, "clinical_result_out_of_range_nullified", "clinical_event", n_oob)

    return out


def _fix_datetime_intervals(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    audit: dict,
    *,
    table: str,
    encounter_end: pd.Series | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if start_col not in out.columns:
        return out
    start = pd.to_datetime(out[start_col], errors="coerce")
    end = pd.to_datetime(out[end_col], errors="coerce") if end_col in out.columns else pd.Series(pd.NaT, index=out.index)

    if encounter_end is not None and end_col in out.columns:
        enc_end = out["ENCOUNTER_ID"].map(encounter_end)
        open_inf = end.isna() & start.notna() & enc_end.notna()
        out.loc[open_inf, end_col] = enc_end.loc[open_inf]
        _audit_set(audit, "open_interval_closed_at_discharge", table, int(open_inf.sum()))
        end = pd.to_datetime(out[end_col], errors="coerce")

    bad = start.notna() & end.notna() & (end < start)
    out.loc[bad, end_col] = start.loc[bad]
    _audit_set(audit, "end_before_start_fixed", table, int(bad.sum()))
    return out


def clean_medication_admin(df: pd.DataFrame, enc: pd.DataFrame, audit: dict) -> pd.DataFrame:
    enc_end = enc.set_index("ENCOUNTER_ID")["DISCH_DT_TM"] if "DISCH_DT_TM" in enc.columns else None
    out = _fix_datetime_intervals(
        df, "ADMIN_START_DT_TM", "ADMIN_END_DT_TM", audit, table="medication_admin", encounter_end=enc_end
    )
    if "INFUSION_RATE" in out.columns:
        ir = pd.to_numeric(out["INFUSION_RATE"], errors="coerce")
        lo, hi = INFUSION_RATE_HARD
        oob = ir.notna() & ((ir < lo) | (ir > hi))
        _audit_set(audit, "infusion_rate_out_of_range", "medication_admin", int(oob.sum()))
        out.loc[oob, "INFUSION_RATE"] = np.nan
    return out


def clean_procedure_event(df: pd.DataFrame, enc: pd.DataFrame, audit: dict) -> pd.DataFrame:
    enc_end = enc.set_index("ENCOUNTER_ID")["DISCH_DT_TM"] if "DISCH_DT_TM" in enc.columns else None
    return _fix_datetime_intervals(
        df, "PROC_START_DT_TM", "PROC_END_DT_TM", audit, table="procedure_event", encounter_end=enc_end
    )


def clean_scai_stage_hourly(df: pd.DataFrame, enc: pd.DataFrame, audit: dict) -> pd.DataFrame:
    out = df.copy()
    if "SCAI_STAGE_CD" in out.columns:
        cd = out["SCAI_STAGE_CD"].astype("string").str.strip().str.upper()
        mapped = cd.map(SCAI_STAGE_MAP)
        bad = cd.notna() & mapped.isna()
        _audit_set(audit, "invalid_scai_cd_nullified", "scai_stage_hourly", int(bad.sum()))
        out["SCAI_STAGE_CD"] = cd.where(mapped.notna(), pd.NA)
        out["SCAI_STAGE_NUM"] = mapped.astype("Int64")

    if "HOUR_FROM_ADMIT" in out.columns and "ENCOUNTER_ID" in out.columns:
        los_h = enc.set_index("ENCOUNTER_ID")["LOS_HOURS"]
        hr = pd.to_numeric(out["HOUR_FROM_ADMIT"], errors="coerce")
        max_h = out["ENCOUNTER_ID"].map(los_h)
        neg = hr.notna() & (hr < 0)
        over = hr.notna() & max_h.notna() & (hr > max_h)
        out.loc[neg, "HOUR_FROM_ADMIT"] = 0
        out.loc[over, "HOUR_FROM_ADMIT"] = max_h.loc[over]
        _audit_set(audit, "scai_hour_negative_clipped", "scai_stage_hourly", int(neg.sum()))
        _audit_set(audit, "scai_hour_above_los_clipped", "scai_stage_hourly", int(over.sum()))

    return out


def clean_table(
    df: pd.DataFrame,
    table: str,
    *,
    valid_event_codes: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean one table; return frame + per-table audit fragment."""
    audit: dict[str, Any] = {"table": table, "n_rows_in": int(len(df))}
    out = df.copy()

    for col in _iter_string_columns(out):
        out[col] = _clean_string_series(out[col], column=col, audit=audit)

    if table == "diagnosis" and "NOMENCLATURE_CD" in out.columns:
        out["NOMENCLATURE_CD"] = _validate_icd(out["NOMENCLATURE_CD"], audit, key="diagnosis.NOMENCLATURE_CD")

    if table == "clinical_event" and valid_event_codes and "EVENT_CD" in out.columns:
        out["EVENT_CD"] = _validate_event_cd(out["EVENT_CD"], valid_event_codes, audit)

    out = _parse_datetimes(out, table, audit)
    out = _coerce_numeric(out, table, audit)
    audit["n_rows_out"] = int(len(out))
    return out, audit


def clean_bundle(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """
    Clean all bundle tables. Expects keys: person, encounter, diagnosis,
    clinical_event, medication_admin, procedure_event, scai_stage_hourly, code_value.
    """
    valid_events: set[str] = set()
    if "code_value" in tables:
        cv = tables["code_value"]
        if "EVENT_CD" in cv.columns:
            valid_events = set(cv["EVENT_CD"].astype("string").str.strip().str.upper().dropna())

    cleaned: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    order = [
        "code_value",
        "person",
        "encounter",
        "diagnosis",
        "clinical_event",
        "medication_admin",
        "procedure_event",
        "scai_stage_hourly",
    ]
    for name in order:
        if name not in tables:
            continue
        if name == "code_value":
            df, aud = clean_table(tables[name], name)
        else:
            df, aud = clean_table(
                tables[name],
                name,
                valid_event_codes=valid_events if name == "clinical_event" else None,
            )
        cleaned[name] = df
        audits[name] = aud

    for name, df in tables.items():
        if name not in cleaned:
            cleaned[name], audits[name] = clean_table(df, name)

    # Table-specific rules (Dataset B)
    deep_audit: dict[str, Any] = {}
    if "person" in cleaned:
        cleaned["person"] = clean_person(cleaned["person"], deep_audit)
    if "encounter" in cleaned:
        person = cleaned.get("person", tables.get("person", pd.DataFrame()))
        cleaned["encounter"] = clean_encounter(cleaned["encounter"], person, deep_audit)
    if "diagnosis" in cleaned:
        cleaned["diagnosis"] = clean_diagnosis(cleaned["diagnosis"], deep_audit)
    if "clinical_event" in cleaned:
        cleaned["clinical_event"] = clean_clinical_event(cleaned["clinical_event"], deep_audit)
    enc = cleaned.get("encounter", pd.DataFrame())
    if "medication_admin" in cleaned and len(enc):
        cleaned["medication_admin"] = clean_medication_admin(cleaned["medication_admin"], enc, deep_audit)
    if "procedure_event" in cleaned and len(enc):
        cleaned["procedure_event"] = clean_procedure_event(cleaned["procedure_event"], enc, deep_audit)
    if "scai_stage_hourly" in cleaned and len(enc):
        cleaned["scai_stage_hourly"] = clean_scai_stage_hourly(cleaned["scai_stage_hourly"], enc, deep_audit)

    impute_audit: dict[str, Any] = {}
    for name, df in cleaned.items():
        if name == "code_value":
            continue
        cleaned[name] = impute_numeric_medians(df, name, impute_audit)

    totals = aggregate_cleaning_totals(audits, deep_audit, impute_audit)
    summary = {
        "tables_cleaned": list(cleaned.keys()),
        "per_table": audits,
        "table_rules_audit": deep_audit,
        "numeric_median_imputation": impute_audit,
        "imputation_policy": "median impute: medication_admin, scai_stage_hourly only; encounter/clinical_event/diagnosis keep NaN",
        "datetime_policy": "unparseable datetimes coerced to NaT; rows retained (no row drops on parse failure)",
        "raw_parquet_policy": "source data/*.parquet not overwritten; use persist_cleaned_bundle for cleaned copies",
        "frontend_value_ranges_file": "los_frontend_value_ranges.json",
        "totals": totals,
        "rules_applied": [
            "sentinels_to_missing",
            "invalid_coded_to_missing",
            "datetime_unparseable",
            "numeric_coercion_failed",
            "invalid_icd_to_missing",
            "invalid_event_cd_to_missing",
            "duplicate_encounter_id_dropped",
            "los_hours_reconciled_to_datetime",
            "los_hours_out_of_range_nullified",
            "clinical_result_out_of_range_nullified",
            "numeric_median_imputed",
            "open_interval_closed_at_discharge",
            "end_before_start_fixed",
            "age_implausible_birth_nullified",
            "invalid_scai_cd_nullified",
        ],
    }
    return cleaned, summary


def persist_cleaned_bundle(
    data_dir: Path,
    tables: dict[str, pd.DataFrame],
    *,
    subdir: str = CLEANED_BUNDLE_SUBDIR,
) -> Path:
    """
    Write cleaned tables to ``{data_dir}/{subdir}/`` without touching raw parquets.
    """
    data_dir = Path(data_dir).resolve()
    out_dir = data_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        written[name] = {"path": str(path.relative_to(data_dir)), "n_rows": int(len(df))}
    manifest = {
        "source_raw_dir": str(data_dir),
        "cleaned_subdir": subdir,
        "tables": written,
        "note": "Cleaned in-memory bundle; raw data/*.parquet unchanged.",
    }
    manifest_path = data_dir / CLEANED_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def copy_frontend_ranges(data_dir: Path, repo_root: Path | None = None) -> Path:
    """Copy ``los_frontend_value_ranges.json`` into ``data_dir`` for Frontend consumers."""
    root = repo_root or Path(__file__).resolve().parent.parent
    src = Path(__file__).resolve().parent / "los_frontend_value_ranges.json"
    dst = Path(data_dir) / "los_frontend_value_ranges.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst
