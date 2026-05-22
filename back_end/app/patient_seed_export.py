"""Rebuild seed patient *clinical* fields from the same parquet rows models use."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.patient_id_align import SCAI_LETTERS, encounter_id_for_person

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__import__("os").environ.get("BAPTIST_DATA_DIR", str(REPO_ROOT / "data")))

VITAL_META: dict[str, tuple[str, str]] = {
    "HR": ("Heart Rate", "bpm"),
    "SBP": ("Systolic Blood Pressure", "mmHg"),
    "DBP": ("Diastolic Blood Pressure", "mmHg"),
    "MAP": ("Mean Arterial Pressure", "mmHg"),
    "RR": ("Respiratory Rate", "/min"),
    "SPO2": ("Peripheral Oxygen Saturation", "%"),
    "TEMP": ("Temperature", "degC"),
    "URINE_OUT_HR": ("Urine Output - Last Hour", "mL"),
    "VIS": ("Vasoactive-Inotrope Score", ""),
}

_data_cache: dict[str, pd.DataFrame] = {}


def _load(name: str) -> pd.DataFrame:
    if name not in _data_cache:
        path = DATA_DIR / "cleaned" / f"{name}.parquet"
        if not path.is_file():
            path = DATA_DIR / f"{name}.parquet"
        _data_cache[name] = pd.read_parquet(path)
    return _data_cache[name]


def _scai_letter(num: int) -> str:
    return SCAI_LETTERS[max(0, min(4, int(num)))]


def _normalcy(code: str, value: float) -> str:
    if code == "SPO2" and value < 92:
        return "LOW"
    if code == "MAP" and value < 65:
        return "LOW"
    if code == "SBP" and value < 90:
        return "LOW"
    if code == "HR" and (value < 50 or value > 120):
        return "ABNORMAL"
    return "NORMAL"


def sync_patient_clinical_from_parquet(patient: dict[str, Any]) -> dict[str, Any]:
    """Overwrite display/clinical fields from parquets for ``model_person_id`` / ``id``."""
    pid = int(patient.get("model_person_id") or patient["id"])
    eid = int(patient.get("model_encounter_id") or patient["encounter_id"])

    person = _load("person")
    enc = _load("encounter")
    dx = _load("diagnosis")
    meds = _load("medication_admin")
    clinical = _load("clinical_event")
    scai = _load("scai_stage_hourly")

    pr = person[person["PERSON_ID"] == pid].iloc[0]
    er = enc[enc["ENCOUNTER_ID"] == eid].iloc[0]

    admit_ts = pd.Timestamp(er["REG_DT_TM"])
    birth_ts = pd.Timestamp(pr["BIRTH_DT_TM"])
    age = int(max(18, min(95, (admit_ts - birth_ts).days / 365.25)))
    sex = str(pr["SEX_CD"]).upper()
    gender = "M" if sex.startswith("M") else "F"

    los_h = float(er.get("LOS_HOURS", 72))
    days = max(1, int(round(los_h / 24.0)))
    hour = int(patient.get("hour_from_admit") or min(int(los_h), 84))
    hour = max(0, min(hour, int(los_h)))

    scai_sub = scai[scai["ENCOUNTER_ID"] == eid].sort_values("HOUR_FROM_ADMIT")
    scai_now = scai_sub[scai_sub["HOUR_FROM_ADMIT"] <= hour]
    if len(scai_now):
        stage_row = scai_now.iloc[-1]
        stage_num = int(stage_row["SCAI_STAGE_NUM"])
        stage = str(stage_row["SCAI_STAGE_CD"])
    else:
        stage_num = 1
        stage = "B"

    trajectory = [
        {
            "hour_from_admit": int(r["HOUR_FROM_ADMIT"]),
            "scai_stage": str(r["SCAI_STAGE_CD"]),
            "scai_stage_num": int(r["SCAI_STAGE_NUM"]),
        }
        for _, r in scai_sub[scai_sub["HOUR_FROM_ADMIT"] <= hour].iterrows()
    ]

    dx_sub = dx[dx["ENCOUNTER_ID"] == eid].sort_values("DIAG_PRIORITY")
    diagnoses = [
        {
            "code": str(r["NOMENCLATURE_CD"]),
            "text": str(r["DIAGNOSIS_TXT"]),
            "priority": int(r["DIAG_PRIORITY"]),
            "classification": str(r["CLASSIFICATION_CD"]),
        }
        for _, r in dx_sub.iterrows()
    ]
    principal = diagnoses[0]["text"] if diagnoses else "Cardiogenic shock"

    med_sub = meds[meds["ENCOUNTER_ID"] == eid].head(8)
    medications = []
    for _, r in med_sub.iterrows():
        rate = r.get("INFUSION_RATE")
        unit = r.get("RATE_UNIT_CD") or ""
        dosage = f"{rate} {unit}".strip() if pd.notna(rate) else ""
        medications.append(
            {
                "name": str(r["MEDICATION_TXT"]),
                "code": str(r["MEDICATION_CD"]),
                "dosage": dosage,
                "route": str(r.get("ROUTE_CD", "IV")),
                "is_vasopressor": str(r["MEDICATION_CD"])
                in ("NOREPINEPHRINE", "EPINEPHRINE", "VASOPRESSIN", "PHENYLEPHRINE", "DOPAMINE"),
                "start": str(r["ADMIN_START_DT_TM"])[:19] if pd.notna(r["ADMIN_START_DT_TM"]) else "",
            }
        )

    clin = clinical[(clinical["ENCOUNTER_ID"] == eid) & (clinical["EVENT_CLASS_CD"] == "VITAL")]
    if "HOUR_FROM_ADMIT" not in clin.columns:
        clin = clin.copy()
        clin["HOUR_FROM_ADMIT"] = (
            pd.to_datetime(clin["EVENT_END_DT_TM"]) - pd.to_datetime(er["REG_DT_TM"])
        ).dt.total_seconds() // 3600
    clin = clin[clin["HOUR_FROM_ADMIT"] <= hour]
    vitals: list[dict[str, Any]] = []
    for code, (title, units) in VITAL_META.items():
        rows = clin[clin["EVENT_CD"] == code]
        if rows.empty:
            continue
        row = rows.sort_values("HOUR_FROM_ADMIT").iloc[-1]
        val = float(row["RESULT_VAL"])
        if not np.isfinite(val):
            continue
        vitals.append(
            {
                "code": code,
                "title": title,
                "value": round(val, 2),
                "units": units,
                "normalcy": _normalcy(code, val),
                "recorded_at": str(row["EVENT_END_DT_TM"])[:19],
            }
        )

    out = dict(patient)
    out["id"] = str(pid)
    out["encounter_id"] = str(eid)
    out["model_person_id"] = str(pid)
    out["model_encounter_id"] = str(eid)
    out["age"] = age
    out["gender"] = gender
    out["days_admitted"] = days
    out["hour_from_admit"] = hour
    out["diagnosis"] = principal
    out["issue"] = principal
    out["unit_cd"] = str(er.get("UNIT_CD", "CICU"))
    out["facility_cd"] = str(er.get("FACILITY_CD", "BHSF_MAIN"))
    out["scai_stage_current"] = stage
    out["scai_trajectory"] = trajectory
    out["vitals"] = vitals
    out["diagnoses"] = diagnoses
    out["medications"] = medications
    out["model_id_match_method"] = "parquet_export"
    return out
