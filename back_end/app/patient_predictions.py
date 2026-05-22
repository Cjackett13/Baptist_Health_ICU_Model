"""Build live patient predictions from seed JSON + parquet feature lookup."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.feature_store import lookup_los_features, lookup_mortality_features
from app.los_display import los_remaining_from_total
from app.los_inference import predict_los_from_features
from app.mortality_inference import predict_mortality_from_features
from app.patient_id_align import model_encounter_id, model_person_id
from app.scai_display import scai_stage_at_hour

REPO_ROOT = Path(__file__).resolve().parents[2]


def _seed_path() -> Path:
    env = os.environ.get("PATIENTS_SEED_JSON")
    if env:
        return Path(env)
    return REPO_ROOT / "Frontend" / "assets" / "patients_seed.json"


_seed_cache: tuple[float, list[dict[str, Any]]] | None = None


def reload_seed_patients() -> list[dict[str, Any]]:
    """Force reload ``patients_seed.json`` (e.g. after export/align scripts)."""
    global _seed_cache
    _seed_cache = None
    return load_seed_patients()


def load_seed_patients() -> list[dict[str, Any]]:
    global _seed_cache
    path = _seed_path()
    if not path.is_file():
        raise FileNotFoundError(f"patients_seed.json not found at {path}")
    mtime = path.stat().st_mtime
    if _seed_cache is not None and _seed_cache[0] == mtime:
        return _seed_cache[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    patients = payload.get("patients", payload)
    if not isinstance(patients, list):
        raise ValueError("patients_seed.json must contain a 'patients' array")
    _seed_cache = (mtime, patients)
    return patients


def _merge_feature_dicts(*dicts: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            if v is not None and (not isinstance(v, float) or v == v):
                out[k] = v
    return out


def _seed_derived_features(patient: dict[str, Any]) -> dict[str, Any]:
    age = patient.get("age") or 65
    days = patient.get("days_admitted") or 1
    meds = patient.get("medications") or []
    dx = patient.get("diagnoses") or []
    gender = str(patient.get("gender", "U")).upper()
    unit = str(patient.get("unit_cd", "CICU")).upper()
    scai = str(patient.get("scai_stage_current", "B")).upper()
    scai_num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(scai, 1)
    return {
        "age_years": float(age),
        "age_mid": float(age),
        "sex_bin": 1 if gender in ("M", "MALE") else 0,
        "time_in_hospital": int(days),
        "days_admitted": int(days),
        "num_medications": len(meds),
        "number_diagnoses": max(1, len(dx)),
        "current_scai": float(scai_num),
        "current_scai_12h": float(scai_num),
        "scai_prop_ge3_12h": 1.0 if scai_num >= 2 else 0.0,
        "unit_cd": unit,
        "admit_type_cd": "EMERGENCY",
        "admit_src_cd": "ED",
        "dx_cat_arrhythmia_arrest": 0,
    }


def predict_for_patient_dict(
    patient: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return updated ``predictions`` map for one seed patient row."""
    person_id = model_person_id(patient)
    encounter_id = model_encounter_id(patient)
    mort_row = lookup_mortality_features(person_id) if person_id else None
    los_row = lookup_los_features(encounter_id) if encounter_id else None
    hour = int(patient.get("hour_from_admit") or 0)
    display_stage = str(patient.get("scai_stage_current", "B")).upper()
    display_num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(display_stage, 1)
    if encounter_id:
        display_stage, display_num = scai_stage_at_hour(encounter_id, hour)

    if mort_row:
        # Use trained feature rows only — seed display fields must not override parquet.
        features = _merge_feature_dicts(mort_row, los_row, overrides)
    else:
        features = _merge_feature_dicts(_seed_derived_features(patient), los_row, overrides)

    # Registry blend uses ICU-hour SCAI (same stage shown in the app), not end-of-stay max.
    features["_registry_scai"] = float(display_num)

    mort_scores, mort_ver, mort_src, alert_thr, death_alert = predict_mortality_from_features(
        features
    )
    los_total_h, icu_total_h, los_ver, los_src = predict_los_from_features(features)
    hosp_total_h, hosp_remain_h, icu_total_h, icu_remain_h = los_remaining_from_total(
        los_total_h, icu_total_h, hour
    )

    pred = dict(patient.get("predictions") or {})
    mortality_risk = mort_scores.hospital_mortality
    pred.update(
        {
            "mortality_risk": mortality_risk,
            "hospital_mortality": mort_scores.hospital_mortality,
            "icu_mortality": mort_scores.icu_mortality,
            "in_hospital_expiry": mort_scores.in_hospital_expiry,
            "death_alert_threshold": alert_thr,
            "death_alert": death_alert,
            # UI fields = remaining stay at hour_from_admit (not total encounter length).
            "hospital_los_days": round(hosp_remain_h / 24.0, 2),
            "icu_los_days": round(icu_remain_h / 24.0, 2),
            "hospital_los_hours": round(hosp_remain_h, 2),
            "icu_los_hours": round(icu_remain_h, 2),
            "hospital_los_total_days": round(hosp_total_h / 24.0, 2),
            "icu_los_total_days": round(icu_total_h / 24.0, 2),
            "hospital_los_total_hours": round(hosp_total_h, 2),
            "icu_los_total_hours": round(icu_total_h, 2),
            "los_prediction_hour": hour,
            "mortality_model_version": mort_ver,
            "los_model_version": los_ver,
            "mortality_source": mort_src,
            "los_source": los_src,
            "predictions_refreshed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return pred


def enrich_patient(patient: dict[str, Any], *, live: bool = True) -> dict[str, Any]:
    out = dict(patient)
    if live:
        out["predictions"] = predict_for_patient_dict(out)
    return out


def list_patients(*, live: bool = True) -> dict[str, Any]:
    rows = [enrich_patient(p, live=live) for p in load_seed_patients()]
    return {"patients": rows, "count": len(rows), "live": live}


def get_patient(encounter_or_person_id: str, *, live: bool = True) -> dict[str, Any] | None:
    key = str(encounter_or_person_id)
    for p in load_seed_patients():
        keys = {
            str(p.get("id", "")),
            str(p.get("encounter_id", "")),
            str(p.get("model_person_id", "")),
            str(p.get("model_encounter_id", "")),
            str(p.get("seed_person_id_legacy", "")),
        }
        if key in keys:
            return enrich_patient(p, live=live)
    return None
