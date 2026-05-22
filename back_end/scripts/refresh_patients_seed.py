#!/usr/bin/env python3
"""
Rewrite Frontend/assets/patients_seed.json predictions + SCAI display fields
from the same parquet rows the mortality/LOS models use at inference.

Run from repo root:
  PYTHONPATH=back_end BAPTIST_DATA_DIR=data python3 back_end/scripts/refresh_patients_seed.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "back_end") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "back_end"))

from app.patient_id_align import model_encounter_id, model_person_id  # noqa: E402
from app.scai_display import scai_stage_at_hour  # noqa: E402
from app.patient_predictions import (  # noqa: E402
    load_seed_patients,
    predict_for_patient_dict,
)

SEED_PATH = REPO_ROOT / "Frontend" / "assets" / "patients_seed.json"
SCAI_LETTERS = ("A", "B", "C", "D", "E")

# Keys owned by trained mortality/LOS bundles — always replace on refresh.
MODEL_PRED_KEYS = frozenset(
    {
        "mortality_risk",
        "hospital_mortality",
        "icu_mortality",
        "in_hospital_expiry",
        "death_alert_threshold",
        "death_alert",
        "hospital_los_days",
        "icu_los_days",
        "hospital_los_hours",
        "icu_los_hours",
        "hospital_los_total_days",
        "icu_los_total_days",
        "hospital_los_total_hours",
        "icu_los_total_hours",
        "los_prediction_hour",
        "mortality_model_version",
        "los_model_version",
        "mortality_source",
        "los_source",
        "predictions_refreshed_at",
    }
)


def scai_from_numeric(val: Any) -> tuple[str, int]:
    if val is None:
        return "B", 1
    try:
        n = int(round(float(val)))
    except (TypeError, ValueError):
        return "B", 1
    n = max(0, min(4, n))
    return SCAI_LETTERS[n], n


def deterioration_from_mortality(mort: float) -> tuple[float, str]:
    if mort >= 0.65:
        return 0.72, "Likely to worsen within 6h"
    if mort >= 0.45:
        return 0.48, "May worsen within 6h"
    if mort >= 0.25:
        return 0.28, "Possible worsening"
    return 0.08, "Unlikely to worsen"


def sync_scai_trajectory(
    trajectory: list[dict[str, Any]] | None,
    *,
    hour_from_admit: int,
    stage: str,
    stage_num: int,
) -> None:
    if not trajectory:
        return
    cutoff = max(0, hour_from_admit - 12)
    for pt in trajectory:
        h = int(pt.get("hour_from_admit", 0))
        if h >= cutoff:
            pt["scai_stage"] = stage
            pt["scai_stage_num"] = stage_num


def refresh_patient(patient: dict[str, Any]) -> dict[str, Any]:
    out = dict(patient)
    encounter_id = model_encounter_id(out)
    hour = int(out.get("hour_from_admit") or 0)
    if encounter_id:
        stage, stage_num = scai_stage_at_hour(encounter_id, hour)
        out["scai_stage_current"] = stage
    else:
        stage = str(out.get("scai_stage_current", "B")).upper()
        stage_num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(stage, 1)
    sync_scai_trajectory(out.get("scai_trajectory"), hour_from_admit=hour, stage=stage, stage_num=stage_num)

    pred = dict(out.get("predictions") or {})
    model_pred = predict_for_patient_dict(out)
    for key in MODEL_PRED_KEYS:
        if key in model_pred:
            pred[key] = model_pred[key]

    mort = float(pred.get("mortality_risk", model_pred.get("mortality_risk", 0.0)))
    det_prob, det_label = deterioration_from_mortality(mort)
    pred["scai_deterioration_6h_prob"] = round(det_prob, 4)
    pred["scai_deterioration_6h_label"] = det_label
    pred["current_scai_stage"] = stage
    pred["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Keep ICU/readmission loosely aligned with mortality for list sorting demos.
    pred["icu_transfer_risk"] = round(min(0.95, mort * 0.85 + 0.08), 4)
    pred["readmission_risk"] = round(min(0.95, mort * 0.55 + 0.05), 4)

    out["predictions"] = pred
    out.pop("condition", None)
    return out


def main() -> None:
    patients = load_seed_patients()
    refreshed = [refresh_patient(p) for p in patients]
    payload = {
        "seed_metadata": {
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "script": "back_end/scripts/refresh_patients_seed.py",
            "patient_count": len(refreshed),
            "note": "SCAI + mortality/LOS aligned to duckdb/los parquet feature rows.",
        },
        "patients": refreshed,
    }
    SEED_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(refreshed)} patients -> {SEED_PATH}")

    sample = refreshed[0]
    sp = sample.get("predictions", {})
    print(
        f"Sample {sample.get('name')} ({sample.get('id')}): "
        f"scai={sample.get('scai_stage_current')} "
        f"mortality={sp.get('mortality_risk'):.3f} "
        f"alert={sp.get('death_alert')}"
    )


if __name__ == "__main__":
    main()
