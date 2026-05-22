#!/usr/bin/env python3
"""
Align Frontend/assets/patients_seed.json to mortality + LOS parquet cohort IDs.

Keeps display names from the seed file; rewrites ``id`` / ``encounter_id`` (and
``model_person_id`` / ``model_encounter_id``) so live inference hits the correct rows.

Run from repo root:
  PYTHONPATH=back_end BAPTIST_DATA_DIR=data python3 back_end/scripts/align_patients_seed_to_models.py
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

from app.feature_store import lookup_los_features, lookup_mortality_features  # noqa: E402
from app.patient_id_align import align_seed_patients  # noqa: E402
from app.patient_predictions import load_seed_patients  # noqa: E402

_scripts = REPO_ROOT / "back_end" / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from refresh_patients_seed import refresh_patient  # noqa: E402

SEED_PATH = REPO_ROOT / "Frontend" / "assets" / "patients_seed.json"


def _verify_aligned(patients: list[dict[str, Any]]) -> dict[str, int]:
    missing_mort = 0
    missing_los = 0
    for p in patients:
        pid = p.get("model_person_id") or p.get("id")
        eid = p.get("model_encounter_id") or p.get("encounter_id")
        if pid and lookup_mortality_features(pid) is None:
            missing_mort += 1
        if eid and lookup_los_features(eid) is None:
            missing_los += 1
    return {"missing_mortality": missing_mort, "missing_los": missing_los}


def main() -> None:
    patients = load_seed_patients()
    aligned, report = align_seed_patients(patients)
    refreshed = [refresh_patient(p) for p in aligned]
    verify = _verify_aligned(refreshed)

    payload = {
        "seed_metadata": {
            "aligned_at": datetime.now(timezone.utc).isoformat(),
            "script": "back_end/scripts/align_patients_seed_to_models.py",
            "patient_count": len(refreshed),
            "alignment": report,
            "parquet_lookup": verify,
            "note": (
                "Display names are cosmetic. Predictions use model_person_id + "
                "model_encounter_id aligned to data/cleaned duckdb + los_modeling parquets."
            ),
        },
        "patients": refreshed,
    }
    SEED_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Aligned {len(refreshed)} patients -> {SEED_PATH}")
    print("Alignment report:", json.dumps(report, indent=2))
    print("Parquet lookup:", verify)
    if verify["missing_mortality"] or verify["missing_los"]:
        raise SystemExit("Some patients still missing parquet rows after alignment")

    sample = refreshed[0]
    sp = sample.get("predictions", {})
    print(
        f"Sample {sample.get('name')} model_id={sample.get('model_person_id')} "
        f"mortality={sp.get('mortality_risk'):.3f} los_days={sp.get('hospital_los_days')}"
    )


if __name__ == "__main__":
    main()
