#!/usr/bin/env python3
"""
Rebuild patients_seed.json clinical display + predictions from parquet cohort rows.

Keeps cosmetic ``name``, ``rank``, ``demo_hospital_id``, ``room_number`` from the
current seed; replaces vitals, diagnoses, meds, SCAI, age, and IDs so UI matches models.

  PYTHONPATH=back_end BAPTIST_DATA_DIR=data python3 back_end/scripts/export_patients_seed_from_data.py
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

from app.patient_predictions import load_seed_patients, reload_seed_patients  # noqa: E402
from app.patient_seed_export import sync_patient_clinical_from_parquet  # noqa: E402

_scripts = REPO_ROOT / "back_end" / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from refresh_patients_seed import refresh_patient  # noqa: E402

SEED_PATH = REPO_ROOT / "Frontend" / "assets" / "patients_seed.json"


def main() -> None:
    patients = load_seed_patients()
    exported: list[dict[str, Any]] = []
    for p in patients:
        synced = sync_patient_clinical_from_parquet(p)
        exported.append(refresh_patient(synced))

    payload = {
        "seed_metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "script": "back_end/scripts/export_patients_seed_from_data.py",
            "patient_count": len(exported),
            "note": (
                "Clinical fields + predictions exported from the same PERSON_ID / "
                "ENCOUNTER_ID rows used by mortality and LOS models."
            ),
        },
        "patients": exported,
    }
    SEED_PATH.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    reload_seed_patients()
    print(f"Exported {len(exported)} patients -> {SEED_PATH}")
    s = exported[0]
    print(
        f"Sample {s['name']}: id={s['id']} scai={s['scai_stage_current']} "
        f"mort={s['predictions']['mortality_risk']:.3f}"
    )


if __name__ == "__main__":
    main()
