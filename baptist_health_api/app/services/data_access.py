"""
Data access layer — boundary between the prediction API and the patient data source.

Implement get_patient_data() to fetch raw clinical data for a given encounter.
The returned dict is passed directly to each model's feature extractor.

Expected keys in the returned dict (extend as needed per model):
  df:                  pd.DataFrame, one row per (ENCOUNTER_ID, HOUR_FROM_ADMIT)
                       with base clinical features + 4h rolling stats pre-computed.
  scai_stage_hourly:   pd.DataFrame, full hourly SCAI stage history for the encounter.
  medications:         pd.DataFrame or None, with ENCOUNTER_ID, MEDICATION_CD,
                       ADMIN_START_DT_TM, ADMIN_END_DT_TM columns.
  age:                 int
  days_admitted:       int
  hour_from_admit:     int

For local testing / demo, _load_from_seed() reads patients_seed.json and
adapts the flat vitals list into the DataFrames each feature extractor needs.
"""

import json
from pathlib import Path


async def get_patient_data(encounter_id: int, hour_from_admit: int) -> dict:
    """
    Fetch raw patient clinical data for model feature extraction.

    TODO: Replace with real data source (parquet files, database, EHR API).
    """
    return _load_from_seed(encounter_id, hour_from_admit)


# ── Seed loader (local demo / testing) ────────────────────────────────────────

_SCAI_STAGE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Clinical defaults used when a vital is absent from the seed record.
_VITAL_DEFAULTS: dict[str, float] = {
    "MAP": 70.0, "SBP": 90.0, "DBP": 60.0, "HR": 80.0, "RR": 16.0,
    "SPO2": 95.0, "TEMP": 36.8, "VIS": 0.0, "URINE_OUT_HR": 50.0,
    "LACTATE": 1.0, "CREATININE": 1.0, "BUN": 15.0, "HGB": 10.0,
    "PLT": 200.0, "WBC": 8.0, "INR": 1.1, "HCO3": 22.0,
    "PH": 7.38, "PCO2": 40.0, "ALT": 30.0, "AST": 30.0,
    "TROPONIN_I": 0.01, "NT_PROBNP": 500.0,
    "CVP": 8.0, "PAD": 15.0, "PAS": 25.0, "PAM": 18.0,
    "PCWP": 12.0, "CI": 2.5, "CO": 5.0, "CPO": 0.6,
    "SVR": 1000.0, "SVO2": 65.0, "PAPI": 2.0,
}


def _load_from_seed(encounter_id: int, hour_from_admit: int) -> dict:
    """Load patient data from the Flutter seed JSON and build feature DataFrames."""
    # Primary: bundled inside the app directory — works in Docker and local dev.
    seed_path = Path(__file__).parents[2] / "seed_data" / "patients_seed.json"
    if not seed_path.exists():
        # Fallback: repo-root sibling path for local dev outside Docker.
        seed_path = Path(__file__).parents[3] / "Frontend" / "assets" / "patients_seed.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found at {seed_path}")

    with open(seed_path) as f:
        seed = json.load(f)

    patients = seed if isinstance(seed, list) else seed.get("patients", [])
    for p in patients:
        if str(p.get("encounter_id")) == str(encounter_id):
            p["hour_from_admit"] = hour_from_admit
            p.update(_build_feature_dataframes(p))
            return p

    raise KeyError(f"encounter_id {encounter_id} not found in seed data")


def _build_feature_dataframes(p: dict) -> dict:
    """
    Convert a seed JSON patient record into the DataFrame structure expected by
    Christie's SCAI and vasopressor feature extractors.

    Single-point vitals from the seed are placed into a one-row DataFrame.
    4h rolling stat columns (mean4h, min4h, max4h) are set equal to the current
    value; delta4h and slope4h are set to 0.  The SCAI pipeline then adds 12h/24h
    extended lookbacks on top of those.  Any columns still missing after the
    pipeline runs are filled with 0 inside scai_features.extract() — same pattern
    vasopressor_features.extract() already uses.
    Predictions from this path are structurally valid but not clinically meaningful;
    suitable for end-to-end API testing only.
    """
    import pandas as pd

    enc_id     = int(p["encounter_id"])
    hour       = int(p.get("hour_from_admit", 24))
    scai_stage = _SCAI_STAGE_MAP.get(str(p.get("scai_stage_current", "B")).upper(), 1)
    age        = int(p.get("age", 65))

    vitals: dict[str, float] = {
        v["code"].upper(): float(v["value"])
        for v in p.get("vitals", [])
    }

    # Resolve each vital: seed value takes priority, then clinical default.
    vital_vals: dict[str, float] = {
        k: vitals.get(k, default) for k, default in _VITAL_DEFAULTS.items()
    }

    row: dict = {
        "ENCOUNTER_ID":          enc_id,
        "HOUR_FROM_ADMIT":       hour,
        "SCAI_STAGE_NUM":        scai_stage,
        "AGE":                   age,
        **vital_vals,
        # Device flags
        "ON_MCS":                int(p.get("mechanical_support", {}).get("on_mcs", False)),
        "ON_INTUBATION":         0,
        "ON_CRRT":               0,
        # Vasopressor / inotrope counts derived from medication list
        "N_VASOPRESSORS_ACTIVE": sum(
            1 for m in p.get("medications", []) if m.get("is_vasopressor")
        ),
        "N_INOTROPES_ACTIVE":    0,
        # VIS sub-components (seed doesn't carry individual doses)
        "ENOXIMONE_DOSE":        0.0,
        "MILRINONE_DOSE":        0.0,
        "VASOPRESSIN_DOSE":      0.0,
    }

    # Pre-compute 4h rolling stat columns expected by the SCAI base feature list.
    # scai_pipeline.build_features() receives these as inputs and only adds
    # 12h/24h extended lookbacks on top. With a single observation:
    #   mean4h = min4h = max4h = current value;  delta4h = slope4h = 0.
    for v, val in vital_vals.items():
        row[f"{v}_mean4h"]  = val
        row[f"{v}_min4h"]   = val
        row[f"{v}_max4h"]   = val
        row[f"{v}_delta4h"] = 0.0
        row[f"{v}_slope4h"] = 0.0

    df = pd.DataFrame([row])

    # Minimal hourly SCAI history — one row at the current scoring hour.
    scai_stage_hourly = pd.DataFrame([{
        "ENCOUNTER_ID":    enc_id,
        "HOUR_FROM_ADMIT": hour,
        "SCAI_STAGE_NUM":  scai_stage,
    }])

    # Medications DataFrame for vasopressor pipeline Block-D features.
    meds_rows = [
        {
            "ENCOUNTER_ID":      enc_id,
            "MEDICATION_CD":     m.get("code", "UNKNOWN"),
            "ADMIN_START_DT_TM": m.get("start"),
            "ADMIN_END_DT_TM":   None,
        }
        for m in p.get("medications", [])
    ]
    medications = pd.DataFrame(meds_rows) if meds_rows else None

    return {
        "df":                df,
        "scai_stage_hourly": scai_stage_hourly,
        "medications":       medications,
    }
