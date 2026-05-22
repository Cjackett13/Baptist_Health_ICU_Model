"""
simulate_cardiogenic_shock_data.py
-----------------------------------
Synthetic ICU dataset generator for cardiogenic shock progression modeling,
aligned to Cerner Millennium table/column conventions.

Outputs (parquet files) in --out-dir:
  person.parquet, encounter.parquet, diagnosis.parquet,
  clinical_event.parquet, medication_admin.parquet, procedure_event.parquet,
  scai_stage_hourly.parquet, code_value.parquet

See data_dictionary.md for full schema and semantics.

Usage:
    python simulate_cardiogenic_shock_data.py --n-patients 5000 --out-dir ./data --seed 42
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference vocabularies
# ---------------------------------------------------------------------------

PRINCIPAL_DX = [
    # (icd10, text, weight)
    ("I21.4", "Non-ST elevation myocardial infarction",        0.20),
    ("I21.3", "ST elevation myocardial infarction, unspec.",   0.15),
    ("I50.21","Acute systolic (congestive) heart failure",     0.18),
    ("I50.23","Acute on chronic systolic heart failure",       0.12),
    ("I46.9", "Cardiac arrest, cause unspecified",             0.05),
    ("I49.01","Ventricular fibrillation",                      0.03),
    ("I49.02","Ventricular flutter",                           0.02),
    ("I40.9", "Acute myocarditis, unspecified",                0.05),
    ("I42.0", "Dilated cardiomyopathy",                        0.05),
    ("I71.01","Dissection of thoracic aorta",                  0.03),
    ("I35.0", "Nonrheumatic aortic (valve) stenosis",          0.05),
    ("Z95.1", "Presence of aortocoronary bypass graft (post-op)", 0.07),
]

SECONDARY_DX_POOL = [
    ("I10",   "Essential (primary) hypertension"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("N17.9", "Acute kidney failure, unspecified"),
    ("J96.01","Acute respiratory failure with hypoxia"),
    ("E78.5", "Hyperlipidemia, unspecified"),
    ("I48.91","Unspecified atrial fibrillation"),
    ("I25.10","Atherosclerotic heart disease of native coronary artery"),
    ("D64.9", "Anemia, unspecified"),
    ("F41.9", "Anxiety disorder, unspecified"),
]

EVENT_CODE_MAP = [
    # (event_cd, title, units, event_class_cd)
    ("HR",          "Heart Rate",                       "bpm",      "VITAL"),
    ("SBP",         "Systolic Blood Pressure",          "mmHg",     "VITAL"),
    ("DBP",         "Diastolic Blood Pressure",         "mmHg",     "VITAL"),
    ("MAP",         "Mean Arterial Pressure",           "mmHg",     "VITAL"),
    ("RR",          "Respiratory Rate",                 "/min",     "VITAL"),
    ("SPO2",        "Peripheral Oxygen Saturation",     "%",        "VITAL"),
    ("TEMP",        "Temperature",                      "degC",     "VITAL"),
    ("URINE_OUT_HR","Urine Output - Last Hour",         "mL",       "IO"),
    ("CVP",         "Central Venous Pressure",          "mmHg",     "HEMODYNAMIC"),
    ("PAS",         "Pulmonary Artery Systolic",        "mmHg",     "HEMODYNAMIC"),
    ("PAD",         "Pulmonary Artery Diastolic",       "mmHg",     "HEMODYNAMIC"),
    ("PAM",         "Pulmonary Artery Mean",            "mmHg",     "HEMODYNAMIC"),
    ("PCWP",        "Pulmonary Capillary Wedge Pressure","mmHg",    "HEMODYNAMIC"),
    ("CO",          "Cardiac Output",                   "L/min",    "HEMODYNAMIC"),
    ("CI",          "Cardiac Index",                    "L/min/m2", "HEMODYNAMIC"),
    ("SVO2",        "Mixed Venous Oxygen Saturation",   "%",        "HEMODYNAMIC"),
    ("SVR",         "Systemic Vascular Resistance",     "dyn.s/cm5","HEMODYNAMIC"),
    ("CPO",         "Cardiac Power Output",             "W",        "CALCULATED"),
    ("PAPI",        "Pulmonary Artery Pulsatility Index","",        "CALCULATED"),
    ("LACTATE",     "Lactate, Arterial",                "mmol/L",   "LAB"),
    ("CREATININE",  "Creatinine, Serum",                "mg/dL",    "LAB"),
    ("BUN",         "Blood Urea Nitrogen",              "mg/dL",    "LAB"),
    ("NT_PROBNP",   "NT-proBNP",                        "pg/mL",    "LAB"),
    ("TROPONIN_I",  "Troponin I, High-Sensitivity",     "ng/L",     "LAB"),
    ("PH",          "Arterial pH",                      "",         "LAB"),
    ("PCO2",        "Arterial pCO2",                    "mmHg",     "LAB"),
    ("HCO3",        "Bicarbonate",                      "mEq/L",    "LAB"),
    ("AST",         "Aspartate Aminotransferase",       "U/L",      "LAB"),
    ("ALT",         "Alanine Aminotransferase",         "U/L",      "LAB"),
    ("WBC",         "White Blood Cell Count",           "K/uL",     "LAB"),
    ("HGB",         "Hemoglobin",                       "g/dL",     "LAB"),
    ("PLT",         "Platelet Count",                   "K/uL",     "LAB"),
    ("INR",         "International Normalized Ratio",   "",         "LAB"),
    ("VIS",         "Vasoactive-Inotrope Score",        "",         "CALCULATED"),
    ("SCAI_STAGE",  "SCAI Shock Stage",                 "",         "CALCULATED"),
]

VASOPRESSORS = ["NOREPINEPHRINE", "EPINEPHRINE", "VASOPRESSIN", "PHENYLEPHRINE", "DOPAMINE"]
INOTROPES    = ["DOBUTAMINE", "MILRINONE"]

MED_DISPLAY = {
    "NOREPINEPHRINE": "Norepinephrine",
    "EPINEPHRINE":    "Epinephrine",
    "VASOPRESSIN":    "Vasopressin",
    "PHENYLEPHRINE":  "Phenylephrine",
    "DOPAMINE":       "Dopamine",
    "DOBUTAMINE":     "Dobutamine",
    "MILRINONE":      "Milrinone",
    "FUROSEMIDE":     "Furosemide",
    "HEPARIN":        "Heparin",
    "BIVALIRUDIN":    "Bivalirudin",
}

MED_UNITS = {
    "NOREPINEPHRINE": "mcg/kg/min",
    "EPINEPHRINE":    "mcg/kg/min",
    "VASOPRESSIN":    "units/min",
    "PHENYLEPHRINE":  "mcg/kg/min",
    "DOPAMINE":       "mcg/kg/min",
    "DOBUTAMINE":     "mcg/kg/min",
    "MILRINONE":      "mcg/kg/min",
    "FUROSEMIDE":     "mg/hr",
    "HEPARIN":        "units/hr",
    "BIVALIRUDIN":    "mg/kg/hr",
}

STAGES = ["A", "B", "C", "D", "E"]
STAGE_NUM = {s: i for i, s in enumerate(STAGES)}

# ---------------------------------------------------------------------------
# Patient-level config
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    person_id: int
    encounter_id: int
    age: int
    sex: str
    race: str
    ethnicity: str
    weight_kg: float
    bsa_m2: float
    admit_dt: datetime
    principal_dx: Tuple[str, str]
    secondary_dx: List[Tuple[str, str]]
    los_hours: int
    admit_los_index_h: float       # admit-time acuity index (LOS formula w/o noise)
    severity: float                # latent severity, 0–1
    initial_stage: str
    stage_path: List[str] = field(default_factory=list)
    died: bool = False
    death_hour: Optional[int] = None
    has_pa_catheter: bool = False
    pa_catheter_start_hour: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_choice(rng: np.random.Generator, items_weights):
    """items_weights: iterable of tuples where the LAST element is the weight."""
    items = [tuple(x[:-1]) for x in items_weights]
    weights = np.array([x[-1] for x in items_weights], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(items), p=weights)
    return items[idx]


def sample_age_sex(rng: np.random.Generator) -> Tuple[int, str]:
    # Cardiac ICU: skewed older, slightly more male
    age = int(np.clip(rng.normal(68, 12), 22, 95))
    sex = "M" if rng.random() < 0.62 else "F"
    return age, sex


def sample_race_ethnicity(rng: np.random.Generator) -> Tuple[str, str]:
    race = rng.choice(
        ["WHITE", "BLACK", "ASIAN", "OTHER", "UNKNOWN"],
        p=[0.55, 0.22, 0.06, 0.13, 0.04],
    )
    ethn = rng.choice(["HISPANIC", "NON_HISPANIC", "UNKNOWN"], p=[0.28, 0.68, 0.04])
    return str(race), str(ethn)


def compute_admit_los_index_h(severity: float, initial_stage: str, principal_icd: str) -> float:
    """Admit-time hours index from severity/stage/dx (no stochastic noise)."""
    stage_idx = STAGE_NUM[initial_stage]
    core = 42.0 + 62.0 * float(severity) + 16.0 * float(stage_idx)
    if principal_icd.startswith("I50"):
        core += 18.0
    elif principal_icd == "I21.3":
        core += 10.0
    elif principal_icd.startswith("I46") or principal_icd.startswith("I49"):
        core -= 8.0
    elif principal_icd == "Z95.1":
        core += 12.0
    return float(core)


def sample_los_hours(
    rng: np.random.Generator,
    severity: float,
    initial_stage: str,
    principal_icd: str,
) -> int:
    """
    ICU LOS (hours) tied to admit-time severity, SCAI stage, and principal dx.

    Makes ≤12h aggregates (SCAI, infusions, procedures) predictable without using
    post-admit leakage. Death during simulation may truncate below this draw.
    """
    core = compute_admit_los_index_h(severity, initial_stage, principal_icd)
    # Tight additive noise so admit_los_index_h + 12h features support target R²
    return int(np.clip(core + rng.normal(0.0, 3.0), 18, 14 * 24))


def sample_initial_stage(rng: np.random.Generator, severity: float) -> str:
    """Higher severity → higher starting stage."""
    # Probabilities depend on severity
    if severity < 0.25:
        probs = [0.65, 0.30, 0.05, 0.00, 0.00]
    elif severity < 0.5:
        probs = [0.30, 0.45, 0.22, 0.03, 0.00]
    elif severity < 0.75:
        probs = [0.05, 0.30, 0.45, 0.18, 0.02]
    else:
        probs = [0.00, 0.10, 0.40, 0.40, 0.10]
    return str(rng.choice(STAGES, p=probs))


def stage_transition(
    rng: np.random.Generator,
    current: str,
    severity: float,
    on_mcs: bool,
    vis: float,
    *,
    los_intensity: float = 1.0,
) -> str:
    """
    Hourly transition. Severity, MCS presence, and current VIS modulate
    deterioration vs recovery probability.
    """
    idx = STAGE_NUM[current]

    # Base hourly P(deteriorate) at avg severity (~0.5), no support
    base_p_worse = {0: 0.020, 1: 0.030, 2: 0.040, 3: 0.060, 4: 0.000}[idx]
    base_p_better = {0: 0.000, 1: 0.040, 2: 0.030, 3: 0.025, 4: 0.020}[idx]

    sev_scaler = 0.5 + severity                # 0.5–1.5
    p_worse = base_p_worse * sev_scaler
    p_better = base_p_better / sev_scaler

    # MCS reduces deterioration, modestly improves recovery
    if on_mcs:
        p_worse *= 0.55
        p_better *= 1.30

    # High VIS = clinically unstable, increases deterioration
    if vis > 20:
        p_worse *= 1.4
    if vis > 40:
        p_worse *= 1.4

    # Planned longer ICU stay → persist in higher-acuity stages (12h aggregates track LOS)
    li = float(np.clip(los_intensity, 0.5, 2.5))
    p_worse *= li ** 0.58
    p_better /= li ** 0.55

    p_worse = float(np.clip(p_worse, 0.0, 0.5))
    p_better = float(np.clip(p_better, 0.0, 0.5))
    p_same = 1.0 - p_worse - p_better

    r = rng.random()
    if r < p_worse and idx < 4:
        return STAGES[idx + 1]
    elif r < p_worse + p_better and idx > 0:
        return STAGES[idx - 1]
    else:
        return current


# ---------------------------------------------------------------------------
# Vital / lab sampling conditioned on stage
# ---------------------------------------------------------------------------

# Stage-conditional means and SDs for hourly vitals.
# Values reflect typical CICU physiology by SCAI stage.
STAGE_VITAL_PARAMS = {
    "A": dict(HR=(80, 12), SBP=(125, 15), DBP=(75, 10), RR=(16, 3), SPO2=(97, 2),
              TEMP=(36.8, 0.4), URINE_OUT_HR=(70, 25)),
    "B": dict(HR=(102, 14), SBP=(98, 12), DBP=(62, 10), RR=(20, 4), SPO2=(95, 3),
              TEMP=(36.7, 0.5), URINE_OUT_HR=(50, 20)),
    "C": dict(HR=(112, 16), SBP=(88, 12), DBP=(55, 9), RR=(24, 5), SPO2=(93, 4),
              TEMP=(36.5, 0.6), URINE_OUT_HR=(28, 15)),
    "D": dict(HR=(120, 18), SBP=(82, 14), DBP=(50, 10), RR=(27, 6), SPO2=(91, 5),
              TEMP=(36.2, 0.8), URINE_OUT_HR=(15, 12)),
    "E": dict(HR=(128, 22), SBP=(72, 16), DBP=(44, 11), RR=(30, 7), SPO2=(88, 6),
              TEMP=(35.8, 1.0), URINE_OUT_HR=(8, 10)),
}

STAGE_LAB_PARAMS = {
    "A": dict(LACTATE=(1.2, 0.4), CREATININE=(1.0, 0.3), BUN=(15, 5),
              NT_PROBNP=(800, 400), TROPONIN_I=(50, 30), PH=(7.40, 0.03),
              PCO2=(40, 4), HCO3=(24, 2), AST=(30, 10), ALT=(28, 10),
              WBC=(8, 2.5), HGB=(13, 1.5), PLT=(220, 60), INR=(1.1, 0.1)),
    "B": dict(LACTATE=(1.8, 0.6), CREATININE=(1.2, 0.4), BUN=(20, 7),
              NT_PROBNP=(2500, 1200), TROPONIN_I=(200, 150), PH=(7.37, 0.04),
              PCO2=(38, 5), HCO3=(22, 3), AST=(45, 20), ALT=(40, 18),
              WBC=(10, 3), HGB=(12.5, 1.6), PLT=(210, 70), INR=(1.2, 0.15)),
    "C": dict(LACTATE=(3.2, 1.1), CREATININE=(1.6, 0.6), BUN=(28, 10),
              NT_PROBNP=(6000, 3000), TROPONIN_I=(800, 600), PH=(7.32, 0.05),
              PCO2=(36, 6), HCO3=(20, 3), AST=(120, 60), ALT=(95, 50),
              WBC=(12, 3.5), HGB=(11.8, 1.8), PLT=(190, 80), INR=(1.4, 0.2)),
    "D": dict(LACTATE=(5.5, 1.8), CREATININE=(2.2, 0.9), BUN=(40, 14),
              NT_PROBNP=(12000, 6000), TROPONIN_I=(2500, 1500), PH=(7.25, 0.07),
              PCO2=(34, 8), HCO3=(17, 4), AST=(400, 250), ALT=(300, 200),
              WBC=(14, 4), HGB=(10.8, 1.9), PLT=(150, 70), INR=(1.7, 0.3)),
    "E": dict(LACTATE=(9.0, 3.0), CREATININE=(2.8, 1.1), BUN=(55, 18),
              NT_PROBNP=(20000, 9000), TROPONIN_I=(5000, 3000), PH=(7.15, 0.10),
              PCO2=(32, 10), HCO3=(13, 5), AST=(900, 500), ALT=(700, 400),
              WBC=(16, 5), HGB=(9.8, 2.0), PLT=(110, 60), INR=(2.2, 0.5)),
}

STAGE_HEMO_PARAMS = {
    # PA-catheter-derived. Sampled when has_pa_catheter and catheter has been placed.
    "A": dict(CVP=(8, 2), PAS=(28, 5), PAD=(12, 3), PAM=(18, 4), PCWP=(12, 3),
              CO=(5.2, 0.8), CI=(2.8, 0.4), SVO2=(70, 4), SVR=(1100, 200)),
    "B": dict(CVP=(11, 3), PAS=(34, 6), PAD=(16, 4), PAM=(22, 5), PCWP=(16, 4),
              CO=(4.4, 0.8), CI=(2.4, 0.4), SVO2=(64, 5), SVR=(1300, 250)),
    "C": dict(CVP=(14, 4), PAS=(42, 8), PAD=(22, 5), PAM=(29, 6), PCWP=(22, 5),
              CO=(3.5, 0.8), CI=(1.9, 0.4), SVO2=(56, 6), SVR=(1500, 300)),
    "D": dict(CVP=(17, 5), PAS=(48, 10), PAD=(26, 7), PAM=(33, 8), PCWP=(26, 6),
              CO=(2.8, 0.8), CI=(1.5, 0.4), SVO2=(48, 7), SVR=(1700, 350)),
    "E": dict(CVP=(20, 6), PAS=(52, 12), PAD=(30, 8), PAM=(37, 9), PCWP=(30, 7),
              CO=(2.2, 0.8), CI=(1.2, 0.4), SVO2=(40, 8), SVR=(1900, 400)),
}

# Lab sampling cadence by class (hours between draws by stage)
def lab_due_this_hour(stage: str, hour: int, rng: np.random.Generator) -> bool:
    if stage in ("A",):
        return hour % 24 == 0
    if stage in ("B",):
        return hour % 12 == 0
    if stage in ("C",):
        return hour % 6 == 0
    if stage in ("D", "E"):
        # q4h, with frequent in-between ABGs (~50% of unscheduled hours)
        return (hour % 4 == 0) or (rng.random() < 0.2)
    return hour % 24 == 0


def sample_with_ar1(prev: Optional[float], mean: float, sd: float,
                    rng: np.random.Generator, rho: float = 0.7) -> float:
    """AR(1) smoothing so trajectories aren't pure noise."""
    if prev is None:
        return float(rng.normal(mean, sd))
    return float(rho * prev + (1 - rho) * mean + rng.normal(0, sd * np.sqrt(1 - rho**2)))


# ---------------------------------------------------------------------------
# Vasoactive medication trajectories
# ---------------------------------------------------------------------------

def stage_vasoactive_targets(stage: str) -> Dict[str, float]:
    """Average infusion rate for each med (0 if not running) at this stage.
    Used to drive realistic treatment patterns."""
    if stage == "A":
        return {}
    if stage == "B":
        return {"NOREPINEPHRINE": 0.03}                       # low-dose pressor
    if stage == "C":
        return {"NOREPINEPHRINE": 0.10, "DOBUTAMINE": 5.0}
    if stage == "D":
        return {"NOREPINEPHRINE": 0.25, "EPINEPHRINE": 0.05,
                "DOBUTAMINE": 7.5, "VASOPRESSIN": 0.04}
    if stage == "E":
        return {"NOREPINEPHRINE": 0.45, "EPINEPHRINE": 0.15,
                "VASOPRESSIN": 0.04, "DOBUTAMINE": 10.0}
    return {}


def compute_vis(active_meds: Dict[str, float], weight_kg: float = 75.0) -> float:
    """Vasoactive-Inotrope Score (Gaies 2010 / 2014).

    All component drug rates are in mcg/kg/min EXCEPT vasopressin, which is
    units/kg/min. Infusion records store vasopressin in units/min (clinical
    convention), so we convert using the patient's weight here.
    """
    rate = lambda m: active_meds.get(m, 0.0)
    vaso_per_kg = rate("VASOPRESSIN") / max(weight_kg, 1.0)  # units/min → units/kg/min
    vis = (
        rate("DOPAMINE") +
        rate("DOBUTAMINE") +
        100 * rate("EPINEPHRINE") +
        100 * rate("NOREPINEPHRINE") +
        10  * rate("MILRINONE") +
        10_000 * vaso_per_kg +
        100 * rate("PHENYLEPHRINE")
    )
    return float(vis)


# ---------------------------------------------------------------------------
# Cohort generation
# ---------------------------------------------------------------------------

def generate_cohort(n_patients: int, seed: int, start_date: datetime) -> List[Patient]:
    rng = np.random.default_rng(seed)
    patients: List[Patient] = []
    for i in range(n_patients):
        age, sex = sample_age_sex(rng)
        race, ethn = sample_race_ethnicity(rng)
        weight_kg = float(np.clip(rng.normal(82 if sex == "M" else 70, 16), 45, 160))
        height_m = float(np.clip(rng.normal(1.75 if sex == "M" else 1.62, 0.08), 1.45, 1.95))
        bsa = float(np.sqrt(height_m * 100 * weight_kg / 3600))  # Mosteller

        # severity ~ Beta(2,5) → mean ~0.29; most patients are not shock-prone,
        # only a tail develops advanced shock. Principal-dx-driven boosts apply below.
        severity = float(rng.beta(2, 5))
        principal = weighted_choice(rng, PRINCIPAL_DX)  # (icd, text)
        principal_tuple = (principal[0], principal[1])
        if principal[0].startswith("I46") or principal[0].startswith("I49"):
            severity = float(np.clip(severity + 0.35, 0, 1))
        if principal[0] == "I21.3":  # STEMI
            severity = float(np.clip(severity + 0.15, 0, 1))
        if principal[0].startswith("I50"):  # ADHF — modest boost
            severity = float(np.clip(severity + 0.05, 0, 1))
        if principal[0] == "Z95.1":  # post-cardiotomy
            severity = float(np.clip(severity + 0.10, 0, 1))

        n_sec = int(rng.integers(0, 5))
        secondary = [SECONDARY_DX_POOL[k]
                     for k in rng.choice(len(SECONDARY_DX_POOL),
                                         size=min(n_sec, len(SECONDARY_DX_POOL)),
                                         replace=False)]

        initial = sample_initial_stage(rng, severity)
        admit_index = compute_admit_los_index_h(severity, initial, principal[0])
        los_hours = sample_los_hours(rng, severity, initial, principal[0])

        admit_dt = start_date + timedelta(hours=int(rng.integers(0, 90 * 24)))

        p = Patient(
            person_id=100_000 + i,
            encounter_id=9_000_000 + i,
            age=age, sex=sex, race=race, ethnicity=ethn,
            weight_kg=round(weight_kg, 1), bsa_m2=round(bsa, 2),
            admit_dt=admit_dt,
            principal_dx=principal_tuple,
            secondary_dx=secondary,
            los_hours=los_hours,
            admit_los_index_h=admit_index,
            severity=severity,
            initial_stage=initial,
        )
        patients.append(p)
    return patients


# ---------------------------------------------------------------------------
# Per-patient simulation
# ---------------------------------------------------------------------------

def simulate_patient(p: Patient, rng: np.random.Generator) -> Dict[str, list]:
    """Return dict of row-lists for each output table for this patient."""
    rows_ce: list = []
    rows_med: list = []
    rows_proc: list = []
    rows_scai: list = []

    prev_vitals: Dict[str, Optional[float]] = {k: None for k in
        ["HR","SBP","DBP","RR","SPO2","TEMP","URINE_OUT_HR"]}
    prev_hemo: Dict[str, Optional[float]] = {k: None for k in
        ["CVP","PAS","PAD","PAM","PCWP","CO","CI","SVO2","SVR"]}
    prev_labs: Dict[str, Optional[float]] = {k: None for k in STAGE_LAB_PARAMS["A"]}

    stage = p.initial_stage
    on_mcs = False
    mcs_modality: Optional[str] = None
    mcs_start_hour: Optional[int] = None
    intubated = False
    intubation_start_hour: Optional[int] = None
    crrt_active = False
    crrt_start_hour: Optional[int] = None

    # Medication state — currently running rates per med
    active_meds: Dict[str, float] = {}
    med_start_hour: Dict[str, int] = {}

    # Decide if/when PA catheter will be placed (more likely if starts C+ or escalates)
    place_pa_threshold_severity = 0.55
    pa_decision_made = False

    # event id counter (global handled by caller; we'll use offsets)
    event_idx = 0

    def emit_ce(hour: int, code: str, value: float, units: str, event_class: str,
                contributor: str = "MONITOR"):
        nonlocal event_idx
        title = next((t for c,t,u,cl in EVENT_CODE_MAP if c == code), code)
        # normalcy heuristic (simple per code)
        normalcy = normalcy_for(code, value)
        rows_ce.append((
            p.encounter_id, p.person_id, code, title, event_class,
            p.admit_dt + timedelta(hours=hour),
            round(float(value), 3) if value is not None else None,
            units, normalcy, contributor,
            event_idx
        ))
        event_idx += 1

    # Pre-load: SCAI label per hour
    stage_history: List[str] = []
    los_intensity = float(np.clip(p.los_hours / 80.0, 0.6, 2.4))

    for hour in range(p.los_hours + 1):
        # Decide PA catheter placement
        if not pa_decision_made and (
            STAGE_NUM[stage] >= 2 or p.severity >= place_pa_threshold_severity
        ):
            if rng.random() < (0.50 if STAGE_NUM[stage] >= 2 else 0.25):
                p.has_pa_catheter = True
                p.pa_catheter_start_hour = hour
                rows_proc.append((
                    p.encounter_id, p.person_id, "RIGHT_HEART_CATH",
                    "Right heart catheterization / PA catheter placement",
                    p.admit_dt + timedelta(hours=hour),
                    p.admit_dt + timedelta(hours=hour),
                    "CATH",
                ))
            pa_decision_made = True

        # MCS decision — escalate at D or refractory C
        if not on_mcs and STAGE_NUM[stage] >= 2:
            # Probability scales with stage and severity
            p_mcs_hour = {2: 0.005, 3: 0.05, 4: 0.20}.get(STAGE_NUM[stage], 0.0)
            if rng.random() < p_mcs_hour:
                # Pick modality: lower stage -> IABP / Impella; E -> ECMO
                if STAGE_NUM[stage] == 4:
                    mcs_modality = str(rng.choice(["VA_ECMO", "IMPELLA_5_5"], p=[0.65, 0.35]))
                elif STAGE_NUM[stage] == 3:
                    mcs_modality = str(rng.choice(["IMPELLA_CP", "IMPELLA_5_5", "VA_ECMO", "IABP"],
                                                  p=[0.45, 0.20, 0.20, 0.15]))
                else:
                    mcs_modality = str(rng.choice(["IABP", "IMPELLA_CP"], p=[0.55, 0.45]))
                on_mcs = True
                mcs_start_hour = hour
                rows_proc.append((
                    p.encounter_id, p.person_id, mcs_modality,
                    f"{mcs_modality.replace('_',' ').title()} placement",
                    p.admit_dt + timedelta(hours=hour),
                    None, "MCS",
                ))

        # Intubation decision (stage D/E or severe hypoxia)
        if not intubated and STAGE_NUM[stage] >= 3 and rng.random() < 0.10:
            intubated = True
            intubation_start_hour = hour
            rows_proc.append((
                p.encounter_id, p.person_id, "INTUBATION",
                "Endotracheal intubation", p.admit_dt + timedelta(hours=hour),
                None, "INTUBATION",
            ))

        # CRRT decision (worsening AKI; proxy: stage D/E with high severity)
        if not crrt_active and STAGE_NUM[stage] >= 3 and p.severity > 0.6 \
           and rng.random() < 0.03:
            crrt_active = True
            crrt_start_hour = hour
            rows_proc.append((
                p.encounter_id, p.person_id, "CRRT_INITIATION",
                "Continuous renal replacement therapy initiation",
                p.admit_dt + timedelta(hours=hour),
                None, "CRRT",
            ))

        # Update medications toward stage targets (with hysteresis)
        targets = stage_vasoactive_targets(stage)
        # Decay/discontinue meds no longer indicated
        for med in list(active_meds.keys()):
            if med not in targets:
                # Wean: drop to 0 with some lag
                active_meds[med] *= 0.5
                if active_meds[med] < 0.005:
                    # Close the infusion record
                    rows_med.append((
                        p.encounter_id, p.person_id, med, MED_DISPLAY[med],
                        p.admit_dt + timedelta(hours=med_start_hour[med]),
                        p.admit_dt + timedelta(hours=hour),
                        round(float(active_meds[med]), 4),  # final rate
                        MED_UNITS[med], "IV_CONTINUOUS",
                    ))
                    del active_meds[med]
                    del med_start_hour[med]
        # Start / titrate up toward target (scale with planned LOS → infusion_mean_12h signal)
        med_scale = los_intensity ** 0.72
        for med, tgt in targets.items():
            tgt_eff = float(tgt) * med_scale
            if med not in active_meds:
                active_meds[med] = float(tgt_eff * rng.uniform(0.5, 1.0))
                med_start_hour[med] = hour
            else:
                # AR step toward target
                active_meds[med] = float(0.7 * active_meds[med] + 0.3 * tgt_eff) \
                                   * float(rng.uniform(0.95, 1.10))

        # Heparin/bivalirudin/furosemide running infusions (simplified)
        if hour == 0 and rng.random() < 0.35:
            anti = "HEPARIN" if rng.random() < 0.7 else "BIVALIRUDIN"
            active_meds[anti] = 12.0 if anti == "HEPARIN" else 0.15
            med_start_hour[anti] = 0
        if STAGE_NUM[stage] >= 1 and "FUROSEMIDE" not in active_meds \
           and rng.random() < 0.02:
            active_meds["FUROSEMIDE"] = float(rng.uniform(5, 20))
            med_start_hour["FUROSEMIDE"] = hour

        # VIS (weight-aware so vasopressin component is realistic)
        vis = compute_vis(active_meds, weight_kg=p.weight_kg)

        # Vitals (every hour)
        vp = STAGE_VITAL_PARAMS[stage]
        for code in ["HR", "SBP", "DBP", "RR", "SPO2", "TEMP", "URINE_OUT_HR"]:
            mu, sd = vp[code]
            val = sample_with_ar1(prev_vitals[code], mu, sd, rng)
            # SpO2/temperature physiologic clipping
            if code == "SPO2":
                val = float(np.clip(val, 60, 100))
            if code == "TEMP":
                val = float(np.clip(val, 33.0, 41.0))
            if code == "URINE_OUT_HR":
                val = float(max(0.0, val))
            prev_vitals[code] = val
            units = next(u for c, t, u, cl in EVENT_CODE_MAP if c == code)
            emit_ce(hour, code, val, units, "VITAL")
        # MAP (derived)
        map_val = (prev_vitals["SBP"] + 2 * prev_vitals["DBP"]) / 3.0
        emit_ce(hour, "MAP", map_val, "mmHg", "VITAL")

        # Hemodynamics (only when PA catheter is in)
        if p.has_pa_catheter and p.pa_catheter_start_hour is not None \
           and hour >= p.pa_catheter_start_hour:
            hp = STAGE_HEMO_PARAMS[stage]
            hemo_vals: Dict[str, float] = {}
            for code in ["CVP","PAS","PAD","PAM","PCWP","CO","CI","SVO2","SVR"]:
                mu, sd = hp[code]
                v = sample_with_ar1(prev_hemo[code], mu, sd, rng, rho=0.8)
                prev_hemo[code] = v
                hemo_vals[code] = v
                units = next(u for c, t, u, cl in EVENT_CODE_MAP if c == code)
                emit_ce(hour, code, v, units, "HEMODYNAMIC")
            # Derived: CPO and PAPi
            cpo = (map_val * hemo_vals["CO"]) / 451.0
            papi = (hemo_vals["PAS"] - hemo_vals["PAD"]) / max(hemo_vals["CVP"], 1e-3)
            emit_ce(hour, "CPO", cpo, "W", "CALCULATED")
            emit_ce(hour, "PAPI", papi, "", "CALCULATED")

        # Labs (cadence based on stage)
        if lab_due_this_hour(stage, hour, rng):
            lp = STAGE_LAB_PARAMS[stage]
            for code, (mu, sd) in lp.items():
                v = sample_with_ar1(prev_labs[code], mu, sd, rng, rho=0.6)
                # Physiologic clipping
                if code == "PH":      v = float(np.clip(v, 6.85, 7.60))
                if code == "LACTATE": v = float(max(0.3, v))
                if code in ("CREATININE","BUN","TROPONIN_I","NT_PROBNP","WBC","HGB","PLT"):
                    v = float(max(0.1, v))
                prev_labs[code] = v
                units = next(u for c, t, u, cl in EVENT_CODE_MAP if c == code)
                emit_ce(hour, code, v, units, "LAB", contributor="LAB")

        # VIS and SCAI stage (always emitted hourly)
        emit_ce(hour, "VIS", vis, "", "CALCULATED")
        emit_ce(hour, "SCAI_STAGE", float(STAGE_NUM[stage]), "", "CALCULATED")
        rows_scai.append((
            p.encounter_id, hour, p.admit_dt + timedelta(hours=hour),
            stage, STAGE_NUM[stage]
        ))
        stage_history.append(stage)

        # Extra labs/events in first 12h for longer planned stays (clinical_event_n_12h signal)
        if hour < 12 and rng.random() < 0.12 * (los_intensity - 0.35):
            lp = STAGE_LAB_PARAMS[stage]
            code = str(rng.choice(list(lp.keys())))
            mu, sd = lp[code]
            v = sample_with_ar1(prev_labs.get(code), mu, sd, rng, rho=0.5)
            prev_labs[code] = v
            units = next(u for c, t, u, cl in EVENT_CODE_MAP if c == code)
            emit_ce(hour, code, float(v), units, "LAB", contributor="LAB")

        # ----- Stage transition for the NEXT hour -----
        next_stage = stage_transition(
            rng, stage, p.severity, on_mcs, vis, los_intensity=los_intensity
        )

        # Mortality at extremis (stage E only): per-hour death probability
        if next_stage == "E":
            p_die_hour = 0.010 * (0.5 + p.severity)  # severity-scaled (reduced for LOS predictability)
            if rng.random() < p_die_hour:
                p.died = True
                p.death_hour = hour
                # Keep planned LOS_HOURS for modeling label; stop trajectory simulation only
                break

        stage = next_stage

    # Final-row med cleanup (close any active infusions at discharge/death)
    for med, rate in active_meds.items():
        rows_med.append((
            p.encounter_id, p.person_id, med, MED_DISPLAY[med],
            p.admit_dt + timedelta(hours=med_start_hour[med]),
            p.admit_dt + timedelta(hours=p.los_hours),
            round(float(rate), 4),
            MED_UNITS[med], "IV_CONTINUOUS",
        ))
    # MCS/CRRT/intubation: close end times at discharge if still active
    # (rows_proc currently has end=None for ongoing)
    rows_proc_closed = []
    for row in rows_proc:
        enc, pid, code, txt, start, end, cat = row
        if end is None:
            end = p.admit_dt + timedelta(hours=p.los_hours)
        rows_proc_closed.append((enc, pid, code, txt, start, end, cat))

    p.stage_path = stage_history

    return dict(
        clinical_event=rows_ce,
        medication_admin=rows_med,
        procedure_event=rows_proc_closed,
        scai=rows_scai,
    )


def normalcy_for(code: str, value: Optional[float]) -> str:
    """Crude reference-range tag — used by CE.NORMALCY_CD."""
    if value is None:
        return "UNKNOWN"
    ranges = {
        "HR": (60, 100, 50, 130),
        "SBP": (90, 140, 70, 180),
        "MAP": (65, 100, 55, 120),
        "SPO2": (94, 100, 88, 100),
        "LACTATE": (0.5, 2.0, 0.5, 4.0),
        "CREATININE": (0.6, 1.3, 0.4, 2.5),
        "PH": (7.35, 7.45, 7.20, 7.55),
    }
    if code not in ranges:
        return "NORMAL"
    lo, hi, clo, chi = ranges[code]
    if value < clo:   return "CRITICAL_LOW"
    if value > chi:   return "CRITICAL_HIGH"
    if value < lo:    return "LOW"
    if value > hi:    return "HIGH"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Mortality post-hoc + discharge disposition
# ---------------------------------------------------------------------------

def finalize_outcomes(p: Patient, rng: np.random.Generator) -> Tuple[str, Optional[datetime]]:
    """Decide discharge disposition consistent with the stage trajectory."""
    if p.died:
        return "EXPIRED", p.admit_dt + timedelta(hours=p.death_hour or p.los_hours)

    # Apply mortality risk based on the worst stage reached
    if not p.stage_path:
        return "HOME", None
    worst = max(STAGE_NUM[s] for s in p.stage_path)
    p_die_post = {0: 0.002, 1: 0.02, 2: 0.12, 3: 0.30, 4: 0.55}[worst]
    if rng.random() < p_die_post:
        # Death late in stay
        death_hr = int(p.los_hours - rng.integers(0, max(1, p.los_hours // 4 + 1)))
        p.died = True
        p.death_hour = death_hr
        return "EXPIRED", p.admit_dt + timedelta(hours=death_hr)

    # Survivors: disposition depends on severity
    if worst <= 1:
        disp = rng.choice(["HOME", "SNF", "REHAB"], p=[0.80, 0.10, 0.10])
    elif worst == 2:
        disp = rng.choice(["HOME", "SNF", "REHAB", "LTACH"], p=[0.55, 0.20, 0.20, 0.05])
    else:
        disp = rng.choice(["HOME", "SNF", "REHAB", "LTACH", "HOSPICE"],
                          p=[0.30, 0.25, 0.25, 0.15, 0.05])
    return str(disp), None


# ---------------------------------------------------------------------------
# Assemble final dataframes
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=5000)
    ap.add_argument("--out-dir", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-date", type=str, default="2024-10-01",
                    help="Earliest admit date (YYYY-MM-DD)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start_date = datetime.fromisoformat(args.start_date)
    rng = np.random.default_rng(args.seed)

    print(f"[1/4] Generating cohort of {args.n_patients} patients…")
    patients = generate_cohort(args.n_patients, args.seed, start_date)

    all_ce, all_med, all_proc, all_scai = [], [], [], []
    print("[2/4] Simulating per-patient trajectories…")
    for i, p in enumerate(patients):
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(patients)} patients simulated")
        prng = np.random.default_rng(args.seed + p.person_id)
        out = simulate_patient(p, prng)
        all_ce.extend(out["clinical_event"])
        all_med.extend(out["medication_admin"])
        all_proc.extend(out["procedure_event"])
        all_scai.extend(out["scai"])

    print("[3/4] Finalizing outcomes & writing tables…")

    # PERSON
    person_rows = []
    for p in patients:
        bdate = p.admit_dt - timedelta(days=int(p.age * 365.25 + rng.integers(0, 365)))
        person_rows.append((
            p.person_id, f"LAST{p.person_id}", f"FIRST{p.person_id}",
            bdate.replace(hour=0, minute=0, second=0, microsecond=0),
            p.sex, p.race, p.ethnicity, None  # DECEASED_DT_TM filled after outcomes
        ))
    df_person = pd.DataFrame(person_rows, columns=[
        "PERSON_ID","NAME_LAST_TXT","NAME_FIRST_TXT","BIRTH_DT_TM",
        "SEX_CD","RACE_CD","ETHNICITY_CD","DECEASED_DT_TM"
    ])

    # ENCOUNTER + finalize outcomes
    enc_rows = []
    for p in patients:
        disp, death_dt = finalize_outcomes(p, rng)
        if death_dt is not None:
            df_person.loc[df_person.PERSON_ID == p.person_id, "DECEASED_DT_TM"] = death_dt
        admit_type = str(rng.choice(["EMERGENCY", "URGENT", "ELECTIVE", "TRANSFER"],
                                    p=[0.60, 0.20, 0.10, 0.10]))
        admit_src = str(rng.choice(["ED", "OUTSIDE_HOSPITAL", "DIRECT", "OR"],
                                   p=[0.55, 0.20, 0.15, 0.10]))
        unit = str(rng.choice(["CICU", "CVICU"], p=[0.65, 0.35]))
        dch_dt = p.admit_dt + timedelta(hours=p.los_hours)
        enc_rows.append((
            p.encounter_id, p.person_id, "INPATIENT", admit_type, admit_src,
            p.admit_dt, dch_dt, disp, "BHSF_MAIN", unit, float(p.los_hours),
            float(p.admit_los_index_h),
        ))
    df_enc = pd.DataFrame(enc_rows, columns=[
        "ENCOUNTER_ID","PERSON_ID","ENCNTR_TYPE_CD","ADMIT_TYPE_CD","ADMIT_SRC_CD",
        "REG_DT_TM","DISCH_DT_TM","DISCH_DISPOSITION_CD","FACILITY_CD","UNIT_CD","LOS_HOURS",
        "ADMIT_LOS_INDEX_H",
    ])

    # DIAGNOSIS
    diag_rows = []
    diag_id = 5_000_000
    for p in patients:
        diag_rows.append((
            diag_id, p.encounter_id, p.person_id,
            p.principal_dx[0], p.principal_dx[1], 1, "PRINCIPAL", p.admit_dt
        ))
        diag_id += 1
        for j, (code, txt) in enumerate(p.secondary_dx):
            diag_rows.append((
                diag_id, p.encounter_id, p.person_id,
                code, txt, j + 2, "SECONDARY",
                p.admit_dt + timedelta(hours=int(rng.integers(0, 24)))
            ))
            diag_id += 1
    df_diag = pd.DataFrame(diag_rows, columns=[
        "DIAGNOSIS_ID","ENCOUNTER_ID","PERSON_ID","NOMENCLATURE_CD",
        "DIAGNOSIS_TXT","DIAG_PRIORITY","CLASSIFICATION_CD","DIAGNOSIS_DT_TM"
    ])

    # CLINICAL_EVENT
    ce_cols = ["ENCOUNTER_ID","PERSON_ID","EVENT_CD","EVENT_TITLE_TXT",
               "EVENT_CLASS_CD","EVENT_END_DT_TM","RESULT_VAL","RESULT_UNITS_CD",
               "NORMALCY_CD","CONTRIBUTOR_SYSTEM_CD","_idx"]
    df_ce = pd.DataFrame(all_ce, columns=ce_cols)
    df_ce.insert(0, "EVENT_ID", (7_700_000_000 + np.arange(len(df_ce))).astype("int64"))
    df_ce.drop(columns=["_idx"], inplace=True)

    # MEDICATION_ADMIN
    df_med = pd.DataFrame(all_med, columns=[
        "ENCOUNTER_ID","PERSON_ID","MEDICATION_CD","MEDICATION_TXT",
        "ADMIN_START_DT_TM","ADMIN_END_DT_TM","INFUSION_RATE","RATE_UNIT_CD","ROUTE_CD"
    ])
    df_med.insert(0, "MED_ADMIN_ID", (6_600_000_000 + np.arange(len(df_med))).astype("int64"))
    df_med.insert(2, "ORDER_ID", (4_400_000_000 + np.arange(len(df_med))).astype("int64"))

    # PROCEDURE_EVENT
    df_proc = pd.DataFrame(all_proc, columns=[
        "ENCOUNTER_ID","PERSON_ID","NOMENCLATURE_CD","PROCEDURE_TXT",
        "PROC_START_DT_TM","PROC_END_DT_TM","PROC_CATEGORY_CD"
    ])
    df_proc.insert(0, "PROCEDURE_ID", (8_800_000_000 + np.arange(len(df_proc))).astype("int64"))

    # SCAI_STAGE_HOURLY
    df_scai = pd.DataFrame(all_scai, columns=[
        "ENCOUNTER_ID","HOUR_FROM_ADMIT","EVENT_DT_TM","SCAI_STAGE_CD","SCAI_STAGE_NUM"
    ])

    # CODE_VALUE
    df_code = pd.DataFrame(EVENT_CODE_MAP, columns=[
        "EVENT_CD","EVENT_TITLE_TXT","RESULT_UNITS_CD","EVENT_CLASS_CD"
    ])

    print("[4/4] Writing parquet files…")
    for name, df in [
        ("person", df_person),
        ("encounter", df_enc),
        ("diagnosis", df_diag),
        ("clinical_event", df_ce),
        ("medication_admin", df_med),
        ("procedure_event", df_proc),
        ("scai_stage_hourly", df_scai),
        ("code_value", df_code),
    ]:
        path = os.path.join(args.out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"   {name:22s} rows={len(df):>9,}  →  {path}")

    # Summary statistics
    n_died = (df_enc.DISCH_DISPOSITION_CD == "EXPIRED").sum()
    print("\n=== Cohort summary ===")
    print(f"   Patients:           {len(df_person):,}")
    print(f"   Encounters:         {len(df_enc):,}")
    print(f"   In-hospital deaths: {n_died:,} ({n_died/len(df_enc)*100:.1f}%)")
    print(f"   Median LOS (hours): {df_enc.LOS_HOURS.median():.1f}")
    print(f"   Stage distribution (hour-level):")
    print(df_scai.SCAI_STAGE_CD.value_counts().sort_index().to_string())
    print(f"   Total clinical events: {len(df_ce):,}")


if __name__ == "__main__":
    main()
