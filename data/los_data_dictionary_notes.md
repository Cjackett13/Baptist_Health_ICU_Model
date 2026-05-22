# Data Dictionary — Cardiogenic Shock Progression Cohort
### Cerner Millennium–Aligned Synthetic ICU Dataset

**Cohort:** Adult patients (≥18y) admitted to a cardiac ICU with a primary cardiology-related diagnosis (acute MI, decompensated HF, arrhythmia with hemodynamic compromise, post-cardiotomy). Each patient has hourly time-series observations spanning the ICU stay (median 4–6 days, range 1–14).

**Schema convention:** Table and column names follow Cerner Millennium / HealtheIntent conventions (uppercase, suffixed `_ID`, `_CD`, `_DT_TM`, `_FLAG`, `_TXT`). `CD` columns are coded reference values; in production these resolve via `CODE_VALUE`. `EVENT_CD` values in `CLINICAL_EVENT` are Cerner-specific numeric codes — here we use mnemonic strings for readability and provide a separate code map.

**Reference timestamp:** `EVENT_END_DT_TM` is the clinically relevant event time (per Cerner convention, the "as-of" time for the observation). All datetimes are timezone-naive UTC.

---

## 1. `PERSON` — Patient master

| Column | Type | Description | Example |
|---|---|---|---|
| `PERSON_ID` | INT64 | Unique person identifier (Cerner millennium person key) | `100001` |
| `NAME_LAST_TXT` | VARCHAR | Last name (synthetic) | `DOE` |
| `NAME_FIRST_TXT` | VARCHAR | First name (synthetic) | `JANE` |
| `BIRTH_DT_TM` | DATETIME | Date of birth | `1948-03-12` |
| `SEX_CD` | VARCHAR | Sex code (`M`, `F`) | `F` |
| `RACE_CD` | VARCHAR | Race code (`WHITE`, `BLACK`, `ASIAN`, `OTHER`, `UNKNOWN`) | `BLACK` |
| `ETHNICITY_CD` | VARCHAR | Ethnicity (`HISPANIC`, `NON_HISPANIC`, `UNKNOWN`) | `HISPANIC` |
| `DECEASED_DT_TM` | DATETIME | Date/time of death, NULL if alive | `2025-04-14 03:22` |

---

## 2. `ENCOUNTER` — ICU admission

| Column | Type | Description | Example |
|---|---|---|---|
| `ENCOUNTER_ID` | INT64 | Unique encounter key | `9000123` |
| `PERSON_ID` | INT64 | FK → `PERSON.PERSON_ID` | `100001` |
| `ENCNTR_TYPE_CD` | VARCHAR | Encounter type (`INPATIENT`) | `INPATIENT` |
| `ADMIT_TYPE_CD` | VARCHAR | Admission type (`EMERGENCY`, `URGENT`, `ELECTIVE`, `TRANSFER`) | `EMERGENCY` |
| `ADMIT_SRC_CD` | VARCHAR | Admit source (`ED`, `OUTSIDE_HOSPITAL`, `DIRECT`, `OR`) | `ED` |
| `REG_DT_TM` | DATETIME | ICU admit datetime | `2025-03-01 14:20` |
| `DISCH_DT_TM` | DATETIME | ICU disposition datetime | `2025-03-08 09:10` |
| `DISCH_DISPOSITION_CD` | VARCHAR | Discharge disposition (`HOME`, `SNF`, `REHAB`, `LTACH`, `EXPIRED`, `HOSPICE`, `AMA`) | `HOME` |
| `FACILITY_CD` | VARCHAR | Facility | `BHSF_MAIN` |
| `UNIT_CD` | VARCHAR | Unit code (`CICU`, `CVICU`, `MICU`) | `CICU` |
| `LOS_HOURS` | FLOAT | Derived length of stay in hours | `163.0` |

---

## 3. `DIAGNOSIS` — Encounter-level diagnoses (ICD-10)

| Column | Type | Description | Example |
|---|---|---|---|
| `DIAGNOSIS_ID` | INT64 | Surrogate key | `5000001` |
| `ENCOUNTER_ID` | INT64 | FK → `ENCOUNTER` | `9000123` |
| `PERSON_ID` | INT64 | FK → `PERSON` | `100001` |
| `NOMENCLATURE_CD` | VARCHAR | ICD-10-CM code | `I21.4` |
| `DIAGNOSIS_TXT` | VARCHAR | Description | `Non-ST elevation MI` |
| `DIAG_PRIORITY` | INT | 1 = principal, 2+ = secondary | `1` |
| `CLASSIFICATION_CD` | VARCHAR | (`PRINCIPAL`, `SECONDARY`, `ADMITTING`) | `PRINCIPAL` |
| `DIAGNOSIS_DT_TM` | DATETIME | Date diagnosed | `2025-03-01 15:00` |

**Principal diagnosis categories simulated:**
| ICD-10 | Description | % of cohort |
|---|---|---|
| `I21.x` | Acute MI (STEMI / NSTEMI) | 35% |
| `I50.x` | Acute decompensated heart failure | 30% |
| `I46.x` / `I49.x` | Cardiac arrest / malignant arrhythmia | 10% |
| `I40.x` / `I42.x` | Myocarditis / cardiomyopathy | 10% |
| `I71.x` / `I35.x` | Aortic / valvular emergency | 8% |
| `Z95.x` post-op | Post-cardiotomy | 7% |

---

## 4. `CLINICAL_EVENT` — All time-stamped observations
*(Cerner's central observation table — vitals, labs, calculated scores, hemodynamics)*

| Column | Type | Description | Example |
|---|---|---|---|
| `EVENT_ID` | INT64 | Unique event key | `7700001234` |
| `ENCOUNTER_ID` | INT64 | FK → `ENCOUNTER` | `9000123` |
| `PERSON_ID` | INT64 | FK → `PERSON` | `100001` |
| `EVENT_CD` | VARCHAR | Cerner event code (mnemonic used here) | `HR` |
| `EVENT_TITLE_TXT` | VARCHAR | Human-readable label | `Heart Rate` |
| `EVENT_CLASS_CD` | VARCHAR | (`VITAL`, `LAB`, `HEMODYNAMIC`, `CALCULATED`, `IO`) | `VITAL` |
| `EVENT_END_DT_TM` | DATETIME | Clinical event time (use this) | `2025-03-02 03:00` |
| `RESULT_VAL` | FLOAT | Numeric result | `112.0` |
| `RESULT_UNITS_CD` | VARCHAR | Units | `bpm` |
| `NORMALCY_CD` | VARCHAR | (`NORMAL`, `HIGH`, `LOW`, `CRITICAL_HIGH`, `CRITICAL_LOW`) | `HIGH` |
| `CONTRIBUTOR_SYSTEM_CD` | VARCHAR | Source system (`MONITOR`, `LAB`, `MANUAL`, `DEVICE`) | `MONITOR` |

### `EVENT_CD` reference (subset)

**Vitals (hourly):**
| Code | Title | Units | Typical Range |
|---|---|---|---|
| `HR` | Heart rate | bpm | 60–100 |
| `SBP` | Systolic BP | mmHg | 90–140 |
| `DBP` | Diastolic BP | mmHg | 60–90 |
| `MAP` | Mean arterial pressure | mmHg | 65–100 |
| `RR` | Respiratory rate | /min | 12–20 |
| `SPO2` | Pulse oximetry | % | 95–100 |
| `TEMP` | Temperature | °C | 36.0–37.5 |
| `URINE_OUT_HR` | Urine output, last hour | mL | 30–100 |

**Hemodynamics (when PA catheter / arterial line in place):**
| Code | Title | Units |
|---|---|---|
| `CVP` | Central venous pressure | mmHg |
| `PAS` | PA systolic | mmHg |
| `PAD` | PA diastolic | mmHg |
| `PAM` | PA mean | mmHg |
| `PCWP` | Pulmonary cap wedge | mmHg |
| `CO` | Cardiac output | L/min |
| `CI` | Cardiac index | L/min/m² |
| `SVO2` | Mixed venous O₂ sat | % |
| `SVR` | Systemic vascular resistance | dyn·s/cm⁵ |

**Calculated (derived from raw hemodynamics, computed each hour):**
| Code | Title | Formula |
|---|---|---|
| `CPO` | Cardiac power output | (MAP × CO) / 451 |
| `PAPI` | PA pulsatility index | (PAS − PAD) / CVP |

**Labs (q6h–q24h):**
| Code | Title | Units | Typical Range |
|---|---|---|---|
| `LACTATE` | Serum lactate | mmol/L | 0.5–2.0 |
| `CREATININE` | Serum creatinine | mg/dL | 0.6–1.3 |
| `BUN` | Blood urea nitrogen | mg/dL | 7–20 |
| `NT_PROBNP` | NT-proBNP | pg/mL | <300 (no HF) |
| `TROPONIN_I` | High-sensitivity troponin I | ng/L | <14 |
| `PH` | Arterial pH | — | 7.35–7.45 |
| `PCO2` | Arterial pCO₂ | mmHg | 35–45 |
| `HCO3` | Bicarbonate | mEq/L | 22–28 |
| `AST` | Aspartate aminotransferase | U/L | 10–40 |
| `ALT` | Alanine aminotransferase | U/L | 7–56 |
| `WBC` | White blood cell count | K/uL | 4.0–11.0 |
| `HGB` | Hemoglobin | g/dL | 12–17 |
| `PLT` | Platelet count | K/uL | 150–400 |
| `INR` | International normalized ratio | — | 0.8–1.2 |

**Calculated severity scores (hourly):**
| Code | Title | Notes |
|---|---|---|
| `VIS` | Vasoactive-Inotrope Score | dopamine + dobutamine + 100×epi + 100×NE + 10×milrinone + 10000×vasopressin |
| `SCAI_STAGE` | SCAI shock stage | A/B/C/D/E (the prediction target backbone) |

---

## 5. `MEDICATION_ADMIN` — Vasoactive infusions and key meds

| Column | Type | Description | Example |
|---|---|---|---|
| `MED_ADMIN_ID` | INT64 | Surrogate key | `6600001` |
| `ENCOUNTER_ID` | INT64 | FK → `ENCOUNTER` | `9000123` |
| `PERSON_ID` | INT64 | FK → `PERSON` | `100001` |
| `ORDER_ID` | INT64 | FK to originating order | `4400789` |
| `MEDICATION_CD` | VARCHAR | Medication mnemonic | `NOREPINEPHRINE` |
| `MEDICATION_TXT` | VARCHAR | Display name | `Norepinephrine` |
| `ADMIN_START_DT_TM` | DATETIME | Infusion start | `2025-03-02 04:00` |
| `ADMIN_END_DT_TM` | DATETIME | Infusion end (NULL if active) | `2025-03-03 18:00` |
| `INFUSION_RATE` | FLOAT | Rate value | `0.08` |
| `RATE_UNIT_CD` | VARCHAR | Rate unit | `mcg/kg/min` |
| `ROUTE_CD` | VARCHAR | (`IV_CONTINUOUS`, `IV_BOLUS`, `PO`) | `IV_CONTINUOUS` |

**Medications simulated:**
- Vasopressors: `NOREPINEPHRINE`, `EPINEPHRINE`, `VASOPRESSIN`, `PHENYLEPHRINE`, `DOPAMINE`
- Inotropes: `DOBUTAMINE`, `MILRINONE`
- Other: `FUROSEMIDE` (bolus / drip), `HEPARIN`, `BIVALIRUDIN`

---

## 6. `PROCEDURE_EVENT` — Mechanical circulatory support & key procedures

| Column | Type | Description | Example |
|---|---|---|---|
| `PROCEDURE_ID` | INT64 | Surrogate key | `8800001` |
| `ENCOUNTER_ID` | INT64 | FK → `ENCOUNTER` | `9000123` |
| `PERSON_ID` | INT64 | FK → `PERSON` | `100001` |
| `NOMENCLATURE_CD` | VARCHAR | Procedure code (CPT / ICD-10-PCS mnemonic) | `IMPELLA_CP` |
| `PROCEDURE_TXT` | VARCHAR | Display name | `Impella CP placement` |
| `PROC_START_DT_TM` | DATETIME | Start | `2025-03-02 08:30` |
| `PROC_END_DT_TM` | DATETIME | End / explant | `2025-03-05 16:00` |
| `PROC_CATEGORY_CD` | VARCHAR | (`MCS`, `CATH`, `INTUBATION`, `CRRT`, `OTHER`) | `MCS` |

**MCS modalities simulated:** `IABP`, `IMPELLA_CP`, `IMPELLA_5_5`, `VA_ECMO`, `LVAD_DURABLE`
**Other procedures:** `INTUBATION`, `CRRT_INITIATION`, `RIGHT_HEART_CATH`, `LEFT_HEART_CATH`, `PCI`

---

## 7. Derived / target tables

### `SCAI_STAGE_HOURLY`
Hourly SCAI shock stage assignment (also embedded in `CLINICAL_EVENT` as `EVENT_CD = SCAI_STAGE`).

| Column | Type | Description |
|---|---|---|
| `ENCOUNTER_ID` | INT64 | FK |
| `HOUR_FROM_ADMIT` | INT | 0, 1, 2, … LOS_HOURS |
| `EVENT_DT_TM` | DATETIME | |
| `SCAI_STAGE_CD` | VARCHAR | `A`, `B`, `C`, `D`, `E` |
| `SCAI_STAGE_NUM` | INT | 0–4 (ordinal) |

**SCAI staging (Baran et al., 2019; abbreviated):**
- **A — At risk:** No shock; high-risk substrate (large MI, severe HF).
- **B — Beginning:** Hypotension or tachycardia without hypoperfusion. SBP < 90 or MAP < 65 *or* HR > 100, lactate normal.
- **C — Classic:** Hypoperfusion requiring intervention. Lactate ≥ 2.0, requires vasopressor/inotrope or MCS, MAP often < 65 despite support.
- **D — Deteriorating:** Worsening despite initial intervention. Escalating support, rising lactate (≥ 5), worsening end-organ function.
- **E — Extremis:** Refractory shock, cardiac arrest, multi-organ failure, on maximal support including ECMO.

### Prediction target (model task)
**Primary outcome:** `Y_PROGRESSION_6H` — binary, 1 if SCAI stage worsens by ≥ 1 within next 6 hours from the prediction time (i.e., A→B, B→C, C→D, or D→E), else 0.

**Prediction time:** Every hour starting at hour 4 of ICU stay (allowing 4h of feature buildup), excluding hours where patient is already at stage E or has died.

**Secondary outcomes (for future analyses):**
- `Y_MORTALITY_INHOSP` — in-hospital death
- `Y_MCS_ESCALATION_24H` — initiation of any MCS within next 24h
- `Y_ECMO_24H` — initiation of VA-ECMO within next 24h

---

## Cohort generation notes & known synthetic-data caveats

- Trajectories are governed by a per-patient latent severity that drives transition probabilities between SCAI stages. Vitals, labs, and treatments are sampled conditional on stage with realistic noise and physiologic correlation.
- Mortality probability is conditional on the maximum SCAI stage reached (A: <1%, B: ~5%, C: ~25%, D: ~45%, E: ~70%) — consistent with published registries.
- Missingness is *not* MCAR: labs are denser when patients are sicker; PA catheter–derived hemodynamics are present only when a Swan was placed (~40% of stage C+ patients).
- All identifiers are synthetic. No PHI.
- This is intended for prototyping, governance review, and pipeline validation — **not** for clinical inference, publication of effect sizes, or vendor benchmarking.

---

## Files produced by the simulator

| File | Description |
|---|---|
| `person.parquet` | `PERSON` table |
| `encounter.parquet` | `ENCOUNTER` table |
| `diagnosis.parquet` | `DIAGNOSIS` table |
| `clinical_event.parquet` | `CLINICAL_EVENT` table (largest — long format, hourly) |
| `medication_admin.parquet` | `MEDICATION_ADMIN` table |
| `procedure_event.parquet` | `PROCEDURE_EVENT` table |
| `scai_stage_hourly.parquet` | Hourly SCAI label series |
| `code_value.parquet` | `EVENT_CD` mnemonic → display reference |

---

## LOS model appendix (Dataset B, v1 ≤12h snapshot)

**Grain:** one row per `ENCOUNTER_ID` (5000 encounters = 5000 persons in current cohort).

**Labels:** `los_hours_total` (raw hours), `remaining_los_hours` (hours remaining at 12h), train on `log1p_los_hours_total`.

**Model features (14 raw columns before one-hot):**

- `med_infusion_mean_12h`
- `scai_prop_ge3_12h`
- `current_scai_12h`
- `proc_n_12h`
- `med_distinct_12h`
- `clinical_event_n_12h`
- `dx_cat_arrhythmia_arrest`
- `med_per_clinical_event_12h`
- `scai_slope_12h`
- `age_years`
- `sex_bin`
- `admit_type_cd`
- `admit_src_cd`
- `unit_cd`

**Transforms:** see `los_feature_columns.json` and `los_final_model_features.json`.

**Split:** `los_split_manifest.json` — patient-level train/val/test IDs only (seed 42).

**Artifacts:** `los_modeling_frame.parquet`, `los_feature_audit.json`, `los_cleaning_log.json`, `eda/los_eda_*.html`.
