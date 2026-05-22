# Two-model ICU workflow (mortality + length of stay)

Baptist ICU decision support uses **two complementary models** on the same cardiogenic-shock cohort (`data/`, 5,000 cleaned encounters). They answer different clinical questions and use different feature windows by design.

## In-hospital mortality (full-stay snapshot)

- **Question:** What is this patient’s risk of **dying during this hospitalization**?
- **Features:** Patient-level aggregates over the **full ICU stay** (medications, procedures, SCAI course, clinical-event load, demographics bucket) from `duckdb_patient_features.parquet`.
- **Output:** Calibrated probability + **alert threshold** (~0.54 on dataset B holdout) tuned to meet **FP ≤ 90** on the test split; OOF fit ~0.50.
- **Use:** Triage at/after workup when longitudinal severity and treatment intensity are known — e.g. multidisciplinary rounds, goals-of-care framing, escalation review.

## Length of stay (≤12h checkpoint)

- **Question:** How much **time remains** in this ICU stay (hours), given what we know **early**?
- **Features:** **Censored at 12 hours** after admission (SCAI trajectory, early meds/procedures/events, admit demographics) — no post-12h or discharge information.
- **Output:** Predicted **remaining LOS in hours** (regression).
- **Use:** Bed management, staffing, and early expectations while the patient is still in the first shift — before the full stay pattern is visible.

## Why both exist

Mortality needs **complete-stay signal** (how sick they became and how intensely they were treated). LOS needs **early, leak-free inputs** so predictions are actionable at hour 12. A single model cannot serve both without either leaking future data (LOS) or missing late deterioration (mortality).

## Typical workflow

1. **Admission → 12h:** LOS model estimates remaining stay; mortality not required for bed planning.
2. **12h → discharge:** Mortality snapshot updated as clinical course evolves; LOS may be refreshed if you add rolling checkpoints later.
3. **Alerts:** Mortality threshold flags high death risk; LOS flags unusually long predicted stay — different operational responses.

Artifacts: `data/xgb_mortality_*` (mortality), `data/los_model/xgb_los_*` (LOS).
