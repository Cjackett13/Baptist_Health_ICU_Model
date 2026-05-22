# LOS model — clinician review sample (manual)

**Purpose:** Satisfy checklist item 26 before Baptist pilot.  
**Model:** `xgb_los_pipeline.joblib` (≤12h features → predicted ICU hours)

## How to pick cases (n ≈ 30)

1. **10 largest errors** on test set (`|actual − predicted|` highest)  
2. **10 random** test encounters (seed 42)  
3. **10 stratified** by unit (CICU / CVICU) and primary dx bucket

Export from notebook or SQL using `los_encounter_labels` + model inference.

## Reviewer records per case

| Field | Value |
|-------|--------|
| Encounter ID | |
| Actual LOS (h) | |
| Predicted LOS (h) | |
| Clinically plausible? (Y/N) | |
| If no — why? | |
| Top drivers agree with chart? (Y/N) | |

## SHAP / drivers to cite

See `los_validation_bundle.json` → `shap.mean_abs_shap` (top features):

- `scai_mean_12h`, `scai_first_12h`, `clinical_event_n_12h`, `dx_cat_adhf`, etc.

## Sign-off

| Role | Name | Date | Approved (Y/N) |
|------|------|------|----------------|
| Clinical champion | | | |
| ML lead | | | |
