# LOS validation bundle (Baptist)

Artifacts: `data/los_model/validation/los_validation_bundle.json`

## Artifact validation (manifest / preprocessor)
- All checks pass: True

## SHAP / importance
- SHAP available: True
- See `mean_abs_shap` and `permutation_importance` in JSON

## Subgroup fairness (test MAE)
- Overall MAE: 9.5 h
- Flagged groups (MAE >5h vs overall): 0

## infusion_residual_within_scai_12h
- PASS for strict policy: bin means fit on train only. WARN: DuckDB audit column uses full-cohort means — do not use parquet residual for training; pipeline overwrites with train-fold version.

## Sanity
- Direction pass rate: 1.0

Retraining plan: `data/los_model/RETRAINING_AND_MONITORING.md`
