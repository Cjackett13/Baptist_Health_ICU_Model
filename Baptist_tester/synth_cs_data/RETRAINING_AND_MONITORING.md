# Mortality model — retraining and monitoring

## Retrain triggers

- **Data drift:** test AUROC or AUPRC drops >0.03 vs training holdout; Brier rises >0.02.
- **Label drift:** in-hospital death rate shifts >5 pp vs training cohort.
- **Volume:** ≥500 new ICU encounters or quarterly scheduled refresh.
- **Feature schema change:** `FEATURE_COLUMNS` or DuckDB sidecar SQL changes.

## Monitoring (production)

| Signal | Cadence | Action |
|--------|---------|--------|
| AUROC / AUPRC / Brier on labeled holdout | Weekly | Alert if Brier > 0.12 or AUPRC < 0.5× prevalence |
| Alert rate at `death_alert_threshold` | Daily | Target ~25–40%; investigate if >55% |
| FN count at alert threshold | Weekly | Keep FN ≤ `mortality_alert_target_max_fn` (30 on test policy) |
| Subgroup AUROC (sex, race, SCAI) | Monthly | Flag any group Δ < −0.05 vs overall |
| DuckDB residual features | Per retrain | Rebuild sidecar from **train-only** SQL to remove cohort leakage |

## Retrain pipeline

```bash
PYTHONPATH=Frontend/lib/models python3 mortality_model.py \
  --data-dir Baptist_tester/synth_cs_data --seed 42

PYTHONPATH=Frontend/lib/models python3 -m mortality_checklist_audit \
  --data-dir Baptist_tester/synth_cs_data
```

## Artifacts to version

- `xgb_mortality_pipeline.joblib`
- `xgb_mortality_model_meta.json`
- `mortality_ml_checklist_audit.json` / `.md`
- `duckdb_patient_features.parquet` (rebuild with train-only residuals before real data)
