# Mortality model — retraining and monitoring (dataset B: `data/`)

Trained on **LOS-cleaned 5,000-patient cohort** (`data/cleaned/*.parquet`), same inclusion rules as `los_cohort_manifest.json`.

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
| FN count at alert threshold | Weekly | Review against `mortality_alert_target_max_fn` |
| Subgroup AUROC (sex, race, SCAI) | Monthly | Flag any group Δ < −0.05 vs overall |
| DuckDB residual features | Per retrain | Rebuild sidecar from **train-only** SQL to remove cohort leakage |

## Retrain pipeline (dataset B)

```bash
# 1. Refresh LOS-cleaned tables (if raw bundle changed)
python3 Baptist_tester/los_data_prep.py --data-dir data --write-cleaned-parquets

# 2. Train mortality (fixed hyperparams from xgb_best_hyperparams.json; same FEATURE_COLUMNS)
export MPLCONFIGDIR="$(pwd)/.mpl_cache_train"
PYTHONPATH=Frontend/lib/models python3 -m mortality_model \
  --data-dir data/cleaned \
  --model-dir data \
  # Residuals: train-fold SCAI-bin means in prepare_train_test_features (not full-cohort DuckDB windows)
  --refresh-duckdb \
  --seed 42 \
  --tune-n-iter 24 --tune-metric recall_at_fp_cap
  # Optional fast path: --xgb-params-json data/xgb_best_hyperparams.json (omit --fixed-n-estimators 120)

# Model v2 notes: train-fold winsorize (no row drops), separate age/weight/sex,
# infusion_x_scai_severity interaction, min_child_weight>=5 in tune grid (fixed JSON uses 10).

# 3. Checklist
PYTHONPATH=Frontend/lib/models python3 -m mortality_checklist_audit \
  --data-dir data/cleaned --model-dir data
```

## Alert threshold (holdout)

- Train-OOF may select ~0.50; if holdout **FP > 90**, use the lowest threshold with **FP ≤ 90** (currently **~0.54** on dataset B).
- **Hard gate:** `test_meets_fp_cap_90_at_alert` in meta. **FN ≤ 30** is aspirational, not a deployment gate.
- Presentation Q&A: `data/mortality_presentation_signoff.md`.

## Sanity audit (checklist items 17, 19, 21)

```bash
PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_sanity_audit.py \
  --data-dir data/cleaned --model-dir data
PYTHONPATH=Frontend/lib/models python3 -m mortality_checklist_audit \
  --data-dir data/cleaned --model-dir data
```

## Sync FastAPI demo artifacts

```bash
python back_end/scripts/sync_mortality_artifacts_from_data.py
```

Copies meta (including `death_alert_threshold`), feature columns, and joblib into `back_end/app/artifact/`. The API reads **`app/artifact/xgb_mortality_model_meta.json`** for alerts (`GET /v1/mortality/model-meta`).

## Artifacts (under `data/`)

- `xgb_mortality_pipeline.joblib`
- `xgb_mortality_model_meta.json`
- `mortality_feature_columns.json`
- `cleaned/duckdb_patient_features.parquet`
- `mortality_ml_checklist_audit.json` / `.md`
- `mortality_presentation_signoff.md`
