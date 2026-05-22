# LOS model — retraining and monitoring plan

**Model:** ICU length of stay (hours) — `models/length_of_stay/`  
**Artifacts:** `data/los_model/xgb_los_pipeline.joblib`  
**Feature manifest:** 14 raw columns (`los_model_config.MODEL_FEATURE_COLUMNS`) → 21 after one-hot  
**Last validated:** 2026-05-20 (weighted build; test MAE ~9.5 h, R² ~0.77)  
**Related docs:** `FEATURE_WEIGHTS_RATIONALE.md`, `validation/LOS_V2_ROLLING_BLUEPRINT.md`

## When to retrain

| Trigger | Action |
|---------|--------|
| New Baptist Snowflake extract (scheduled quarterly or ad hoc) | Full prep + retrain |
| Test **MAE** drifts **>15%** above baseline for 2 consecutive weeks | Investigate + retrain |
| Test **R²** drops **>0.10** absolute vs last approved build | Investigate + retrain |
| Schema change (new columns, unit codes, SCAI definitions) | Update prep scripts, feature plan, retrain |
| Fairness: any subgroup MAE **>5 h** worse than overall for 2 runs | Review features/thresholds; retrain or adjust |

## Monitoring (production / pilot)

**Weekly (automated):**
- Test-cohort **MAE, RMSE, R²** on a fresh holdout or rolling 20% person split
- **% predictions within 36 h** of actual (clinical tolerance band)
- **Alert volume** if binarized “long stay” rule is used (e.g. pred > 7 days)
- Input schema validation: required ≤12h features present, null rates vs training

**Monthly:**
- Subgroup MAE by `race_cd`, `sex_cd`, `unit_cd`, primary diagnosis bucket
- Permutation importance / SHAP drift: top-10 feature ranking vs last training
- Compare **median predictor baseline** — model should stay clearly better

**Quarterly:**
- Full checklist audit: `python3 -m models.length_of_stay.checklist_audit --data-dir data`
- Full validation bundle: `python3 -m models.length_of_stay.validate --data-dir data`
- Clinician review sample (20–50 cases: largest errors + random)

## Retraining procedure (reproducible)

```bash
# 1. Refresh data (synthetic or Snowflake export → data/)
python3 Baptist_tester/simulate_cardiogenic_shock_data.py --n-patients 5000 --out-dir data --seed 42
python3 Baptist_tester/los_data_prep.py --data-dir data --force

# 2. Feature plan
python3 Baptist_tester/los_feature_selection_plan.py --data-dir data

# 3. Tune (optional)
python3 -m models.length_of_stay.tune --data-dir data --trials 50 --use-feature-weights --retrain

# 4. Train (manifest-enforced 14 features)
python3 -m models.length_of_stay.train --data-dir data

# 4b. Optional: uniform weights ablation
python3 -m models.length_of_stay.train --data-dir data --model-dir data/los_model_uniform --no-feature-weights

# 5. Validate
python3 -m models.length_of_stay.validate_artifacts --data-dir data
python3 -m models.length_of_stay.validate --data-dir data
python3 -m models.length_of_stay.checklist_audit --data-dir data
```

Record in `xgb_los_model_meta.json`: `data_dir`, git commit hash, training date, `metrics_test`, `targets_met_test`.

## Rollback

Keep prior `xgb_los_pipeline.joblib` as `xgb_los_pipeline.joblib.bak.<YYYYMMDD>` before promoting a new build.

## Known limitations (document for compliance)

- Trained on **synthetic cardiogenic-shock cohort** until Baptist EMR is connected.
- **`race_cd` / `ethnicity_cd` excluded from training** — subgroup fairness via `subgroup_mae.py` + `person.parquet` only.
- **XGB feature weights** are intentional (clinical vs admin); see `FEATURE_WEIGHTS_RATIONALE.md`.
- **v1 features** censored at ≤12h; rolling checkpoints documented in `validation/LOS_V2_ROLLING_BLUEPRINT.md`.
- Performance on real EMR data is expected to differ; re-validate all targets after first real-data train.

## Contacts / ownership

| Role | Responsibility |
|------|----------------|
| ML engineer | Retrain, monitoring jobs, artifact versioning |
| Clinical champion | Threshold approval, subgroup fairness sign-off |
| Data engineering | Snowflake → `data/` refresh, schema contracts |
