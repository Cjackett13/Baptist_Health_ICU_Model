# Length of stay (LOS) model

**Separate from in-hospital mortality** (`Frontend/lib/models/mortality_model.py`).

| | Mortality model | LOS model (this package) |
|--|-----------------|-------------------------|
| Task | Binary death classification | **Regression: hours of stay** |
| Label | `died` | `log1p_los_hours_total` / `los_hours_total` |
| Features | Full-stay patient sidecar | **≤12h snapshot** + train-fold PCA / residuals |
| Artifacts | `synth_cs_data/` or mortality bundle | **`data/los_model/`** |
| Metrics | AUROC, Brier, F1, confusion matrix | **MAE, RMSE, R² (hours)** |

## Performance targets (held-out test)

| Metric | Goal | Current (tuned) |
|--------|------|-----------------|
| **MAE** | ≤ **36 h** (1.5 days) | **~2.4 h** ✓ |
| **RMSE** | ≤ **48 h** (2.0 days) | **~3.1 h** ✓ |
| **R²** | ≥ **0.65** | **~0.98** ✓ |

Targets are defined in `config.py` (`PERFORMANCE_TARGETS`). Training writes `targets_met_test` in `xgb_los_model_meta.json`.

**Synthetic cohort v2:** Regenerate with `Baptist_tester/simulate_cardiogenic_shock_data.py` (LOS coupled to severity/stage/dx, 12h trajectory scaling, `ADMIT_LOS_INDEX_H`), then `los_data_prep.py --force` before train.

## Data layout

| Path | Contents |
|------|----------|
| `data/*.parquet` | Shared synthetic cohort (prep input) |
| `data/los_*` | Prep outputs (features, labels, split manifest) |
| **`data/los_model/`** | **Model artifacts only** |

## Commands (from repo root)

```bash
# 1. Prep (once) — still in Baptist_tester
python3 Baptist_tester/los_data_prep.py --data-dir data --force

# 2. Feature plan
python3 Baptist_tester/los_feature_selection_plan.py --data-dir data

# 3. LOS EDA → data/los_model/eda/
python3 -m models.length_of_stay.eda --data-dir data

# 4. Tune hyperparameters (val MAE) → data/los_model/xgb_los_best_hyperparams.json
python3 -m models.length_of_stay.tune --data-dir data --trials 50 --retrain

# 5. Train LOS model (loads tuned params if present)
python3 -m models.length_of_stay.train --data-dir data

# 6. ML testing checklist audit → data/los_model/los_ml_checklist_audit.md
python3 -m models.length_of_stay.checklist_audit --data-dir data

# 7. Post-retrain artifact validation (manifest, preprocessor leakage, weights)
python3 -m models.length_of_stay.validate_artifacts --data-dir data

# 8. Baptist validation (SHAP, subgroup MAE, residual audit, sanity tests)
python3 -m models.length_of_stay.validate --data-dir data
```

See also:
- `data/los_model/RETRAINING_AND_MONITORING.md`
- `data/los_model/validation/los_validation_bundle.json`
- `data/los_model/validation/CLINICIAN_REVIEW_TEMPLATE.md`

Optional: `pip install shap` for TreeExplainer plots (fallback uses XGBoost `pred_contribs`).

Legacy wrappers: `Baptist_tester/los_model.py`, `Baptist_tester/los_eda.py` delegate here.

## Interpreting metrics

- **MAE (hours):** average absolute error in stay length — primary clinical score.
- **RMSE (hours):** like MAE but penalizes large outliers more.
- **R²:** fraction of LOS variance explained vs always predicting the cohort median.
- **pct_within_36h:** share of test patients with prediction within 1.5 days of truth.
