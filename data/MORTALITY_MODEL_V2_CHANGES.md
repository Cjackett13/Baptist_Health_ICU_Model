# Mortality model v2 — changelog (dataset B)

## Implemented (ChatGPT review)

| # | Change | Files |
|---|--------|-------|
| 1 | `infusion_x_scai_severity` = med_infusion_mean × scai_prop_ge3 | `mortality_duckdb_features.py`, `FEATURE_COLUMNS` |
| 2 | Outliers: **winsorize clip** (train-fold fences), **0 rows dropped** | `mortality_model.py` (`_apply_train_fold_winsorize`) |
| 3 | Demographics: `age_years`, `weight_kg_est`, `sex_bin` (removed `demo_profile_bucket`) | `mortality_model.py` |
| 4 | `min_child_weight`: tune grid **5–20**; fixed JSON **10.0** | `xgb_best_hyperparams.json`, `_xgb_search_param_distributions` |
| 5 | `CALIBRATION_OOF_BRIER_PENALTY` **2.25 → 1.0** | `mortality_model.py` |

## Retrain results (5k cleaned, seed 42)

| Metric | v1 (pre-change) | v2 |
|--------|-----------------|-----|
| n_train / n_test | 3469 / 1153 | **3750 / 1250** (all 5000 kept) |
| Test AUROC | 0.938 | 0.921 |
| Test AUPRC | 0.859 | 0.840 |
| High-SCAI tertile AUROC | 0.744 | **0.757** |
| High-SCAI tertile AUPRC | 0.914 | **0.908** |
| Alert threshold (historical) | 0.43 | 0.51 (v2 retrain) |
| Alert threshold (current holdout, FP≤90) | — | **0.54** (see `mortality_presentation_signoff.md`) |

## Note on high-SCAI AUROC

When death rate ≈74% in the top SCAI tertile, AUROC is a poor headline metric (little class separation to rank). **AUPRC ~0.91** in that tertile indicates strong ranking among positives.

## v4 (pre-processing)

| Change | Detail |
|--------|--------|
| Winsorize | Default upper fence **99.5%** (was 98%); severity features **no upper clip** (lower bound only) |
| Rates | `clinical_events_last_4h`, `clinical_events_per_icu_hour`, `med_admin_per_icu_hour`, `proc_per_icu_hour` |
| Demographics | `demo_profile_bucket` + `frailty_index`; raw `weight_kg_est` / `sex_bin` removed from model inputs |

## v3 (feature + training audit)

| Change | Detail |
|--------|--------|
| Features | `scai_slope_12h`, `scai_max`, `scai_increase_count`; `inf_early_minus_all`, `infusion_early_above_mean` (replaces ratio in model); `clinical_event_residual_within_scai` |
| Trees | `MORTALITY_MIN_N_ESTIMATORS=400` after OOF early-stop on AUPRC |
| Tune | `min_child_weight` grid **3–8**; default tune metric `recall_at_fp_cap` |
| Balance | `scale_pos_weight` mult **2.55** (train neg/pos × mult) |
| Infusion | Model uses `inf_early_minus_all` + `infusion_early_above_mean`; ratio kept in sidecar for audit only |

## Retrain (v3, dataset B)

```bash
export MPLCONFIGDIR="$(pwd)/.mpl_cache_train"
PYTHONPATH=Frontend/lib/models python3 Frontend/lib/models/mortality_model.py \
  --data-dir data/cleaned --model-dir data --seed 42 \
  --refresh-duckdb --tune-n-iter 24 --tune-metric recall_at_fp_cap
```

Do **not** pass `--fixed-n-estimators 120` (under-trains); early stopping enforces **≥400** trees.
