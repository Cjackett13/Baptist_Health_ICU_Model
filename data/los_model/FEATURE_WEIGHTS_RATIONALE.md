# LOS model — XGBoost feature weights (intentional design)

**Model:** `data/los_model/xgb_los_pipeline.joblib` (14-feature manifest, v1 ≤12h)  
**Not learned from data** — weights are set in `models/length_of_stay/features.py` (`DEFAULT_RAW_FEATURE_WEIGHTS`) and optionally overridden by tiers in `los_training_feature_plan.json` (Spearman vs `los_hours_total` on the exploratory cohort).

## Why use feature weights?

XGBoost `feature_weights` scale how often each column is considered for splits. We use them to:

1. **Emphasize early clinical acuity** (SCAI, infusion, events) over administrative codes.
2. **De-emphasize admit type / unit** one-hots that can absorb variance without improving bedside usefulness.
3. Align training with the **feature selection plan** (high-|ρ| ≤12h signals ranked “very_high” → weight 10).

This is a **deliberate product choice**, not an accidental default.

## Weight tiers (current)

| Tier | Weight | Examples |
|------|--------|----------|
| Very high (clinical drivers) | **8–10** | `current_scai_12h`, `scai_prop_ge3_12h`, `med_infusion_mean_12h`, `clinical_event_n_12h`, … |
| Moderate | **2–6** | `scai_slope_12h`, `proc_n_12h`, `dx_cat_arrhythmia_arrest` |
| Demographics (included) | **1** | `age_years`, `sex_bin` |
| Administrative categoricals | **0.15–0.35** | `admit_type_cd_*`, `unit_cd_*`, `admit_src_cd_*` |

Plan file rows with `planned_xgb_feature_weight: 10.0` for included manifest columns explain why meta JSON shows round **10.0** on multiple clinical columns.

## Uniform-weight comparison

Run a no-weights retrain to confirm sensitivity:

```bash
python3 -m models.length_of_stay.train --data-dir data --model-dir data/los_model_uniform --no-feature-weights
```

See `validation/uniform_weights_comparison.json`:

| Build | Test MAE | Test R² |
|-------|----------|---------|
| Weighted (production) | 9.51 h | 0.772 |
| Uniform (`--no-feature-weights`) | 9.58 h | 0.772 |

Delta **<0.1 h MAE** — weights are optional for accuracy; keep weighted build for clinical prioritization narrative.

## What to tell Baptist

- We **prioritize physiologic early-course signals** for ICU stay length at the 12h checkpoint.
- **Race/ethnicity are not in the model**; fairness is evaluated via subgroup MAE using `person.parquet` only.
- **SHAP / permutation importance** (validation bundle) shows which columns actually move predictions on the holdout set — use that alongside weights for narrative.

## References

- `Baptist_tester/los_feature_selection_plan.py` — Spearman tiers → planned weights  
- `models/length_of_stay/features.py` — `DEFAULT_RAW_FEATURE_WEIGHTS`  
- `data/los_model/validation/los_validation_bundle.json` — `weights_sanity`, `permutation_importance`, `mean_abs_shap`
