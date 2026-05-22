# LOS model — ML testing checklist audit

**Audited:** 2026-05-20T20:28:23.998633+00:00
**Summary:** {'pass': 23, 'fail': 0, 'warn': 1, 'na': 7, 'total': 31}

## Test metrics
```json
{
  "mae_hours": 9.514508033752442,
  "rmse_hours": 11.973207872848686,
  "r2_hours": 0.7718380704005663,
  "mae_days": 0.396,
  "rmse_days": 0.499,
  "mae_log1p": 0.11484389475242801,
  "baseline_mae_hours_median_predictor": 20.199,
  "mae_improvement_vs_baseline_pct": 52.9,
  "pct_within_24h": 94.7,
  "pct_within_36h": 99.6,
  "pct_within_48h": 100.0
}
```

## Checklist

| ID | Category | Item | Status | Finding |
|---:|----------|------|--------|---------|
| 1 | Data Quality | Class distribution check | **N/A** | LOS model is regression (continuous hours), not binary classification. |
| 2 | Data Quality | Class imbalance assessment | **N/A** | Not applicable to continuous LOS target. |
| 3 | Data Quality | Train/test split verification | **PASS** | Split by PERSON_ID: GroupShuffleSplit on PERSON_ID (60% train / 20% val / 20% test). Train∩test person overlap=0. |
| 4 | Data Quality | Label verification | **PASS** | Label `los_hours_total`: 0.00% NaN. Training transform `log1p_los_hours_total` = log1p(hours). Constructed from encou... |
| 5 | Feature Quality | Feature manifest vs trained model | **PASS** | Trained feature_columns_raw_included matches los_model_config.MODEL_FEATURE_COLUMNS (14). |
| 6 | Feature Quality | Feature relevance (prediction-time) | **PASS** | Model uses 14 manifest features (≤12h snapshot + admit-time categoricals). race_cd/ethnicity excluded from training (... |
| 7 | Feature Quality | Data leakage audit | **PASS** | Label columns in feature matrix: none. Preprocessor fit on train fold only (SCAI PCA disabled; residuals train-fold).... |
| 8 | Feature Quality | Feature scaling check | **PASS** | XGBoost regressor — no StandardScaler on full matrix. Train preprocessor uses train-fold median impute, winsorize, sq... |
| 9 | Feature Quality | Missing value handling | **PASS** | Median/mode imputation at train time. Current null % (included features): max=0.0%. |
| 10 | Feature Quality | Feature importance sanity check | **PASS** | SHAP/perm available (xgboost_pred_contribs). Test MAE 9.5h. Weights are intentional — see data/los_model/FEATURE_WEIG... |
| 11 | Model Evaluation | Right metric for the problem | **PASS** | Regression task uses MAE, RMSE, R² (hours) — not accuracy/AUC. Targets: MAE≤36.0h, RMSE≤48.0h, R²≥0.65. |
| 12 | Model Evaluation | AUC-ROC interpretation | **N/A** | Mortality/classification metrics — not used for LOS regression. |
| 13 | Model Evaluation | AUC-PR interpretation | **N/A** | Mortality/classification metrics — not used for LOS regression. |
| 14 | Model Evaluation | Brier score interpretation | **N/A** | Mortality/classification metrics — not used for LOS regression. |
| 15 | Model Evaluation | Cross-validation stability | **PASS** | 5-fold GroupKFold by PERSON_ID: MAE=9.7±0.3h, R²=0.749±0.007. |
| 16 | Bias and Fairness | Demographic subgroup performance | **PASS** | Test MAE 9.5h; 0 groups with MAE >5h vs overall. Demographics from person.parquet (eval only). Sample race_cd: [{'n':... |
| 17 | Bias and Fairness | Clinical subgroup performance | **PASS** | Subgroup MAE by unit_cd, primary_dx, admit_type. unit_cd rows: [{'n': 354, 'mae_hours': 9.059140049131576, 'rmse_hour... |
| 18 | Bias and Fairness | Simpson's paradox check | **PASS** | Review flagged subgroup deltas vs overall MAE before sign-off; no race/ethnicity in model X. |
| 19 | Sanity | Direction test | **PASS** | p5→p95 sweeps on median patient (5 features). Pass count 5/5. Deltas: clinical_event_n_12h: 32.58h, med_infusion_mean... |
| 20 | Sanity | Monotonicity test | **PASS** | Same 5-feature direction battery as item 19; confirms model responds logically to acuity/load. |
| 21 | Sanity | Edge case test | **PASS** | Extreme numeric 999→145.11h, 0→54.56h; no crash, finite output in (0, 10000)h. Winsorization in train preprocessor. |
| 22 | Sanity | Prediction consistency test | **PASS** | Same median template predicted 20× — identical outputs. |
| 23 | Sanity | Threshold analysis | **N/A** | Regression has no classification threshold; use MAE buckets or long-stay binary if needed. |
| 24 | Clinical Utility | Alert rate test | **N/A** | LOS regression outputs hours, not binary alerts. |
| 25 | Clinical Utility | Lead time test | **PASS** | v1 checkpoint: prediction_hour=12, features censored at 12h. Rolling mode documented in los_feature_policy (not enabl... |
| 26 | Clinical Utility | Baseline comparison | **PASS** | Test MAE 9.5h vs median-predictor baseline 20.2h (52.9% improvement). |
| 27 | Clinical Utility | Human expert comparison | **WARN** | Clinician review of high-error cases not automated. |
| 28 | Deployment | Reproducibility test | **PASS** | random_state=42, split seed=42. Regenerate: simulate → los_data_prep → tune → train. |
| 29 | Deployment | Train/serve consistency | **PASS** | Single joblib pipeline (LosTrainPreprocessor + XGBRegressor). Serve must load same bundle; no separate scaler files. |
| 30 | Deployment | Model card documentation | **PASS** | See models/length_of_stay/README.md, xgb_los_model_meta.json, performance_targets.json. |
| 31 | Deployment | Retraining plan | **PASS** | Retrain/monitoring: RETRAINING_AND_MONITORING.md. Feature weights rationale: FEATURE_WEIGHTS_RATIONALE.md. |
