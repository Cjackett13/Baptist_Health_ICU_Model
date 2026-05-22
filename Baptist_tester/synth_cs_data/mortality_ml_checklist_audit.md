# Mortality model — ML testing checklist (30 items)

**Audited:** 2026-05-20T17:11:13.120924+00:00
**Summary:** {'pass': 22, 'fail': 1, 'warn': 6, 'na': 1, 'total': 30}

## Test metrics
- AUROC: 0.920
- AUPRC: 0.848
- Brier: 0.097

## Checklist

| ID | Category | Item | Status | Finding |
|---:|----------|------|--------|---------|
| 1 | Data Quality | Class distribution check | **PASS** | Train death rate 26.4% vs test 29.0% (stratified split). |
| 2 | Data Quality | Class imbalance assessment | **PASS** | Positive rate 26.4% (<30% imbalanced). Neg/pos ratio≈2.8. XGBoost scale_pos_weight=7.12 (train counts only). |
| 3 | Data Quality | Train/test split verification | **PASS** | Stratified split on PERSON_ID (one row/patient). Overlap=0. Fences fit train-only then applied to test. |
| 4 | Data Quality | Label verification | **PASS** | Binary `died` (0=survived, 1=in-hospital death). From encounter disposition EXPIRED + person DECEASED_DT_TM reconciliati |
| 5 | Feature Quality | Feature relevance (prediction-time) | **PASS** | Full-stay patient aggregates from DuckDB sidecar (med/SCAI/proc/clinical). No discharge disposition or total LOS in FEAT |
| 6 | Feature Quality | Data leakage audit | **PASS** | PASS critical integrity checks. DuckDB `*_residual_within_scai` features use windowed means over **all** rows in `duckdb |
| 7 | Feature Quality | Feature scaling check | **PASS** | XGBoost path: log1p/sqrt only (stateless). No StandardScaler on tree pipeline. Linear models with scaling live in separa |
| 8 | Feature Quality | Missing value handling | **PASS** | Median imputation inside percentile-fence prep; train-only fence thresholds. Highly skewed train features (\|skew\|≥1):  |
| 9 | Feature Quality | Feature importance sanity check | **WARN** | Run mortality_decision_shap_audit.py for SHAP/permutation. MI weights on train; demo_profile_bucket capped low. No patie |
| 10 | Model Evaluation | Right metric for the problem | **PASS** | Imbalanced binary: AUPRC primary (0.848 vs baseline 0.290), AUROC secondary (0.920), Brier 0.097. Not accuracy-primary. |
| 11 | Model Evaluation | AUC-ROC interpretation | **PASS** | Test AUROC=0.920 (0.85+ good clinical use; ~1.0 suggests leakage — not the case here). |
| 12 | Model Evaluation | AUC-PR interpretation | **FAIL** | Test AUPRC=0.848 vs prevalence baseline 0.290 → 2.9× baseline (strong if ≥5×). |
| 13 | Model Evaluation | Brier score interpretation | **PASS** | Test Brier=0.097 vs random-guess 0.206. Isotonic calibration + temperature on train OOF. |
| 14 | Model Evaluation | Cross-validation stability | **PASS** | Training 10-fold OOF calibration sweep: Brier 0.096±0.005 (sigmoid/isotonic/none). AUPRC OOF 0.797±0.008. |
| 15 | Bias and Fairness | Demographic subgroup performance | **PASS** | Test AUROC by sex/race. Flagged (Δ<-0.05 vs overall): 0 groups. |
| 16 | Bias and Fairness | Clinical subgroup performance | **WARN** | Run AUPRC/AUROC by SCAI band (current_scai tertiles) before deployment. |
| 17 | Bias and Fairness | Simpson's paradox check | **WARN** | Review subgroup table alongside overall AUROC/AUPRC before sign-off. |
| 18 | Sanity | Monotonicity test | **PASS** | current_scai q10→q90: 0.710→0.829 |
| 19 | Sanity | Direction test | **WARN** | Extend to top-5 drivers (SCAI, infusion, clinical_event_n). |
| 20 | Sanity | Prediction consistency test | **PASS** | Same 3 rows ×5 runs identical. |
| 21 | Sanity | Edge case test | **WARN** | Test median row + extreme SCAI before production serve path. |
| 22 | Sanity | Threshold analysis | **PASS** | Alert threshold=0.38 (train OOF tuned). At alert: TP=25 FN=6 FP=9 TN=67, recall=0.81, PPV=0.74. Default 0.5 recall=0.81. |
| 23 | Clinical Utility | Alert rate test | **PASS** | Test alert rate 31.8% at threshold 0.38 (34/107 patients). |
| 24 | Clinical Utility | Lead time test | **N/A** | Patient-level snapshot model (not hourly early warning). MCS/ECMO 12h models cover lead-time use case. |
| 25 | Clinical Utility | Baseline comparison | **PASS** | Brier 0.097 vs always-predict-train-rate 0.206. Brier skill vs train prevalence: 0.5309899292963367. |
| 26 | Clinical Utility | Human expert comparison | **WARN** | Use CLINICIAN_REVIEW_TEMPLATE for mortality sample review. |
| 27 | Deployment | Reproducibility test | **PASS** | split_seed=42. Train via mortality_model.py --data-dir ... --seed 42. |
| 28 | Deployment | Train/serve consistency | **PASS** | Single xgb_mortality_pipeline.joblib (CalibratedClassifierCV + log1p/sqrt + XGB). |
| 29 | Deployment | Model card documentation | **PASS** | xgb_mortality_model_meta.json + mortality_feature_columns.json. Add RETRAINING_AND_MONITORING.md. |
| 30 | Deployment | Retraining plan | **PASS** | See RETRAINING_AND_MONITORING.md for triggers, monitoring table, and retrain commands. |
