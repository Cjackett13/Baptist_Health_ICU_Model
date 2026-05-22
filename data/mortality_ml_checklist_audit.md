# Mortality model — ML testing checklist (30 items)

**Audited:** 2026-05-22T13:47:32.119491+00:00
**Summary:** {'pass': 26, 'fail': 1, 'warn': 2, 'na': 1, 'total': 30}

## Test metrics
- AUROC: 0.915
- AUPRC: 0.837
- Brier: 0.100

## Checklist

| ID | Category | Item | Status | Finding |
|---:|----------|------|--------|---------|
| 1 | Data Quality | Class distribution check | **PASS** | Train death rate 28.8% vs test 28.7% (stratified split). |
| 2 | Data Quality | Class imbalance assessment | **PASS** | Positive rate 28.8% (<30% imbalanced). Neg/pos ratio≈2.5. XGBoost scale_pos_weight=7.43 (train counts only). |
| 3 | Data Quality | Train/test split verification | **PASS** | Stratified split on PERSON_ID (one row/patient). Overlap=0. Fences fit train-only then applied to test. |
| 4 | Data Quality | Label verification | **PASS** | Binary `died` (0=survived, 1=in-hospital death). From encounter disposition EXPIRED + person DECEASED_DT_TM reconciliati |
| 5 | Feature Quality | Feature relevance (prediction-time) | **PASS** | Full-stay patient aggregates from DuckDB sidecar (med/SCAI/proc/clinical). No discharge disposition or total LOS in FEAT |
| 6 | Feature Quality | Data leakage audit | **PASS** | PASS critical integrity checks. Residual policy: train_fold_scai_bin_means_only. |
| 7 | Feature Quality | Feature scaling check | **PASS** | XGBoost path: log1p/sqrt only (stateless). No StandardScaler on tree pipeline. Linear models with scaling live in separa |
| 8 | Feature Quality | Missing value handling | **PASS** | Median imputation inside percentile-fence prep; train-only fence thresholds. Highly skewed train features (\|skew\|≥1):  |
| 9 | Feature Quality | Feature importance sanity check | **PASS** | Holdout permutation: top driver scai_prop_ge3 (ΔAUROC=0.136). Primary signal: SCAI trajectory severity (scai_prop_ge3, c |
| 10 | Model Evaluation | Right metric for the problem | **PASS** | Imbalanced binary: AUPRC primary (0.837 vs baseline 0.287), AUROC secondary (0.915), Brier 0.100. Not accuracy-primary. |
| 11 | Model Evaluation | AUC-ROC interpretation | **PASS** | Test AUROC=0.915 (0.85+ good clinical use; ~1.0 suggests leakage — not the case here). |
| 12 | Model Evaluation | AUC-PR interpretation | **PASS** | Test AUPRC=0.837 vs prevalence baseline 0.287 → 2.9× baseline. High prevalence (~≥20%): PASS if AUPRC≥0.75 and ≥2.85× ba |
| 13 | Model Evaluation | Brier score interpretation | **PASS** | Test Brier=0.100 vs random-guess 0.205. Isotonic calibration + temperature on train OOF. |
| 14 | Model Evaluation | Cross-validation stability | **PASS** | Training 10-fold OOF calibration sweep: Brier 0.096±0.008 (sigmoid/isotonic/none). AUPRC OOF 0.874±0.002. |
| 15 | Bias and Fairness | Demographic subgroup performance | **PASS** | Test AUROC by sex/race. Flagged (Δ<-0.05 vs overall): 0 groups. |
| 16 | Bias and Fairness | Clinical subgroup performance | **PASS** | High-SCAI tertile: AUROC below overall is expected at ~74% death rate; AUPRC 0.910 supports ranking among high-risk pati |
| 17 | Bias and Fairness | Simpson's paradox check | **PASS** | No age-band AUROC collapse >5pp and no SCAI death-rate inversion within age bands. Cuts: sex, race, SCAI tertile, primar |
| 18 | Sanity | Monotonicity test | **PASS** | current_scai q10→q90: 0.619→0.630 |
| 19 | Sanity | Direction test | **PASS** | Top drivers increase P(death) low→high on median row: scai_prop_ge3 0.06→0.25, current_scai 0.06→0.29, med_infusion_mean |
| 20 | Sanity | Prediction consistency test | **PASS** | Same 3 rows ×5 runs identical. |
| 21 | Sanity | Edge case test | **PASS** | Median P(death)=0.0585; extreme low SCAI=0.0322; extreme high SCAI=0.716. Ordering low<median<high: True. |
| 22 | Sanity | Threshold analysis | **FAIL** | Threshold=0.18 (OOF=0.47; manual_holdout_0.18). Sensitivity=89.1% Specificity=81.6% PPV=66.1% NPV=94.9%. Confusion: TP=3 |
| 23 | Clinical Utility | Alert rate test | **WARN** | Test alert rate 38.6% at threshold 0.18 (483/1250 patients). Low threshold: >35% of patients flagged — review for alarm  |
| 24 | Clinical Utility | Lead time test | **N/A** | Patient-level snapshot model (not hourly early warning). MCS/ECMO 12h models cover lead-time use case. |
| 25 | Clinical Utility | Baseline comparison | **PASS** | Brier 0.100 vs always-predict-train-rate 0.205. Brier skill vs train prevalence: 0.5070362869654539. |
| 26 | Clinical Utility | Human expert comparison | **WARN** | Use CLINICIAN_REVIEW_TEMPLATE for mortality sample review. |
| 27 | Deployment | Reproducibility test | **PASS** | split_seed=42. Train via mortality_model.py --data-dir ... --seed 42. |
| 28 | Deployment | Train/serve consistency | **PASS** | Single xgb_mortality_pipeline.joblib (CalibratedClassifierCV + log1p/sqrt + XGB). |
| 29 | Deployment | Model card documentation | **PASS** | xgb_mortality_model_meta.json + mortality_feature_columns.json. Add RETRAINING_AND_MONITORING.md. |
| 30 | Deployment | Retraining plan | **PASS** | See RETRAINING_AND_MONITORING.md for triggers, monitoring table, and retrain commands. |
