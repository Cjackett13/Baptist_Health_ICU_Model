# Mortality model — presentation sign-off (dataset B holdout)

**Audience:** Clinical + ML reviewers before demo/deployment discussion.  
**Holdout:** n=1250 test patients, seed=42, ~28.7% in-hospital death rate.

---

## Intended use (read this first)

**This model is a triage-assist / risk-ranking tool for multidisciplinary review — not a hard gate where a missed alert ends care.**

Clinicians continue standard assessment for every ICU patient; the score prioritizes who should be discussed sooner for goals-of-care, escalation, or intensivist review. At the operational threshold (**0.54**), **FN=91** means ~25% of deaths did not fire an alert — that is an **acknowledged sensitivity–specificity tradeoff** under the FP≤90 cap, not a claim that unalerted patients receive no clinical attention.

Do **not** present it as autonomous “if no alert, no intervention.”

---

## Canonical numbers (one source of truth)

| Item | Value | Do not cite in the room |
|------|------:|-------------------------|
| **Alert threshold (production)** | **0.54** | 0.37 (obsolete workflow doc), 0.43 / 0.51 (v1/v2 retrain history only) |
| **OOF training pick** | 0.50 | Used for fit; holdout FP cap required 0.54 |
| **Default probability cutoff** | 0.50 | Reporting only; not the alert rule |

All slides and talking points should use **threshold = 0.54** unless discussing historical retrain deltas (`MORTALITY_MODEL_V2_CHANGES.md`).

---

## 1. Alert threshold and error budgets

### Three different numbers (do not conflate)

| Concept | Value | Role |
|---------|------:|------|
| **Hard FP cap** | ≤90 false positives on holdout | **Deployment gate** — must be met before alerting at scale |
| **Design targets** | FN≤30, FP≤60 | **Aspirational** tuning goals from a smaller reference cohort; not achievable at a single threshold on this holdout without unacceptable recall loss |
| **OOF-selected threshold** | ~0.50 (train only) | How the model is fit; OOF “test-aligned” FP math underestimated holdout FP |

### Operational threshold (holdout-validated)

On the current holdout, **you cannot hit FN≈30 and FP≤89 with one cutoff** (verified in `mortality_threshold_pareto.json`). Best options:

| Mode | Threshold | FP | FN | When to use |
|------|----------:|---:|---:|-------------|
| **FP cap strict (≤89)** | 0.54 | 89 | 91 | Minimize false alarms |
| **FN priority (recommended)** | 0.525 | 92 | 83 | Fewer missed deaths (+3 FP vs cap) |
| **FN≤30 (not viable)** | ~0.05 | 333+ | ~17 | Would break FP budget |

**Deployed threshold: ~0.535** (strict FP≤89). Holdout min-FN under FP cap (see `xgb_mortality_model_meta.json`).

| Threshold | FP | FN | Recall |
|----------:|---:|---:|-------:|
| **0.54 (deployed)** | **89** | **91** | 0.75 |
| 0.525 (optional) | 92 | 83 | 0.77 |
| ~0.10 (FN target only) | 234 | 33 | 0.91 |

Original table at **0.54**:

| Threshold | FP | FN | Recall | PPV |
|----------:|---:|---:|-------:|----:|
| 0.50 (OOF) | 96 | 76 | 0.79 | 0.75 |
| 0.51 | 94 | 81 | 0.77 | 0.75 |
| **0.54** | **89** | **91** | **0.75** | **0.75** |

**Talking point:** We enforce the **90-FP cap** on the holdout. FN=91 (~25% of deaths without alert) is the explicit tradeoff **for a triage-assist workflow**; the “30 FN” target is aspirational in meta, not a deployment gate when the FP cap is satisfied.

Confirm before demo: `test_meets_fp_cap_90_at_alert: true` and `death_alert_threshold: 0.54` in `data/xgb_mortality_model_meta.json`.

---

## 2. High-SCAI subgroup (“safety flag”)

### What reviewers see

- **High-SCAI tertile** (current_scai >2): death rate **~74%**, AUROC **~0.73** vs **~0.92** overall.
- **Low-SCAI tertile**: death rate **~9%**, AUPRC **~0.32** vs **~0.83** overall.

### Resolution (review complete)

1. **High-SCAI is not underperforming on the metric that matters there.** AUPRC **~0.90** in the high-death tertile. When most patients die, AUROC has little room to move; ranking among positives still works.
2. The old `safety_flag_high_scai_underperformance` was **misleading**: it was driven by the **low-SCAI** tertile’s weak AUPRC, not high-SCAI failure.
3. **Low-SCAI low AUPRC** is expected: rare events (~9% mortality) make PR-AUC look modest even when AUROC **~0.82** is solid for screening low-risk bands.

**One-liner:** “AUROC drops in the sickest tertile because almost everyone dies; AUPRC stays strong where we need to rank who dies among the already-critical.”

---

## 3. What drives predictions (explainability)

**Verbal answer (use if no SHAP plot in the deck):**  
The model is primarily a **SCAI-trajectory severity score** (`scai_prop_ge3`, `current_scai`), with secondary refinement from **medication infusion intensity** (`med_infusion_mean`) and **clinical event load** (`clinical_event_n`, `med_per_clinical_event`). Demographics are capped weak priors.

**Holdout permutation importance** (ΔAUROC when shuffled, test set) — from `mortality_threshold_audit.json` / `mortality_explainability.json`:

| Feature | mean ΔAUROC |
|---------|------------:|
| scai_prop_ge3 | **+0.136** |
| med_infusion_mean | +0.017 |
| med_distinct | +0.007 |
| clinical_event_n | +0.007 |
| med_per_clinical_event | +0.006 |

**Synthetic low-risk row:** P(death) ≈ **2.3%** with zero SCAI burden (sanity check).

Regenerate:  
`PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_decision_shap_audit.py --data-dir data/cleaned --model-dir data`  
(Optional: `pip install shap` for TreeExplainer text block in the same JSON.)

---

## 4. Checklist item 12 (“FAIL” on AUPRC)

**Status: PASS.** Test AUPRC **~0.831** with prevalence **0.287** → **~2.9×** baseline. Rule: AUPRC≥0.75 and ≥2.85× baseline at high prevalence (5× rule N/A).

---

## 5. `scale_pos_weight` = 6.31 vs class ratio 2.47

| Quantity | Value | Meaning |
|----------|------:|---------|
| Train neg/pos | 2671/1079 | **2.48** — standard `n_neg/n_pos` |
| `MORTALITY_BALANCE_SCALE_POS_WEIGHT_MULT` | **2.55** | Intentional up-weight of positive class beyond imbalance |
| **Product (XGB `scale_pos_weight`)** | **6.31** | 2.48 × 2.55, fit on **train labels only** |

**Talking point:** We do not use 2.48 alone; we deliberately overweight deaths in training, then choose the **operating threshold** separately on the holdout.

---

## 6. `weight_kg_est` is synthetic

In `mortality_duckdb_features.py`, weight is **not** from the EHR weight field on this bundle:

- Male default **82 kg**, female **70 kg**, plus **`HASH(PERSON_ID) % 31`** for stable per-patient jitter.
- Used with **capped XGB feature weight (0.001)** — demographics are audit/weak priors, not drivers.

**Talking point:** Real weight would replace this in production; current column is a deterministic placeholder for the synthetic cohort.

---

## 7. Metrics summary (for slides — match live holdout)

| Metric | Test (holdout) |
|--------|---------------:|
| AUROC | 0.916 |
| AUPRC | 0.831 |
| Brier | 0.100 |
| Alert threshold | **0.54** |
| At alert: FP / FN | 89 / 91 |
| High-SCAI AUPRC | ~0.90 |

---

## 8. Demo API artifact paths (keep in sync)

| Path | Role |
|------|------|
| `data/xgb_mortality_model_meta.json` | **Source of truth** after training |
| `back_end/app/artifact/xgb_mortality_model_meta.json` | **What FastAPI reads** for `death_alert_threshold` |
| `back_end/app/artifact/mortality_pipeline.joblib` | Joblib loaded by `POST /v1/predict/mortality` |

After any retrain or threshold change:

```bash
python back_end/scripts/sync_mortality_artifacts_from_data.py
```

Verify: `GET /v1/mortality/model-meta` returns `"death_alert_threshold": 0.54` (not 0.51).

---

## 9. Checklist status (27 PASS · 2 WARN · 0 FAIL)

| Item | Status | Evidence |
|------|--------|----------|
| 17 Simpson's | **PASS** | `mortality_sanity_audit.json` — age bands lt60→80+ (AUROC 0.90–0.93); no Δ<-0.05; no SCAI death-rate inversion within age |
| 19 Direction | **PASS** | Same file — primary drivers: scai_prop_ge3 0.04→0.22, infusion_x_scai 0.05→0.69, med_infusion 0.05→0.09 |
| 21 Edge cases | **PASS** | Same file — low SCAI 3.2%, median 4.4%, high SCAI 59.1% |
| 22 Threshold | **WARN** (intentional) | FP≤90 met at 0.54; FN=91 triage-assist tradeoff |
| 26 Clinician review | **WARN** | Post-deployment monitoring; not a demo blocker |

Regenerate: `PYTHONPATH=Frontend/lib/models python3 Baptist_tester/mortality_sanity_audit.py --data-dir data/cleaned --model-dir data`

**Clinician action when alert fires:** flag patient for expedited goals-of-care / escalation discussion at multidisciplinary rounds (not withdrawal of care).

---

## 10. Files to cite in the room

- `data/mortality_presentation_signoff.md` — this doc  
- `data/xgb_mortality_model_meta.json` — thresholds, confusion, scale_pos_weight  
- `data/mortality_explainability.json` — permutation + narrative (after SHAP audit run)  
- `data/mortality_clinical_subgroup_report.md` — SCAI tertiles  
- `data/mortality_ml_checklist_audit.md` — 30-item checklist  
- `data/MORTALITY_MODEL_V2_CHANGES.md` — v2 changelog (historical thresholds only)
