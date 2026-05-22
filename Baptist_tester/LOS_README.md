# Length of stay (LOS) model — separate from mortality

**Mortality (unchanged):** `Frontend/lib/models/mortality_model.py` — predicts in-hospital death.

**LOS (this pipeline):** `models/length_of_stay/` — predicts **length of stay in hours** (regression).

Prep/EDA helpers remain in `Baptist_tester/los_*.py`; training entrypoint is `python3 -m models.length_of_stay.train`.

## Dataset B (active)

**Path:** `data/` — **5,000 patients**. All LOS prep, cleaning, EDA, and modeling use this bundle only.

Do not point LOS scripts at `Baptist_tester/synth_cs_data/` (500-patient sample); that is out of scope for this model.

Simulator (v2 — LOS coupled to acuity):  
`python3 Baptist_tester/simulate_cardiogenic_shock_data.py --n-patients 5000 --out-dir ./data --seed 42`  
Then: `python3 Baptist_tester/los_data_prep.py --data-dir data --force`  
Train: `python3 -m models.length_of_stay.tune --data-dir data --trials 50 --use-feature-weights --retrain`

## Scripts

| Script | Purpose |
|--------|---------|
| `los_bundle_audit.py` | Table inventory, PK/FK audit, LOS_HOURS correlation heatmap |
| `los_duckdb_features.py` | DuckDB ≤12h sidecar + MI/Pearson audit vs LOS |
| `los_model_config.py` | Final model features, transforms, drops |
| `los_publish_artifacts.py` | Writes feature JSON, audit, split manifest, modeling frame |
| `los_modeling_frame.py` | Encounter-grain ID / label / feature split + missingness |
| `los_baptist_style_eda.py` | EDA HTML mirroring baptist_sample_eda layout |
| `los_data_prep.py` | Cleaning, dual labels, 24h checkpoint frame |
| `los_eda.py` | Full Plotly EDA index (`eda/los_eda_index.html`) |
| `los_eda_full.py` | EDA implementation (target, features, leakage, multicollinearity) |
| `los_feature_policy.py` | Forbidden fields, validation guards |
| `los_data_cleaning.py` | Cleaning; category/`_CD` columns; rich audit totals; optional `data/cleaned/` export |
| `los_frontend_value_ranges.json` | UI hard/typical ranges (copied to `data/` on prep) |
| `los_prep_signoff.py` | Leakage checklist, target QA, writes `los_data_prep_signoff.json` |

## Data prep sign-off (v1 ≤12h snapshot)

After `los_publish_artifacts.py`, run:

```bash
python3 Baptist_tester/los_prep_signoff.py --data-dir data
```

**Status:** `data/los_data_prep_signoff.json` → `APPROVED_FOR_TRAINING` when all checks pass.

| Check | Result |
|-------|--------|
| Final model features (14) | All ≤12h (`*_12h`) or admit-time; no `_0_to_t`, labels, or full-stay names in X |
| `los_feature_columns.json` v4 | `model_feature_columns` matches `los_final_model_features.json`; `prediction_hour` / `feature_window_hours` in **dropped** only |
| `los_modeling_frame.parquet` | 5000 × 19 — IDs + 3 labels + 14 features |
| Targets | `los_hours_total`, `log1p_los_hours_total`, `remaining_los_hours` — **0% missing**; LOS ∈ (0, 336] h |

**Training:** `python3 -m models.length_of_stay.train --data-dir data` → `data/los_model/xgb_los_pipeline.joblib`

**EDA:** `python3 -m models.length_of_stay.eda --data-dir data` → `data/los_model/eda/los_eda_index.html`

```bash
# From repo root
python3 Baptist_tester/los_bundle_audit.py --data-dir data
python3 Baptist_tester/los_data_prep.py --data-dir data --force
python3 Baptist_tester/los_data_prep.py --data-dir data --write-cleaned-parquets  # optional: data/cleaned/*.parquet
python3 Baptist_tester/los_duckdb_features.py --data-dir data --force
python3 Baptist_tester/los_publish_artifacts.py --data-dir data
python3 Baptist_tester/los_eda.py --data-dir data --no-open
# opens eda/los_eda_index.html (target, correlations, leakage, outliers, …)

# Streamlit: correlation matrix + Q–Q for top |r| features
pip install -r Baptist_tester/requirements_los_eda.txt
streamlit run Baptist_tester/los_streamlit_eda.py
```

## Branch

Develop LOS on git branch **`LOS_model`** (not `mortality_rate_model`).
