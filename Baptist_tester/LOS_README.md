# Length of stay (LOS) model — separate from mortality

All LOS scripts live under `Baptist_tester/los_*.py` and **do not modify**
`Frontend/lib/models/mortality_model.py` or mortality artifacts.

## Data bundle

Default parquet path: **`Baptist_tester/synth_cs_data/`** (8 core tables + `code_value`).

## Scripts

| Script | Purpose |
|--------|---------|
| `los_bundle_audit.py` | Table inventory, PK/FK audit, LOS_HOURS correlation heatmap |
| `los_data_prep.py` | Cleaning, dual labels, 24h checkpoint frame |
| `los_eda.py` | Plotly EDA on `los_patient_frame.parquet` |
| `los_feature_policy.py` | Forbidden fields, validation guards |

```bash
# From repo root
python3 Baptist_tester/los_bundle_audit.py
python3 Baptist_tester/los_data_prep.py --force
python3 Baptist_tester/los_eda.py --no-open
```

## Branch

Develop LOS on git branch **`LOS_model`** (not `mortality_rate_model`).
