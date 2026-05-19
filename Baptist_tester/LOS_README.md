# Length of stay (LOS) model — separate from mortality

All LOS scripts live under `Baptist_tester/los_*.py` and **do not modify**
`Frontend/lib/models/mortality_model.py` or mortality artifacts.

## Data bundle

| Path | Patients | Use |
|------|----------|-----|
| **`data/`** (default) | **5000** | Training, EDA, cleaning |
| `Baptist_tester/synth_cs_data/` | 500 | Fast dev / smoke tests only |

Simulator default: `python simulate_cardiogenic_shock_data.py --n-patients 5000 --out-dir ./data`

## Scripts

| Script | Purpose |
|--------|---------|
| `los_bundle_audit.py` | Table inventory, PK/FK audit, LOS_HOURS correlation heatmap |
| `los_data_prep.py` | Cleaning, dual labels, 24h checkpoint frame |
| `los_eda.py` | Plotly EDA on `los_patient_frame.parquet` |
| `los_feature_policy.py` | Forbidden fields, validation guards |

```bash
# From repo root
python3 Baptist_tester/los_bundle_audit.py --data-dir data
python3 Baptist_tester/los_data_prep.py --data-dir data --force
python3 Baptist_tester/los_eda.py --data-dir data --no-open

# Quick dev (500 patients only):
# python3 Baptist_tester/los_data_prep.py --data-dir Baptist_tester/synth_cs_data --force
```

## Branch

Develop LOS on git branch **`LOS_model`** (not `mortality_rate_model`).
