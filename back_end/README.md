# Baptist ICU Prediction API

FastAPI service that runs **mortality** (XGB classifier) and **LOS** (XGB regressor) on patient feature rows and feeds the Flutter app.

## Quick start

```bash
# From repo root — sync trained joblibs + meta into back_end/app/artifact/
python3 back_end/scripts/sync_all_artifacts.py

cd back_end
pip install -r requirements.txt
PYTHONPATH=. BAPTIST_DATA_DIR=../data uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## Flutter

In `Frontend/lib/config/api_config.dart`:

- `useLiveApi = true`
- `baseUrl = 'http://127.0.0.1:8001'` (iOS Simulator → host machine)

Then `PatientRepository` loads `GET /patients?live=true` with ML-refreshed `predictions`.

## Main endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/patients?live=true` | All seed patients + live mortality & LOS |
| GET | `/patients/{id}` | One patient by `id` or `encounter_id` |
| POST | `/v1/predict/patient` | Both models from `person_id` / `encounter_id` |
| POST | `/v1/predict/mortality` | Legacy 8-field mortality (what-if simulator) |
| POST | `/v1/predict/los` | LOS hours from encounter features |
| GET | `/v1/mortality/model-meta` | Alert threshold & deployment meta |

## Data dependencies

- `data/cleaned/duckdb_patient_features.parquet` — mortality features (by `PERSON_ID`)
- `data/los_modeling_features.parquet` — LOS features (by `ENCOUNTER_ID`)
- `Frontend/assets/patients_seed.json` — demo patient list (align IDs: `python3 back_end/scripts/align_patients_seed_to_models.py`, then refresh: `python3 back_end/scripts/refresh_patients_seed.py`)

Set `BAPTIST_DATA_DIR` if data live outside `../data`.

## Environment

| Variable | Default |
|----------|---------|
| `BAPTIST_DATA_DIR` | `<repo>/data` |
| `MORTALITY_MODEL_PATH` | `app/artifact/xgb_mortality_pipeline.joblib` |
| `LOS_MODEL_PATH` | `app/artifact/xgb_los_pipeline.joblib` |
| `PATIENTS_SEED_JSON` | `Frontend/assets/patients_seed.json` |
| `CORS_ALLOW_ORIGINS` | `*` |
