# Baptist Health ICU / Cardiogenic Shock Tracker

Flutter app for ICU cardiogenic shock monitoring, MCS/ECMO escalation screening, patient information editing, and family-friendly summaries.

## Repository layout

```
Baptist_Health_ICU_Model/
├── Frontend/          # Flutter app (see Frontend/README.md)
├── Baptist_tester/    # Data prep, EDA, mortality helpers
├── models/            # LOS training package (length_of_stay)
├── data/              # Dataset B parquet + model artifacts
├── back_end/          # FastAPI mortality inference (optional)
└── README.md
```

Machine-learning API and parquet pipelines may also live in the separate **BaptistHealthContributions** workspace (`cardiogenic_shock_package/`, `scripts/`).

## Frontend quick start

```bash
cd Frontend
flutter pub get
flutter run
```

## ICU models (two separate pipelines)

| Model | Location | Task | Metrics |
|-------|----------|------|---------|
| **Mortality** | `Frontend/lib/models/mortality_model.py` | In-hospital death (binary) | AUROC, Brier, F1, confusion matrix |
| **Length of stay** | [`models/length_of_stay/`](models/length_of_stay/README.md) | ICU stay duration (regression) | MAE, RMSE, R² in hours |

LOS prep uses `Baptist_tester/los_*.py` on `data/`; trained artifacts live in **`data/los_model/`**.

```bash
python3 Baptist_tester/los_data_prep.py --data-dir data --force
python3 -m models.length_of_stay.eda --data-dir data
python3 -m models.length_of_stay.train --data-dir data
```

## Vertex AI chat dashboard

```bash
pip install streamlit google-cloud-aiplatform
export VERTEX_PROJECT_ID=<your-gcp-project-id>
export VERTEX_LOCATION=us-central1
export VERTEX_MODEL=gemini-1.5-pro
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
streamlit run Baptist_tester/vertex_ai_dashboard.py
```
