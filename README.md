# Baptist Health ICU / Cardiogenic Shock Tracker

Flutter app for ICU cardiogenic shock monitoring, MCS/ECMO escalation screening, and family-friendly summaries.

## Repository layout

```
Baptist_Health_ICU_Model/
├── Frontend/          # Flutter app (see Frontend/README.md)
├── README.md
└── .gitignore
```

Machine-learning API and parquet pipelines live in the separate **BaptistHealthContributions** workspace (`cardiogenic_shock_package/`, `scripts/`).

## Frontend quick start

```bash
cd Frontend
flutter pub get
flutter run
```
