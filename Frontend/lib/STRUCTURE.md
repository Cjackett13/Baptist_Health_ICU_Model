# Frontend `lib/` layout

```
lib/
├── main.dart                 # App entry + MaterialApp theme
├── config/                   # API base URLs and environment
├── theme/                    # Brand colors (BhColors)
├── models/                   # Patient / prediction data types
├── services/                 # Repository, API clients, patient care assistant, PDF
├── screens/                  # Full-page routes (home, role select, sign-in)
├── features/
│   ├── home/                 # ICU floor UI (search, rooms, hospital picker)
│   ├── patient_detail/       # Patient card + clinician detail sheet
│   └── patient_portal/       # Patient dashboard + care assistant chat
└── widgets/                  # Reusable UI (prediction cards, collapsible sections)
```

## Conventions

- **screens/** — one route per file; navigated from `main` or other screens.
- **features/** — feature-specific UI that is not a top-level route.
- **widgets/** — shared components with no feature-specific business logic.
- **services/** — data loading, HTTP, PDF; no `BuildContext`.

## Assets

- `assets/patients_seed.json` — 60 patients (10 per demo hospital). **Names are display-only**; models use `model_person_id` / `model_encounter_id`. Rebuild clinical + predictions from parquets: `PYTHONPATH=back_end BAPTIST_DATA_DIR=data python3 back_end/scripts/export_patients_seed_from_data.py` then restart API (`uvicorn` on 8001) or `POST /admin/reload-seed`.

## API

Point `lib/config/api_config.dart` at the cardiogenic shock API (`uvicorn` on port 8000) for live MCS/ECMO scores.
