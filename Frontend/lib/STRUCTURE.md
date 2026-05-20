# Frontend `lib/` layout

```
lib/
├── main.dart                 # App entry + MaterialApp theme
├── config/                   # API base URLs and environment
├── theme/                    # Brand colors (BhColors)
├── models/                   # Patient / prediction data types
├── services/                 # Repository, API clients, PDF export
├── screens/                  # Full-page routes (home, role select, sign-in)
├── features/
│   ├── home/                 # ICU floor UI (search, rooms, hospital picker)
│   └── patient_detail/       # Patient card + clinician/family detail sheet
└── widgets/                  # Reusable UI (prediction cards, collapsible sections)
```

## Conventions

- **screens/** — one route per file; navigated from `main` or other screens.
- **features/** — feature-specific UI that is not a top-level route.
- **widgets/** — shared components with no feature-specific business logic.
- **services/** — data loading, HTTP, PDF; no `BuildContext`.

## Assets

- `assets/patients_seed.json` — offline demo cohort (regenerate from repo-root `scripts/export_patient_seed.py`).

## API

Point `lib/config/api_config.dart` at the cardiogenic shock API (`uvicorn` on port 8000) for live MCS/ECMO scores.
