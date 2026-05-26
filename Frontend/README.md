# Baptist Health Cardiogenic Shock Tracker (Flutter)

ICU clinician and family-facing app for cardiogenic shock monitoring, MCS/ECMO screening scores, and care summaries.

## Run

```bash
cd Frontend
flutter pub get
flutter run
```

Optional: start the Python API (see parent repo `cardiogenic_shock_package`) so MCS/ECMO scores refresh from `/predict/cohort/batch`.

## Project layout

See [lib/STRUCTURE.md](lib/STRUCTURE.md) for the `lib/` folder organization.

## Key paths

| Path | Purpose |
|------|---------|
| `lib/main.dart` | App entry |
| `lib/config/api_config.dart` | Shock API URL |
| `lib/services/patient_repository.dart` | Seed JSON + optional API enrich |
| `assets/patients_seed.json` | Demo patient cohort |
