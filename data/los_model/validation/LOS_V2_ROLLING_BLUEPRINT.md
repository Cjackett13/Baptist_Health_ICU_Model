# LOS v2 — rolling checkpoint feature blueprint

v1 (current): one row per encounter, features censored at **12h** (`ROLLING_FEATURE_MODE='v1_static_12h'`).

v2 (planned): one row per **(encounter, prediction_hour)** checkpoint (24h cadence from `los_feature_policy`).

## Grain change

| Mode | Grain | Feature suffix |
|------|--------|----------------|
| v1 | `ENCOUNTER_ID` | `_12h` (fixed) |
| v2 | `ENCOUNTER_ID` + `prediction_hour` | `_0_to_t` or `_{h}h` via `aggregate_suffix_for_prediction_hour()` |

## Target at each checkpoint

- `los_hours_total` — fixed for the stay (label, never in X).
- `remaining_los_hours` = `max(0, los_hours_total - prediction_hour)` — **must recompute** per checkpoint row.

## DuckDB sketch (`los_duckdb_features.py`)

```sql
-- Conceptual rolling snapshot (parameterize :prediction_hour and :window_end)
SELECT
    e.ENCOUNTER_ID,
    :prediction_hour AS prediction_hour,
    AVG(s.SCAI_STAGE) FILTER (WHERE s.HOUR_FROM_ADMIT <= :prediction_hour) AS scai_mean_rolling,
    MAX(s.SCAI_STAGE) FILTER (WHERE s.HOUR_FROM_ADMIT <= :prediction_hour) AS scai_last_rolling,
    COUNT(*) FILTER (
        WHERE m.ADMIN_START_DT_TM <= e.REG_DT_TM + INTERVAL ':prediction_hour hours'
    ) AS med_admin_rows_rolling
FROM encounter e
LEFT JOIN scai_stage_hourly s ON ...
LEFT JOIN medication_admin m ON ...
GROUP BY e.ENCOUNTER_ID, prediction_hour;
```

## Implementation checklist

1. Set `ROLLING_FEATURE_MODE = 'rolling'` in `los_feature_policy.py`.
2. Extend `los_duckdb_features.py` to emit checkpoint rows (not only 12h).
3. Update `los_modeling_frame.py` / `los_publish_artifacts.py` for multi-row encounters.
4. Retrain with `GroupShuffleSplit` still on `PERSON_ID` (all checkpoints for a patient share a fold).
5. Re-run `validate_artifacts`, `checklist_audit`, `subgroup_mae` (demographics joined for eval only).

## Manifest

Keep **14 clinical features**; only the **suffix/window** changes (e.g. `current_scai_24h` at hour-24 checkpoint). Update `los_model_config.MODEL_*` when v2 column names are frozen.
