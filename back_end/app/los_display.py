"""Convert total LOS model output to clinician-facing remaining stay at ICU hour."""

from __future__ import annotations


def los_remaining_from_total(
    hospital_los_hours_total: float,
    icu_los_hours_total: float,
    hour_from_admit: int,
) -> tuple[float, float, float, float]:
    """
    Return (hospital_total_h, hospital_remaining_h, icu_total_h, icu_remaining_h).

    The LOS regressor predicts **total** encounter length from ≤12h admit features.
    The app shows **remaining** stay at ``hour_from_admit`` (aligned with shock API hour).
    """
    hour = max(0, int(hour_from_admit))
    hosp_total = max(0.5, float(hospital_los_hours_total))
    icu_total = max(0.5, float(icu_los_hours_total))
    hosp_remaining = max(0.5, hosp_total - hour)
    icu_remaining = max(0.5, icu_total - hour)
    return hosp_total, hosp_remaining, icu_total, icu_remaining
