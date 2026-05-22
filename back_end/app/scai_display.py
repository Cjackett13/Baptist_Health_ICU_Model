"""SCAI stage for UI at a given ICU hour (aligned with shock API ``hour_from_admit``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from app.feature_store import data_dir

SCAI_LETTERS = ("A", "B", "C", "D", "E")


@lru_cache(maxsize=1)
def _scai_hourly_df() -> pd.DataFrame:
    for sub in ("cleaned/scai_stage_hourly.parquet", "scai_stage_hourly.parquet"):
        path = data_dir() / sub
        if path.is_file():
            return pd.read_parquet(path)
    raise FileNotFoundError("scai_stage_hourly.parquet not found under data_dir")


def scai_stage_at_hour(encounter_id: str | int, hour_from_admit: int) -> tuple[str, int]:
    """Return (letter, 0–4) for the last documented stage at or before ``hour_from_admit``."""
    eid = int(encounter_id)
    hour = max(0, int(hour_from_admit))
    sub = _scai_hourly_df()
    rows = sub[(sub["ENCOUNTER_ID"] == eid) & (sub["HOUR_FROM_ADMIT"] <= hour)].sort_values(
        "HOUR_FROM_ADMIT"
    )
    if rows.empty:
        return "B", 1
    row = rows.iloc[-1]
    num = int(row["SCAI_STAGE_NUM"])
    num = max(0, min(4, num))
    letter = str(row.get("SCAI_STAGE_CD", SCAI_LETTERS[num])).upper()
    if letter not in SCAI_LETTERS:
        letter = SCAI_LETTERS[num]
    return letter, num
