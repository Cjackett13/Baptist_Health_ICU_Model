"""Parquet feature lookup for mortality (person) and LOS (encounter)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def data_dir() -> Path:
    return Path(os.environ.get("BAPTIST_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()


@lru_cache(maxsize=1)
def mortality_features_df() -> pd.DataFrame:
    path = data_dir() / "cleaned" / "duckdb_patient_features.parquet"
    if not path.is_file():
        path = data_dir() / "duckdb_patient_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Mortality features parquet not found under {data_dir()}")
    df = pd.read_parquet(path)
    if "PERSON_ID" in df.columns:
        df = df.set_index("PERSON_ID", drop=False)
    return df


@lru_cache(maxsize=1)
def los_features_df() -> pd.DataFrame:
    path = data_dir() / "los_modeling_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"LOS features parquet not found at {path}")
    df = pd.read_parquet(path)
    df["ENCOUNTER_ID"] = df["ENCOUNTER_ID"].astype(str)
    return df.set_index("ENCOUNTER_ID", drop=False)


def lookup_mortality_features(person_id: str | int) -> dict[str, Any] | None:
    pid = int(person_id)
    df = mortality_features_df()
    if pid not in df.index:
        return None
    row = df.loc[pid]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row.to_dict()


def lookup_los_features(encounter_id: str | int) -> dict[str, Any] | None:
    eid = str(encounter_id)
    df = los_features_df()
    if eid not in df.index:
        return None
    row = df.loc[eid]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row.to_dict()
