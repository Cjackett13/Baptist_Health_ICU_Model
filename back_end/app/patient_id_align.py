"""Map demo seed patients to mortality/LOS parquet rows by stable clinical keys.

Display names in ``patients_seed.json`` are cosmetic. Models key on ``PERSON_ID`` and
``ENCOUNTER_ID`` (encounter_id = 9_000_000 + (person_id - 100_000) in the synthetic cohort).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.feature_store import los_features_df, mortality_features_df

SCAI_LETTERS = ("A", "B", "C", "D", "E")


def encounter_id_for_person(person_id: int) -> str:
    return str(9_000_000 + (int(person_id) - 100_000))


def person_id_for_encounter(encounter_id: str | int) -> int:
    return int(encounter_id) - 9_000_000 + 100_000


def _scai_num(stage: Any) -> float:
    if stage is None:
        return 1.0
    s = str(stage).strip().upper()
    if s in SCAI_LETTERS:
        return float(SCAI_LETTERS.index(s))
    try:
        return float(max(0, min(4, int(round(float(stage))))))
    except (TypeError, ValueError):
        return 1.0


def seed_fingerprint(patient: dict[str, Any]) -> np.ndarray:
    """Vector comparable to cohort rows (NaN = ignore in distance)."""
    age = float(patient.get("age") or 65)
    gender = str(patient.get("gender", "U")).upper()
    sex = 1.0 if gender in ("M", "MALE") else 0.0
    days = float(patient.get("days_admitted") or patient.get("time_in_hospital") or 1)
    hour = float(patient.get("hour_from_admit") or days * 24)
    scai = _scai_num(patient.get("scai_stage_current"))
    meds = float(len(patient.get("medications") or []))
    dx = float(len(patient.get("diagnoses") or []))
    vitals = {v.get("code"): float(v.get("value", np.nan)) for v in (patient.get("vitals") or [])}
    hr = vitals.get("HR", np.nan)
    mapv = vitals.get("MAP", np.nan)
    sbp = vitals.get("SBP", np.nan)
    spo2 = vitals.get("SPO2", np.nan)
    return np.array([age, sex, days, hour, scai, meds, dx, hr, mapv, sbp, spo2], dtype=float)


def cohort_fingerprint(person_id: int, mort_df: pd.DataFrame) -> np.ndarray:
    row = mort_df.loc[int(person_id)]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    age = float(row.get("age_years", row.get("age_mid", 65)))
    sex = float(row.get("sex_bin", 0))
    days = float(row.get("time_in_hospital", 1))
    scai = float(row.get("current_scai", row.get("current_scai_12h", 1)))
    meds = float(row.get("num_medications", 0))
    dx = float(row.get("number_diagnoses", 0))
    return np.array([age, sex, days, np.nan, scai, meds, dx, np.nan, np.nan, np.nan, np.nan])


# Feature weights for L2 distance (hour + vitals weighted lower when cohort side is NaN).
FINGERPRINT_WEIGHTS = np.array([3.0, 3.0, 2.0, 1.0, 4.0, 0.5, 1.0, 0.4, 0.4, 0.4, 0.4], dtype=float)


def fingerprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if not mask.any():
        return 1e9
    d = a[mask] - b[mask]
    w = FINGERPRINT_WEIGHTS[mask]
    return float(np.sqrt(((d * d) * w).sum() / w.sum()))


def _linear_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:
        # Greedy fallback when scipy is not installed.
        n_rows, n_cols = cost.shape
        used_cols: set[int] = set()
        row_ind: list[int] = []
        col_ind: list[int] = []
        for i in range(n_rows):
            best_j = None
            best_c = float("inf")
            for j in range(n_cols):
                if j in used_cols:
                    continue
                if cost[i, j] < best_c:
                    best_c = cost[i, j]
                    best_j = j
            if best_j is None:
                raise RuntimeError("assignment failed")
            used_cols.add(best_j)
            row_ind.append(i)
            col_ind.append(best_j)
        return np.array(row_ind), np.array(col_ind)


def resolve_model_ids(
    patient: dict[str, Any],
    *,
    mort_df: pd.DataFrame | None = None,
    claimed_person_ids: set[int] | None = None,
    candidate_person_ids: list[int] | None = None,
) -> tuple[int, str, str]:
    """
    Return (person_id, encounter_id, match_method).

    match_method: ``direct`` | ``assigned`` | ``nearest``
    """
    if mort_df is None:
        mort_df = mortality_features_df()
    claimed = claimed_person_ids or set()

    raw_id = patient.get("model_person_id") or patient.get("id") or patient.get("person_id")
    if raw_id is not None:
        pid = int(raw_id)
        if pid in mort_df.index and pid not in claimed:
            return pid, encounter_id_for_person(pid), "direct"

    if candidate_person_ids:
        seed_fp = seed_fingerprint(patient)
        best_pid = None
        best_dist = float("inf")
        for pid in candidate_person_ids:
            if pid in claimed:
                continue
            dist = fingerprint_distance(seed_fp, cohort_fingerprint(pid, mort_df))
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
        if best_pid is not None:
            return best_pid, encounter_id_for_person(best_pid), "assigned"

    # Last resort: nearest in full cohort (may collide if caller does not track claimed).
    seed_fp = seed_fingerprint(patient)
    best_pid = None
    best_dist = float("inf")
    for pid in mort_df.index.astype(int):
        if int(pid) in claimed:
            continue
        dist = fingerprint_distance(seed_fp, cohort_fingerprint(int(pid), mort_df))
        if dist < best_dist:
            best_dist = dist
            best_pid = int(pid)
    if best_pid is None:
        raise ValueError(f"Could not align patient {patient.get('name')!r} to model cohort")
    return best_pid, encounter_id_for_person(best_pid), "nearest"


def align_seed_patients(
    patients: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Rewrite each patient with ``model_person_id`` / ``model_encounter_id`` and sync ``id`` /
    ``encounter_id`` so inference and shock APIs use the same parquet rows.
    """
    mort_df = mortality_features_df()
    los_df = los_features_df()
    cohort_ids = sorted(int(x) for x in mort_df.index.astype(int))

    direct: list[dict[str, Any]] = []
    needs_assign: list[dict[str, Any]] = []
    for p in patients:
        raw = p.get("model_person_id") or p.get("id")
        if raw is not None and int(raw) in mort_df.index:
            direct.append(p)
        else:
            needs_assign.append(p)

    claimed: set[int] = set()
    aligned: list[dict[str, Any]] = []
    stats = {"direct": 0, "assigned": 0, "reassigned_collision": 0}

    for p in direct:
        pid, eid, _ = resolve_model_ids(p, mort_df=mort_df, claimed_person_ids=claimed)
        claimed.add(pid)
        out = _apply_ids(p, pid, eid, "direct")
        aligned.append(out)
        stats["direct"] += 1

    available = [pid for pid in cohort_ids if pid not in claimed]
    if needs_assign:
        if len(needs_assign) > len(available):
            raise ValueError(
                f"Not enough cohort patients to assign {len(needs_assign)} seed rows "
                f"({len(available)} available)"
            )
        cost = np.zeros((len(needs_assign), len(available)), dtype=float)
        for i, p in enumerate(needs_assign):
            seed_fp = seed_fingerprint(p)
            for j, pid in enumerate(available):
                cost[i, j] = fingerprint_distance(seed_fp, cohort_fingerprint(pid, mort_df))
        row_ind, col_ind = _linear_assignment(cost)
        for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            pid = available[j]
            claimed.add(pid)
            eid = encounter_id_for_person(pid)
            out = _apply_ids(needs_assign[i], pid, eid, "assigned")
            aligned.append(out)
            stats["assigned"] += 1

    aligned.sort(key=lambda x: int(x.get("rank", 0)))
    for p in aligned:
        eid = str(p["encounter_id"])
        if eid not in los_df.index:
            stats["reassigned_collision"] = stats.get("reassigned_collision", 0) + 1

    report = {
        "cohort_size": len(cohort_ids),
        "seed_count": len(patients),
        **stats,
        "unique_model_person_ids": len({int(p["model_person_id"]) for p in aligned}),
    }
    return aligned, report


def _apply_ids(patient: dict[str, Any], person_id: int, encounter_id: str, method: str) -> dict[str, Any]:
    out = dict(patient)
    prev_pid = out.get("id")
    out["model_person_id"] = str(person_id)
    out["model_encounter_id"] = encounter_id
    out["id"] = str(person_id)
    out["encounter_id"] = encounter_id
    if prev_pid and str(prev_pid) != str(person_id):
        out["seed_person_id_legacy"] = str(prev_pid)
    out["model_id_match_method"] = method
    return out


def model_person_id(patient: dict[str, Any]) -> str | None:
    raw = patient.get("model_person_id") or patient.get("id") or patient.get("person_id")
    return str(raw) if raw is not None else None


def model_encounter_id(patient: dict[str, Any]) -> str | None:
    raw = patient.get("model_encounter_id") or patient.get("encounter_id")
    return str(raw) if raw is not None else None
