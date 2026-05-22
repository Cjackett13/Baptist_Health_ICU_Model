"""
Demographic bucket definitions for mortality modeling.

``demo_profile_bucket`` encodes (age band, weight band, sex) as a single ordinal
0 … N-1 so the tree model can use one low-influence column instead of raw age/sex/weight.

Category labels shown to users are **1-indexed** (category 1 … N).
"""

from __future__ import annotations

# Age bands: [min, max) in years at admission
AGE_BAND_EDGES: list[tuple[int, int, str]] = [
    (0, 60, "age_lt_60"),
    (60, 70, "age_60_69"),
    (70, 80, "age_70_79"),
    (80, 200, "age_80_plus"),
]

# Weight bands in kg (synthetic bundle may estimate weight when not stored on person)
WEIGHT_BAND_EDGES: list[tuple[float, float, str]] = [
    (0.0, 65.0, "wt_lt_65kg"),
    (65.0, 85.0, "wt_65_85kg"),
    (85.0, 500.0, "wt_gt_85kg"),
]

SEX_LABELS: dict[int, str] = {0: "female", 1: "male"}

N_AGE_BANDS = len(AGE_BAND_EDGES)
N_WEIGHT_BANDS = len(WEIGHT_BAND_EDGES)
N_SEX = 2
N_DEMO_BUCKETS = N_AGE_BANDS * N_WEIGHT_BANDS * N_SEX

# XGBoost ``feature_weights`` ceiling for this column (MI-based weights are 1 … 9).
DEMO_PROFILE_FEATURE_WEIGHT_CAP: float = 0.001

LEGEND_JSON_NAME = "demo_profile_bucket_legend.json"


def demo_bucket_index(*, age_bin: int, weight_bin: int, sex_bin: int) -> int:
    """0-indexed bucket id."""
    return int(age_bin) * (N_WEIGHT_BANDS * N_SEX) + int(weight_bin) * N_SEX + int(sex_bin)


def demo_category_number(bucket_index: int) -> int:
    """1-indexed category for UI (category 1 … N)."""
    return int(bucket_index) + 1


def demo_bucket_label(bucket_index: int) -> str:
    """Human-readable label, e.g. ``category_7: age_60_69 + wt_65_85kg + female``."""
    b = int(bucket_index)
    if b < 0 or b >= N_DEMO_BUCKETS:
        return f"category_{b + 1}: unknown"
    sex_bin = b % N_SEX
    b //= N_SEX
    weight_bin = b % N_WEIGHT_BANDS
    age_bin = b // N_WEIGHT_BANDS
    age_l = AGE_BAND_EDGES[age_bin][2]
    wt_l = WEIGHT_BAND_EDGES[weight_bin][2]
    sex_l = SEX_LABELS.get(sex_bin, "?")
    return f"category_{demo_category_number(int(bucket_index))}: {age_l} + {wt_l} + {sex_l}"


def build_demo_profile_legend() -> dict[str, object]:
    entries = []
    for i in range(N_DEMO_BUCKETS):
        entries.append(
            {
                "demo_profile_bucket": i,
                "demo_category": demo_category_number(i),
                "label": demo_bucket_label(i),
            }
        )
    return {
        "n_buckets": N_DEMO_BUCKETS,
        "age_bands": [{"min": a, "max_exclusive": b, "code": c} for a, b, c in AGE_BAND_EDGES],
        "weight_bands_kg": [{"min": a, "max_exclusive": b, "code": c} for a, b, c in WEIGHT_BAND_EDGES],
        "sex_codes": SEX_LABELS,
        "feature_weight_cap_in_xgb": DEMO_PROFILE_FEATURE_WEIGHT_CAP,
        "buckets": entries,
    }
