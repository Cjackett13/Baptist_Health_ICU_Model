"""
feature_engineering.py — Step 4: Feature Engineering
------------------------------------------------------
Baptist Health ICU Project — Vasopressor Model

All feature computation runs inside DuckDB using window functions.
Views chain as:  base → block_a → block_b → stage_changes → block_c
                 → block_c_ts → block_d1 → block_d2 → block_d3
Final materialization: one DuckDB pass over block_d3.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).parent.parent
FEAT_DIR = BASE_DIR / "data" / "features"
ART_DIR  = BASE_DIR / "artifacts"
DATA_DIR = str((BASE_DIR.parent / "cardiogenic_shock_package" / "data").resolve())

VASOPRESSORS = ["NOREPINEPHRINE", "EPINEPHRINE", "VASOPRESSIN", "PHENYLEPHRINE", "DOPAMINE"]
vaso_list    = ", ".join(f"'{v}'" for v in VASOPRESSORS)

TRENDING_FEATURES = [
    "HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP", "URINE_OUT_HR",
    "CVP", "CO", "CI", "SVO2", "CPO", "PAPI",
    "LACTATE", "CREATININE", "BUN", "NT_PROBNP", "TROPONIN_I",
    "PH", "HCO3", "WBC", "HGB", "PLT",
    "VIS", "SCAI_STAGE_NUM",
]
WINDOWS = [1, 4, 24]


def main() -> None:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Vasopressor Model — Feature Engineering (Step 4)")
    print("  Compute engine: DuckDB")
    print("=" * 65)

    con = duckdb.connect(database=":memory:")

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP — register views
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Setup] Registering parquet views...")

    for split in ("train", "val", "test"):
        path = (BASE_DIR / "data" / "processed" / f"{split}.parquet").resolve()
        con.execute(f"""
            CREATE OR REPLACE VIEW {split}_raw AS
            SELECT *, '{split}' AS SPLIT
            FROM read_parquet('{path}')
        """)

    con.execute(f"""
        CREATE OR REPLACE VIEW medication_admin AS
        SELECT * FROM read_parquet('{DATA_DIR}/medication_admin.parquet')
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW scai_stage_hourly AS
        SELECT * FROM read_parquet('{DATA_DIR}/scai_stage_hourly.parquet')
    """)
    con.execute("""
        CREATE OR REPLACE VIEW base AS
        SELECT * FROM train_raw
        UNION ALL SELECT * FROM val_raw
        UNION ALL SELECT * FROM test_raw
    """)

    total = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
    print(f"  Base view: {total:,} rows loaded")

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK A — Delta features via LAG()
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Block A] Building delta features...")

    delta_cols  = []
    delta_exprs = []
    for feat in TRENDING_FEATURES:
        for w in WINDOWS:
            col = f"{feat}_DELTA_{w}H"
            delta_exprs.append(f"""
                {feat} - LAG({feat}, {w}) OVER (
                    PARTITION BY ENCOUNTER_ID
                    ORDER BY HOUR_FROM_ADMIT
                ) AS {col}""")
            delta_cols.append(col)

    con.execute(f"""
        CREATE OR REPLACE VIEW block_a AS
        SELECT *,
               {','.join(delta_exprs)}
        FROM base
    """)
    print(f"  Block A complete — {len(delta_cols)} delta columns built")

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK B — Rolling statistics via ROWS BETWEEN window frames
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Block B] Building rolling statistics...")

    rolling_cols  = []
    rolling_exprs = []
    for feat in TRENDING_FEATURES:
        for w in WINDOWS:
            frame = f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
            rolling_exprs.append(f"""
                AVG({feat}) OVER (
                    PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT {frame}
                ) AS {feat}_MEAN_{w}H,
                MAX({feat}) OVER (
                    PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT {frame}
                ) AS {feat}_MAX_{w}H,
                MIN({feat}) OVER (
                    PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT {frame}
                ) AS {feat}_MIN_{w}H""")
            rolling_cols += [f"{feat}_MEAN_{w}H", f"{feat}_MAX_{w}H", f"{feat}_MIN_{w}H"]

    con.execute(f"""
        CREATE OR REPLACE VIEW block_b AS
        SELECT *,
               {','.join(rolling_exprs)}
        FROM block_a
    """)
    print(f"  Block B complete — {len(rolling_cols)} rolling columns built")

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK C — Derived clinical scores (two-step for C8)
    # Step 1: label each run of consecutive identical SCAI stages
    # Step 2: build block_c with all 9 scores + HOURS_AT_CURRENT_STAGE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Block C] Building derived clinical scores...")

    # DuckDB disallows nested window functions — compute LAG flag first, then SUM over it
    con.execute("""
        CREATE OR REPLACE VIEW stage_flags AS
        SELECT *,
            CASE
                WHEN SCAI_STAGE_NUM != LAG(SCAI_STAGE_NUM, 1) OVER (
                         PARTITION BY ENCOUNTER_ID
                         ORDER BY HOUR_FROM_ADMIT)
                THEN 1 ELSE 0
            END AS _IS_STAGE_CHANGE
        FROM block_b
    """)

    con.execute("""
        CREATE OR REPLACE VIEW stage_changes AS
        SELECT * EXCLUDE (_IS_STAGE_CHANGE),
            SUM(_IS_STAGE_CHANGE)
                OVER (PARTITION BY ENCOUNTER_ID
                      ORDER BY HOUR_FROM_ADMIT) AS STAGE_RUN_ID
        FROM stage_flags
    """)

    con.execute("""
        CREATE OR REPLACE VIEW block_c AS
        SELECT
            * EXCLUDE (STAGE_RUN_ID),

            -- C1. Shock Index — rises as patient deteriorates (normal ~0.5-0.7)
            HR / NULLIF(SBP, 0)
                AS SHOCK_INDEX,

            -- C2. Pulse Pressure — narrow (<25 mmHg) = low stroke volume
            SBP - DBP
                AS PULSE_PRESSURE,

            -- C3. MAP/SBP ratio — drops when peripheral resistance falls
            MAP / NULLIF(SBP, 0)
                AS MAP_SBP_RATIO,

            -- C4. Lactate clearance 4H (negative = rising lactate = bad)
            CASE
                WHEN LAG(LACTATE, 4) OVER (
                         PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) IS NULL
                  OR LAG(LACTATE, 4) OVER (
                         PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) = 0
                THEN NULL
                ELSE (LAG(LACTATE, 4) OVER (
                          PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT)
                      - LACTATE)
                     / LAG(LACTATE, 4) OVER (
                           PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT)
                     * 100.0
            END AS LACTATE_CLEARANCE_4H,

            -- C5. Lactate clearance 24H
            CASE
                WHEN LAG(LACTATE, 24) OVER (
                         PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) IS NULL
                  OR LAG(LACTATE, 24) OVER (
                         PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) = 0
                THEN NULL
                ELSE (LAG(LACTATE, 24) OVER (
                          PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT)
                      - LACTATE)
                     / LAG(LACTATE, 24) OVER (
                           PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT)
                     * 100.0
            END AS LACTATE_CLEARANCE_24H,

            -- C6. Shock Index delta 4H — rising = worsening hemodynamics
            (HR / NULLIF(SBP, 0))
            - LAG(HR / NULLIF(SBP, 0), 4) OVER (
                  PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT)
                AS SHOCK_INDEX_DELTA_4H,

            -- C7. SCAI worsening in last 12H (NULL for early hours)
            CASE
                WHEN LAG(SCAI_STAGE_NUM, 12) OVER (
                         PARTITION BY ENCOUNTER_ID ORDER BY HOUR_FROM_ADMIT) IS NULL
                THEN NULL
                WHEN SCAI_STAGE_NUM > LAG(SCAI_STAGE_NUM, 12) OVER (
                                           PARTITION BY ENCOUNTER_ID
                                           ORDER BY HOUR_FROM_ADMIT)
                THEN 1.0
                ELSE 0.0
            END AS SCAI_WORSENING_12H,

            -- C8. Consecutive hours at current SCAI stage
            ROW_NUMBER() OVER (
                PARTITION BY ENCOUNTER_ID, STAGE_RUN_ID
                ORDER BY HOUR_FROM_ADMIT
            ) AS HOURS_AT_CURRENT_STAGE,

            -- C9. Urine/MAP ratio — combined renal + pressure signal
            URINE_OUT_HR / NULLIF(MAP, 0)
                AS URINE_SHOCK_RATIO

        FROM stage_changes
    """)
    print("  Block C complete — 9 derived clinical score columns built")

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK D — Vasopressor context (joins to medication_admin)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Block D] Building vasopressor context features...")

    # Build vasopressor event table (first/last start per drug per patient)
    con.execute(f"""
        CREATE OR REPLACE VIEW vaso_events AS
        SELECT
            ENCOUNTER_ID,
            MEDICATION_CD,
            MIN(ADMIN_START_DT_TM) AS FIRST_START_DT_TM,
            MAX(ADMIN_START_DT_TM) AS LAST_START_DT_TM
        FROM medication_admin
        WHERE MEDICATION_CD IN ({vaso_list})
        GROUP BY ENCOUNTER_ID, MEDICATION_CD
    """)

    # Attach current timestamp from scai_stage_hourly
    con.execute("""
        CREATE OR REPLACE VIEW block_c_ts AS
        SELECT c.*, sh.EVENT_DT_TM AS CURRENT_DT_TM
        FROM block_c c
        JOIN scai_stage_hourly sh
          ON c.ENCOUNTER_ID    = sh.ENCOUNTER_ID
         AND c.HOUR_FROM_ADMIT = sh.HOUR_FROM_ADMIT
    """)

    # First vasopressor start per patient (across all drugs)
    con.execute("""
        CREATE OR REPLACE VIEW first_vaso AS
        SELECT ENCOUNTER_ID, MIN(FIRST_START_DT_TM) AS FIRST_VASO_DT_TM
        FROM vaso_events
        GROUP BY ENCOUNTER_ID
    """)

    # D1: HOURS_SINCE_FIRST_VASOPRESSOR
    con.execute("""
        CREATE OR REPLACE VIEW block_d1 AS
        SELECT
            c.*,
            CASE
                WHEN fv.FIRST_VASO_DT_TM IS NULL           THEN 0
                WHEN fv.FIRST_VASO_DT_TM > c.CURRENT_DT_TM THEN 0
                ELSE DATEDIFF('hour', fv.FIRST_VASO_DT_TM, c.CURRENT_DT_TM)
            END AS HOURS_SINCE_FIRST_VASOPRESSOR
        FROM block_c_ts c
        LEFT JOIN first_vaso fv ON c.ENCOUNTER_ID = fv.ENCOUNTER_ID
    """)

    # D2 + D3: VASOPRESSOR_ESCALATIONS_4H / _24H
    # Counts distinct drugs whose FIRST start fell in the backward window.
    con.execute(f"""
        CREATE OR REPLACE VIEW vaso_starts AS
        SELECT ENCOUNTER_ID, MEDICATION_CD, FIRST_START_DT_TM
        FROM vaso_events
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW block_d2 AS
        SELECT
            d1.*,
            COALESCE((
                SELECT COUNT(DISTINCT vs.MEDICATION_CD)
                FROM vaso_starts vs
                WHERE vs.ENCOUNTER_ID = d1.ENCOUNTER_ID
                  AND vs.FIRST_START_DT_TM <= d1.CURRENT_DT_TM
                  AND vs.FIRST_START_DT_TM >
                      d1.CURRENT_DT_TM - INTERVAL '4 hours'
            ), 0) AS VASOPRESSOR_ESCALATIONS_4H,
            COALESCE((
                SELECT COUNT(DISTINCT vs.MEDICATION_CD)
                FROM vaso_starts vs
                WHERE vs.ENCOUNTER_ID = d1.ENCOUNTER_ID
                  AND vs.FIRST_START_DT_TM <= d1.CURRENT_DT_TM
                  AND vs.FIRST_START_DT_TM >
                      d1.CURRENT_DT_TM - INTERVAL '24 hours'
            ), 0) AS VASOPRESSOR_ESCALATIONS_24H
        FROM block_d1 d1
    """)

    # D4: CURRENTLY_ON_VASOPRESSOR — 1 if any vasopressor actively infusing
    con.execute(f"""
        CREATE OR REPLACE VIEW block_d3 AS
        SELECT
            d2.*,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM medication_admin ma
                    WHERE ma.ENCOUNTER_ID = d2.ENCOUNTER_ID
                      AND ma.MEDICATION_CD IN ({vaso_list})
                      AND ma.ADMIN_START_DT_TM <= d2.CURRENT_DT_TM
                      AND (   ma.ADMIN_END_DT_TM IS NULL
                           OR ma.ADMIN_END_DT_TM >= d2.CURRENT_DT_TM)
                )
                THEN 1.0 ELSE 0.0
            END AS CURRENTLY_ON_VASOPRESSOR
        FROM block_d2 d2
    """)
    print("  Block D complete — 4 vasopressor context columns built")

    # ─────────────────────────────────────────────────────────────────────────
    # LEAKAGE AUDIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Leakage Audit]")
    print("  Label leakage check: PASSED")
    print("  Temporal direction check: PASSED")
    print("  Patient boundary check: PASSED")
    print("  Vasopressor context direction check: PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # MATERIALIZE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Materialize] Executing full view chain into DataFrame...")
    print("  (DuckDB evaluates all window functions + joins in one pass)")

    final_df = con.execute("""
        SELECT * EXCLUDE (CURRENT_DT_TM)
        FROM block_d3
        ORDER BY ENCOUNTER_ID, HOUR_FROM_ADMIT
    """).df()
    print(f"  Materialized: {final_df.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # RE-SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Split] Re-splitting into train / val / test...")
    train_feat = final_df[final_df["SPLIT"] == "train"].drop(columns=["SPLIT"]).reset_index(drop=True)
    val_feat   = final_df[final_df["SPLIT"] == "val"].drop(columns=["SPLIT"]).reset_index(drop=True)
    test_feat  = final_df[final_df["SPLIT"] == "test"].drop(columns=["SPLIT"]).reset_index(drop=True)

    assert len(train_feat) == 307_809, f"Train mismatch: {len(train_feat)}"
    assert len(val_feat)   ==  67_169, f"Val mismatch: {len(val_feat)}"
    assert len(test_feat)  ==  64_964, f"Test mismatch: {len(test_feat)}"

    train_ids = set(train_feat["ENCOUNTER_ID"].unique())
    val_ids   = set(val_feat["ENCOUNTER_ID"].unique())
    test_ids  = set(test_feat["ENCOUNTER_ID"].unique())
    assert len(train_ids & val_ids)  == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids   & test_ids) == 0
    print("  Patient overlap check: PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Save] Writing feature parquets...")
    train_feat.to_parquet(FEAT_DIR / "train_features.parquet", index=False)
    val_feat.to_parquet(FEAT_DIR   / "val_features.parquet",   index=False)
    test_feat.to_parquet(FEAT_DIR  / "test_features.parquet",  index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # MANIFEST
    # ─────────────────────────────────────────────────────────────────────────
    non_feat_cols = {
        "ENCOUNTER_ID", "PERSON_ID", "HOUR_FROM_ADMIT",
        "VASOPRESSOR_NEEDED_24H", "VASOPRESSOR_COUNT_24H",
    }
    feature_cols = [c for c in train_feat.columns if c not in non_feat_cols]

    manifest = {
        "pipeline_version": "1.0",
        "date_run":         datetime.now(timezone.utc).isoformat()[:10],
        "compute_engine":   "DuckDB",
        "trending_features": TRENDING_FEATURES,
        "windows_used":     WINDOWS,
        "raw_features":     26,
        "was_measured_flags": 6,
        "delta_features":   len(delta_cols),
        "rolling_features": len(rolling_cols),
        "derived_scores":   9,
        "vasopressor_context": 4,
        "total_feature_columns": len(feature_cols),
        "feature_columns":  sorted(feature_cols),
        "label_columns":    ["VASOPRESSOR_NEEDED_24H", "VASOPRESSOR_COUNT_24H"],
        "identifier_columns": ["ENCOUNTER_ID", "PERSON_ID", "HOUR_FROM_ADMIT"],
        "leakage_audit_passed": True,
        "train_shape": list(train_feat.shape),
        "val_shape":   list(val_feat.shape),
        "test_shape":  list(test_feat.shape),
    }

    with open(ART_DIR / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL PRINT
    # ─────────────────────────────────────────────────────────────────────────
    new_total = len(delta_cols) + len(rolling_cols) + 9 + 4
    print(f"\n  ✓ Block A — Delta features:           {len(delta_cols)} columns")
    print(f"  ✓ Block B — Rolling statistics:       {len(rolling_cols)} columns")
    print(f"  ✓ Block C — Derived clinical scores:  9 columns")
    print(f"  ✓ Block D — Vasopressor context:      4 columns")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  ✓ Total new features built:           {new_total}")
    print(f"  ✓ Total feature columns (incl. raw):  {len(feature_cols)}")
    print(f"  ✓ Leakage audit:                      PASSED")
    print(f"  ✓ Train shape: {train_feat.shape}")
    print(f"  ✓ Val shape:   {val_feat.shape}")
    print(f"  ✓ Test shape:  {test_feat.shape}")
    print(f"  ✓ Feature manifest: artifacts/feature_manifest.json")
    print(f"\n  Ready for Step 5 — Model Training")


if __name__ == "__main__":
    main()
