"""
SCAI feature extractor — Christie's split models:
  scai_deterioration_model.pkl  (stages A/B/C, SCAI_STAGE_NUM 0-2, 211 features)
  scai_stage_d_model.pkl        (stage D,       SCAI_STAGE_NUM 3,   237 features)

Neither model uses VIS normalisation — raw feature values are passed straight to XGBoost.
The correct bundle is chosen at runtime based on the patient's current SCAI stage.
"""

import numpy as np
import pandas as pd


def extract(raw_patient_data: dict, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Build a feature DataFrame for the appropriate split SCAI model.

    raw_patient_data must contain:
      "df"                — pd.DataFrame, one row per scoring hour with base clinical
                            features.  Must include ENCOUNTER_ID and HOUR_FROM_ADMIT.
      "scai_stage_hourly" — pd.DataFrame, full hourly SCAI stage history for the
                            encounter (scai_stage_hourly.parquet format).

    Returns a DataFrame with exactly the columns expected by the selected bundle,
    cast to float32.
    """
    from app.models.features.scai_pipeline import build_features
    from app.models import loader

    df_main:   pd.DataFrame = raw_patient_data["df"]
    hourly_df: pd.DataFrame = raw_patient_data["scai_stage_hourly"]

    # Determine current SCAI stage to pick the right model bundle
    stage_num = 1  # default to B if unknown
    if "SCAI_STAGE_NUM" in df_main.columns and len(df_main) > 0:
        stage_num = int(df_main["SCAI_STAGE_NUM"].iloc[-1])

    bundle_key = "scai_d" if stage_num == 3 else "scai_abc"
    bundle = loader.get(bundle_key)

    # Build features — no VIS normalisation for either new model
    df = build_features(df_main, hourly_df, norm_stats_dict=None)

    nan_cols = [c for c in df.columns if "delta" in c.lower() or "slope" in c.lower()]
    df[nan_cols] = df[nan_cols].fillna(0)

    if feature_cols is None:
        feature_cols = bundle["feature_cols"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=missing)],
            axis=1,
        )

    return df[feature_cols].astype(np.float32)
