"""
Baptist Health ICU — Unified Enhanced XGBoost
All 4 fixes in one model:
  Fix 1: Within-stage VIS z-score normalization (reuse existing stats)
  Fix 2: Extended 12h / 24h rolling lookback (13 base features → +130)
  Fix 3: Four SCAI trajectory features from hourly table
  Fix 4: Stage-aware sample weights (scale_pos_weight=1; weights handle imbalance)
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = Path("/Users/christiejackett/Downloads/Diabetes_dataset/cardiogenic_shock_package")
TRAIN_PATH   = BASE / "processed/train_unscaled.parquet"
TEST_PATH    = BASE / "processed/test_unscaled.parquet"
STAGE_HOURLY = BASE / "data/scai_stage_hourly.parquet"
NORM_STATS   = BASE / "artifacts/stage_d_fix_option_b/vis_normalization_stats.json"
OUT_PASS     = BASE / "artifacts/xgboost_unified_final.pkl"
OUT_FAIL     = BASE / "artifacts/unified_attempt/xgboost_unified_attempt.pkl"

LABEL        = "Y_PROGRESSION_6H"
STAGE_COL    = "SCAI_STAGE_NUM"
ENC_COL      = "ENCOUNTER_ID"

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS = {
    "overall_auroc":  0.9108,
    "overall_aupr":   0.8471,
    "overall_brier":  0.0230,
    "stage_a_auroc":  0.90,
    "stage_b_auroc":  0.90,
    "stage_c_auroc":  0.87,
    "stage_d_auroc":  0.82,
    "stage_d_sens":   0.70,
}

# ── Stage importance weights ──────────────────────────────────────────────────
STAGE_IMPORTANCE = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.5}

# ── Features to extend with 12h/24h lookback ─────────────────────────────────
LOOKBACK_BASE = [
    "VIS_mean4h", "VIS_delta4h",
    "MAP_mean4h", "MAP_delta4h",
    "LACTATE_mean4h", "LACTATE_delta4h",
    "CREATININE_mean4h", "CREATININE_delta4h",
    "CVP_mean4h", "CVP_delta4h",
    "HR_mean4h", "HR_delta4h",
    "SBP_mean4h",
]

# ── VIS features to normalise within-stage ───────────────────────────────────
VIS_FEATURES = [
    "VIS_mean4h", "VIS_delta4h", "VIS_slope4h",
    "VIS_max4h",  "VIS_min4h",
    "ENOXIMONE_DOSE", "MILRINONE_DOSE",
    "VASOPRESSIN_DOSE",
]

XGBOOST_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.5,
    reg_alpha=0.1,
    reg_lambda=2.0,
    scale_pos_weight=1,          # sample_weight handles imbalance
    tree_method="hist",
    random_state=42,
    n_estimators=2000,
    early_stopping_rounds=50,
)

DROP_COLS = [LABEL, ENC_COL, "PERSON_ID", "HOUR_FROM_ADMIT", "EVENT_DT_TM",
             "SCAI_STAGE_CD", "ADMIT_DT_TM"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
train_raw = pd.read_parquet(TRAIN_PATH)
test_raw  = pd.read_parquet(TEST_PATH)
hourly    = pd.read_parquet(STAGE_HOURLY)

print(f"  train: {train_raw.shape}  test: {test_raw.shape}")
print(f"  hourly: {hourly.shape}  columns: {list(hourly.columns)}")

# Load within-stage normalisation stats (keys are strings in JSON)
with open(NORM_STATS) as f:
    raw_stats = json.load(f)
norm_stats = {int(k): v for k, v in raw_stats.items()}
print(f"  norm_stats stages: {sorted(norm_stats.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIX 3 — TRAJECTORY FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def build_trajectory_features(df_main: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Four features derived from hourly SCAI stage trajectory.
    All look backward only (shift/expanding on sorted hours within encounter).
    """
    h = hourly_df.sort_values([ENC_COL, "HOUR_FROM_ADMIT"]).copy()

    # ── per-encounter trajectory stats ───────────────────────────────────────
    rows = []
    for enc_id, grp in h.groupby(ENC_COL):
        stages = grp["SCAI_STAGE_NUM"].values
        hours  = grp["HOUR_FROM_ADMIT"].values
        n      = len(stages)

        consec_d     = np.zeros(n, dtype=float)
        hours_in_run = np.zeros(n, dtype=float)
        prior_max    = np.zeros(n, dtype=float)
        changes_24h  = np.zeros(n, dtype=float)

        run_len   = 0
        run_start = stages[0]

        for i in range(n):
            # consecutive Stage D hours (reset when stage != 3)
            if stages[i] == 3:
                run_len += 1
            else:
                run_len = 0
            consec_d[i] = run_len

            # hours since current run started
            if i == 0:
                hours_in_run[i] = 0.0
            elif stages[i] != stages[i - 1]:
                hours_in_run[i] = 0.0
            else:
                hours_in_run[i] = hours[i] - hours[i - 1] + hours_in_run[i - 1]

            # expanding max of strictly prior hours (shift by 1)
            if i == 0:
                prior_max[i] = stages[0]
            else:
                prior_max[i] = float(np.max(stages[:i]))

            # count stage changes in prior 24h window
            if i == 0:
                changes_24h[i] = 0.0
            else:
                window_mask = (hours >= hours[i] - 24) & (hours < hours[i])
                w_stages    = stages[window_mask]
                changes_24h[i] = float(np.sum(np.diff(w_stages) != 0)) if len(w_stages) > 1 else 0.0

        for i in range(n):
            rows.append({
                ENC_COL: enc_id,
                "HOUR_FROM_ADMIT": hours[i],
                "TRAJ_consec_stage_d_hours": consec_d[i],
                "TRAJ_hours_in_current_stage": hours_in_run[i],
                "TRAJ_prior_max_stage": prior_max[i],
                "TRAJ_stage_changes_24h": changes_24h[i],
            })

    traj_df = pd.DataFrame(rows)

    merged = df_main.merge(
        traj_df,
        on=[ENC_COL, "HOUR_FROM_ADMIT"],
        how="left",
    )
    traj_cols = ["TRAJ_consec_stage_d_hours", "TRAJ_hours_in_current_stage",
                 "TRAJ_prior_max_stage", "TRAJ_stage_changes_24h"]
    merged[traj_cols] = merged[traj_cols].fillna(0.0)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIX 2 — EXTENDED ROLLING LOOKBACK (12h / 24h)
# ══════════════════════════════════════════════════════════════════════════════
def add_extended_lookback(df: pd.DataFrame, base_features: list) -> pd.DataFrame:
    """
    For each base feature, add 12h-mean and 24h-mean rolling windows.
    Windows are computed within each encounter (no cross-encounter bleed).
    Delta features measure change over the window (current - window_mean of prior rows).
    """
    out = df.copy()
    present = [f for f in base_features if f in out.columns]

    for feat in present:
        for w in (12, 24):
            col_mean  = f"{feat}_mean{w}h"
            col_delta = f"{feat}_delta{w}h"

            # rolling mean (min_periods=1 so early rows still get a value)
            out[col_mean] = (
                out.groupby(ENC_COL)[feat]
                   .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            # delta = current value minus rolling mean of prior w rows (shift 1)
            prior_mean = (
                out.groupby(ENC_COL)[feat]
                   .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            out[col_delta] = out[feat] - prior_mean
            out[col_delta] = out[col_delta].fillna(0.0)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIX 1 — WITHIN-STAGE VIS NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
MIN_STD = 1e-3

def apply_vis_normalisation(df: pd.DataFrame, stats: dict, features: list) -> pd.DataFrame:
    """
    Z-score VIS features within each SCAI stage using pre-computed stats.
    Stages not in stats dict are left unchanged.
    """
    out = df.copy()
    for stage, sdict in stats.items():
        mask = out[STAGE_COL] == stage
        if mask.sum() == 0:
            continue
        for feat in features:
            if feat not in sdict or feat not in out.columns:
                continue
            mean  = sdict[feat]["mean"]
            std   = sdict[feat]["std"]
            ok    = sdict[feat].get("normalizable", std >= MIN_STD)
            if not ok:
                continue
            out[feat] = out[feat].astype(np.float64)
            out.loc[mask, feat] = (out.loc[mask, feat] - mean) / std
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, hourly_df: pd.DataFrame,
                   norm_stats_dict: dict) -> pd.DataFrame:
    df = build_trajectory_features(df, hourly_df)
    df = add_extended_lookback(df, LOOKBACK_BASE)
    df = apply_vis_normalisation(df, norm_stats_dict, VIS_FEATURES)
    return df


print("Building train features …")
train = build_features(train_raw, hourly, norm_stats)
print("Building test features …")
test  = build_features(test_raw, hourly, norm_stats)

drop_present = [c for c in DROP_COLS if c in train.columns]
feature_cols = [c for c in train.columns if c not in drop_present]

X_all   = train[feature_cols].astype(np.float32)
y_all   = train[LABEL].values
enc_all = train[ENC_COL].values
stage_all = train[STAGE_COL].values

X_test  = test[feature_cols].astype(np.float32)
y_test  = test[LABEL].values
stage_test = test[STAGE_COL].values

print(f"Feature count: {len(feature_cols)}")
print(f"Train rows: {len(y_all)}  positives: {y_all.sum()}  ({y_all.mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PATIENT-WISE SPLIT  80/10/10  (fit / val / cal)
# ══════════════════════════════════════════════════════════════════════════════
patients = train[ENC_COL].unique()
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
fit_idx, holdout_idx = next(gss1.split(X_all, y_all, groups=enc_all))

X_fit   = X_all.iloc[fit_idx];     y_fit   = y_all[fit_idx]
stage_fit = stage_all[fit_idx]
enc_fit = enc_all[fit_idx]

X_hold  = X_all.iloc[holdout_idx]; y_hold  = y_all[holdout_idx]
enc_hold = enc_all[holdout_idx]
stage_hold = stage_all[holdout_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
val_idx_r, cal_idx_r = next(gss2.split(X_hold, y_hold, groups=enc_hold))

X_val  = X_hold.iloc[val_idx_r];  y_val  = y_hold[val_idx_r]
X_cal  = X_hold.iloc[cal_idx_r];  y_cal  = y_hold[cal_idx_r]

print(f"\nSplit: fit={len(y_fit)} val={len(y_val)} cal={len(y_cal)} test={len(y_test)}")
print(f"  fit  pos={y_fit.sum()} ({y_fit.mean()*100:.1f}%)")
print(f"  val  pos={y_val.sum()} ({y_val.mean()*100:.1f}%)")
print(f"  cal  pos={y_cal.sum()} ({y_cal.mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIX 4 — STAGE-AWARE SAMPLE WEIGHTS  (computed from fit split only)
# ══════════════════════════════════════════════════════════════════════════════
def compute_sample_weights(y: np.ndarray, stage: np.ndarray,
                           stage_importance: dict) -> np.ndarray:
    """
    weight_i = stage_importance[stage_i] × (neg_count/pos_count within stage) if y_i=1
               stage_importance[stage_i] × 1.0                                if y_i=0
    Global neg/pos ratio replaces scale_pos_weight=7.6 (which is now 1 in XGB params).
    """
    weights = np.ones(len(y), dtype=np.float64)
    for s in np.unique(stage):
        mask = stage == s
        pos  = (y[mask] == 1).sum()
        neg  = (y[mask] == 0).sum()
        ratio = (neg / pos) if pos > 0 else 1.0
        imp   = stage_importance.get(int(s), 1.0)
        pos_w = imp * ratio
        neg_w = imp * 1.0
        weights[mask & (y == 1)] = pos_w
        weights[mask & (y == 0)] = neg_w
    return weights


w_fit = compute_sample_weights(y_fit, stage_fit, STAGE_IMPORTANCE)
print(f"\nSample weight range: {w_fit.min():.2f} – {w_fit.max():.2f}")
for s in sorted(np.unique(stage_fit)):
    m = stage_fit == s
    print(f"  Stage {s}: n={m.sum()}, pos={y_fit[m].sum()}, "
          f"w_pos={w_fit[m & (y_fit==1)].mean():.2f} w_neg={w_fit[m & (y_fit==0)].mean():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAIN XGBOOST  (with retry if early stop fires < 100 trees)
# ══════════════════════════════════════════════════════════════════════════════
def train_xgb(params, X_fit, y_fit, w_fit, X_val, y_val):
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


print("\nTraining XGBoost …")
params = XGBOOST_PARAMS.copy()
model = train_xgb(params, X_fit, y_fit, w_fit, X_val, y_val)
best_iter = model.best_iteration + 1
print(f"  best_iteration={best_iter}")

if best_iter < 100:
    print(f"  Early stop at {best_iter} < 100 — retrying with lr=0.01, n_estimators=2000, early_stopping_rounds=75")
    params2 = params.copy()
    params2["learning_rate"]        = 0.01
    params2["n_estimators"]         = 2000
    params2["early_stopping_rounds"] = 75
    model = train_xgb(params2, X_fit, y_fit, w_fit, X_val, y_val)
    best_iter = model.best_iteration + 1
    print(f"  Retry best_iteration={best_iter}")

# Retrain on fit+val combined with best_iter trees (no early stopping)
print(f"Retraining on fit+val ({len(y_fit)+len(y_val)} rows) with {best_iter} trees …")
X_fitval = pd.concat([X_fit, X_val], axis=0).reset_index(drop=True)
y_fitval = np.concatenate([y_fit, y_val])
stage_fitval = np.concatenate([stage_fit, stage_all[holdout_idx][val_idx_r]])
w_fitval = compute_sample_weights(y_fitval, stage_fitval, STAGE_IMPORTANCE)

final_params = {k: v for k, v in (params2 if best_iter >= 100 or best_iter < 100 else params).items()
                if k not in ("early_stopping_rounds",)}
final_params["early_stopping_rounds"] = None
final_params["n_estimators"] = best_iter

model_final = xgb.XGBClassifier(**final_params)
model_final.fit(X_fitval, y_fitval, sample_weight=w_fitval, verbose=False)


# ══════════════════════════════════════════════════════════════════════════════
# 9. PLATT CALIBRATION  (fit on cal split)
# ══════════════════════════════════════════════════════════════════════════════
print("Fitting Platt scaler on cal split …")
raw_cal = model_final.predict_proba(X_cal)[:, 1]
platt = LogisticRegression(C=1.0, max_iter=1000)
platt.fit(raw_cal.reshape(-1, 1), y_cal)

def calibrated_proba(model, platt_scaler, X):
    raw = model.predict_proba(X)[:, 1]
    return platt_scaler.predict_proba(raw.reshape(-1, 1))[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 10. EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)

proba_test = calibrated_proba(model_final, platt, X_test)

auroc  = roc_auc_score(y_test, proba_test)
aupr   = average_precision_score(y_test, proba_test)
brier  = brier_score_loss(y_test, proba_test)

print(f"  Overall AUC-ROC : {auroc:.4f}  (target ≥{TARGETS['overall_auroc']})")
print(f"  Overall AUC-PR  : {aupr:.4f}  (target ≥{TARGETS['overall_aupr']})")
print(f"  Brier score     : {brier:.4f}  (target ≤{TARGETS['overall_brier']})")

# Per-stage AUC-ROC
stage_results = {}
for s in sorted(np.unique(stage_test)):
    mask = stage_test == s
    if y_test[mask].sum() < 3:
        print(f"  Stage {s} AUC-ROC : skipped (< 3 positives)")
        continue
    s_auroc = roc_auc_score(y_test[mask], proba_test[mask])
    stage_results[s] = s_auroc
    label = {0: "A", 1: "B", 2: "C", 3: "D"}.get(int(s), str(s))
    tgt_key = f"stage_{label.lower()}_auroc"
    tgt = TARGETS.get(tgt_key, None)
    flag = ""
    if tgt is not None:
        flag = "✅" if s_auroc >= tgt else f"❌ (target ≥{tgt})"
    print(f"  Stage {label} AUC-ROC : {s_auroc:.4f}  {flag}")

# Stage D sensitivity at 0.5 threshold
if 3 in stage_results:
    mask_d = stage_test == 3
    pred_d = (proba_test[mask_d] >= 0.5).astype(int)
    sens_d = (pred_d[y_test[mask_d] == 1] == 1).mean() if y_test[mask_d].sum() > 0 else 0.0
    flag_sens = "✅" if sens_d >= TARGETS["stage_d_sens"] else f"❌ (target ≥{TARGETS['stage_d_sens']})"
    print(f"  Stage D Sensitivity: {sens_d:.3f}  {flag_sens}")
else:
    sens_d = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# 11. CHECK ALL TARGETS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
checks = {
    "overall_auroc":  auroc  >= TARGETS["overall_auroc"],
    "overall_aupr":   aupr   >= TARGETS["overall_aupr"],
    "overall_brier":  brier  <= TARGETS["overall_brier"],
    "stage_a_auroc":  stage_results.get(0, 0) >= TARGETS["stage_a_auroc"],
    "stage_b_auroc":  stage_results.get(1, 0) >= TARGETS["stage_b_auroc"],
    "stage_c_auroc":  stage_results.get(2, 0) >= TARGETS["stage_c_auroc"],
    "stage_d_auroc":  stage_results.get(3, 0) >= TARGETS["stage_d_auroc"],
    "stage_d_sens":   sens_d >= TARGETS["stage_d_sens"],
}
all_pass = all(checks.values())
print("ALL TARGETS MET ✅" if all_pass else "SOME TARGETS FAILED ❌")
for k, v in checks.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# 12. SAVE ARTIFACT
# ══════════════════════════════════════════════════════════════════════════════
artifact = {
    "xgboost_model":        model_final,
    "platt_scaler":         platt,
    "feature_cols":         feature_cols,
    "vis_normalization_stats": norm_stats,
    "normalization_features":  VIS_FEATURES,
    "lookback_base_features":  LOOKBACK_BASE,
    "stage_importance":     STAGE_IMPORTANCE,
    "calibration_metrics": {
        "overall_auroc":  auroc,
        "overall_aupr":   aupr,
        "brier_score":    brier,
        "stage_auroc":    stage_results,
        "stage_d_sensitivity": sens_d,
        "best_iteration": best_iter,
        "n_features":     len(feature_cols),
    },
}

if all_pass:
    OUT_PASS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PASS, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved to {OUT_PASS}")
else:
    OUT_FAIL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FAIL, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nTargets not all met — saved attempt to {OUT_FAIL}")
    print("Two-model system (enhanced_option_a) remains the best configuration.")

print("\nDone.")
