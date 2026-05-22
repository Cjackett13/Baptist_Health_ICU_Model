#!/usr/bin/env python3
"""
Run Baptist-facing LOS validation bundle:
  SHAP, subgroup MAE, residual audit, sanity tests.

  python3 -m models.length_of_stay.validate --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.length_of_stay.config import DATA_DIR_DEFAULT, REPO_ROOT
from models.length_of_stay.residual_audit import run_residual_audit
from models.length_of_stay.sanity_tests import run_sanity_tests
from models.length_of_stay.shap_audit import run_shap_audit
from models.length_of_stay.subgroup_mae import run_subgroup_mae
from models.length_of_stay.validate_artifacts import run_validate_artifacts


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="LOS clinical validation bundle")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args()
    data_dir = _resolve(args.data_dir)
    model_dir = _resolve(args.model_dir) if args.model_dir else data_dir / "los_model"
    out_dir = model_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "artifact_validation": run_validate_artifacts(data_dir, model_dir),
        "shap": run_shap_audit(data_dir, model_dir),
        "subgroup_mae": run_subgroup_mae(data_dir, model_dir),
        "infusion_residual_audit": run_residual_audit(data_dir),
        "sanity_tests": run_sanity_tests(data_dir, model_dir),
    }
    (out_dir / "los_validation_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )

    lines = [
        "# LOS validation bundle (Baptist)",
        "",
        "Artifacts: `data/los_model/validation/los_validation_bundle.json`",
        "",
        "## Artifact validation (manifest / preprocessor)",
        f"- All checks pass: {bundle['artifact_validation'].get('all_pass')}",
        "",
        "## SHAP / importance",
        f"- SHAP available: {bundle['shap'].get('shap_available')}",
        "- See `mean_abs_shap` and `permutation_importance` in JSON",
        "",
        "## Subgroup fairness (test MAE)",
        f"- Overall MAE: {bundle['subgroup_mae']['overall_test']['mae_hours']:.1f} h",
        f"- Flagged groups (MAE >5h vs overall): {len(bundle['subgroup_mae']['flagged_mae_worse_than_overall_by_5h'])}",
        "",
        "## infusion_residual_within_scai_12h",
        f"- {bundle['infusion_residual_audit'].get('leakage_assessment', '')}",
        "",
        "## Sanity",
        f"- Direction pass rate: {bundle['sanity_tests'].get('direction_pass_rate')}",
        "",
        "Retraining plan: `data/los_model/RETRAINING_AND_MONITORING.md`",
    ]
    (out_dir / "los_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: "ok" for k in bundle}, indent=2))
    print(f"Wrote {out_dir / 'los_validation_bundle.json'}")
    print(f"Wrote {out_dir / 'los_validation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
