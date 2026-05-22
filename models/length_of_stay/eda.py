#!/usr/bin/env python3
"""
LOS model EDA — writes to ``<data-dir>/los_model/eda/`` (not mortality).

  python3 -m models.length_of_stay.eda --data-dir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from models.length_of_stay.config import ARTIFACT_DIR_DEFAULT, DATA_DIR_DEFAULT, REPO_ROOT, TARGET_EDA


def _resolve(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Length-of-stay model EDA")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--model-dir", type=Path, default=None, help="Default: <data-dir>/los_model")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--target", default=TARGET_EDA)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    data_dir = _resolve(args.data_dir)
    model_dir = _resolve(args.model_dir) if args.model_dir else data_dir / "los_model"

    if args.force_prep:
        if str(REPO_ROOT / "Baptist_tester") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "Baptist_tester"))
        from los_data_prep import run_prep

        run_prep(data_dir, force=True)

    if str(REPO_ROOT / "Baptist_tester") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "Baptist_tester"))

    from los_eda_full import _open_files, run_full_eda

    need = [data_dir / "los_modeling_features.parquet", data_dir / "los_encounter_labels.parquet"]
    if not all(p.is_file() for p in need):
        print("Missing tables — run: python3 Baptist_tester/los_data_prep.py --data-dir data --force")
        return 1

    result = run_full_eda(data_dir, target=args.target, artifact_dir=model_dir)
    print(f"LOS EDA — {result['audit']}")
    print(f"Index: {result['index']}")
    if not args.no_open:
        _open_files([result["index"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
