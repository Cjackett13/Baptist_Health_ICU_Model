#!/usr/bin/env python3
"""Deprecated — use ``python3 -m models.length_of_stay.eda``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.length_of_stay.eda import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
