"""
ICU length-of-stay (LOS) regression model — separate from mortality.

Train: ``python3 -m models.length_of_stay.train --data-dir data``
EDA:   ``python3 -m models.length_of_stay.eda --data-dir data``
"""

from models.length_of_stay.config import ARTIFACT_DIR_DEFAULT, DATA_DIR_DEFAULT

__all__ = ["ARTIFACT_DIR_DEFAULT", "DATA_DIR_DEFAULT"]
