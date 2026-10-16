"""Ranking, finance-shaped, calibration and bootstrap metrics. Scores are probabilities in [0, 1]."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

NAN = float("nan")


def _single_class(y) -> bool:
    y = np.asarray(y)
    return y.sum() == 0 or y.sum() == len(y)


def pr_auc(y, s) -> float:
    return NAN if _single_class(y) else float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return NAN if _single_class(y) else float(roc_auc_score(y, s))
