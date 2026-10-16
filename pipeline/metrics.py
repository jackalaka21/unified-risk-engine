"""Ranking, finance-shaped, calibration and bootstrap metrics. Scores are probabilities in [0, 1]."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

NAN = float("nan")


def _single_class(y) -> bool:
    y = np.asarray(y)
    return y.sum() == 0 or y.sum() == len(y)


def pr_auc(y, s) -> float:
    return NAN if _single_class(y) else float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return NAN if _single_class(y) else float(roc_auc_score(y, s))


def recall_at_fpr(y_val, s_val, y_test, s_test, fpr=0.01) -> dict:
    y_val, s_val, y_test, s_test = map(np.asarray, (y_val, s_val, y_test, s_test))
    negatives = s_val[y_val == 0]
    if len(negatives) == 0 or (y_test == 1).sum() == 0 or (y_test == 0).sum() == 0:
        return {"threshold": NAN, "recall": NAN, "realised_fpr": NAN}
    thr = float(np.quantile(negatives, 1 - fpr))
    flagged = s_test > thr
    return {"threshold": thr, "recall": float(flagged[y_test == 1].mean()),
            "realised_fpr": float(flagged[y_test == 0].mean())}


def alert_budget(y, s, amount, budget) -> dict:
    y, s, amount = np.asarray(y), np.asarray(s), np.asarray(amount, dtype=float)
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-s, kind="stable")[:k]
    pos = y == 1
    total = amount[pos].sum()
    return {"budget": budget, "k": k,
            "recall": float(y[top].sum() / pos.sum()) if pos.any() else NAN,
            "precision": float(y[top].mean()),
            "amount_recall": float(amount[top][y[top] == 1].sum() / total) if total > 0 else NAN}


def _cost(thr, y, s, amount, review_cost) -> float:
    flagged = s > thr
    return float(amount[(y == 1) & ~flagged].sum() + review_cost * ((y == 0) & flagged).sum())


def expected_cost(y_val, s_val, a_val, y_test, s_test, a_test, review_cost) -> dict:
    y_val, s_val, a_val = np.asarray(y_val), np.asarray(s_val), np.asarray(a_val, dtype=float)
    y_test, s_test, a_test = np.asarray(y_test), np.asarray(s_test), np.asarray(a_test, dtype=float)
    candidates = np.append(np.unique(np.quantile(s_val, np.linspace(0, 1, 201))), s_val.max() + 1.0)
    costs = [_cost(c, y_val, s_val, a_val, review_cost) for c in candidates]
    thr = float(candidates[int(np.argmin(costs))])
    return {"threshold": thr, "cost": _cost(thr, y_test, s_test, a_test, review_cost),
            "cost_flag_none": float(a_test[y_test == 1].sum()),
            "cost_flag_all": float(review_cost * (y_test == 0).sum())}


def calibration(y, p, bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            conf, acc, n_b = float(p[m].mean()), float(y[m].mean()), int(m.sum())
            ece += n_b / len(y) * abs(conf - acc)
            rows.append((edges[b], edges[b + 1], conf, acc, n_b))
    table = pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"])
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}, table


def bootstrap_metric(fn, y, s, n_boot=1000, seed=0, alpha=0.05):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], s[i])
        if not np.isnan(v):
            values.append(v)
    if not values:
        return NAN, NAN
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def weekly_table(times, y, s, start, n_boot=0, seed=0) -> pd.DataFrame:
    times, y, s = np.asarray(times), np.asarray(y), np.asarray(s)
    week = np.floor((times - start) / (7 * 86400)).astype(int)
    rows = []
    for w in sorted(set(week[times >= start])):
        m = (week == w) & (times >= start)
        lo = hi = NAN
        if n_boot:
            lo, hi = bootstrap_metric(pr_auc, y[m], s[m], n_boot, seed)
        rows.append({"week": int(w), "n": int(m.sum()), "n_pos": int(y[m].sum()),
                     "prevalence": float(y[m].mean()), "pr_auc": pr_auc(y[m], s[m]),
                     "pr_auc_lo": lo, "pr_auc_hi": hi})
    return pd.DataFrame(rows, columns=["week", "n", "n_pos", "prevalence", "pr_auc", "pr_auc_lo", "pr_auc_hi"])
