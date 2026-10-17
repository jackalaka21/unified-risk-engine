"""Graph-style aggregates for the tabular models: users sharing an entity, and delayed-label fraud rates.

All features look only at strictly earlier transactions. The label-based ones go further: fraud labels arrive late, so a
transaction at time t may only use labels of transactions at or before t - embargo, exactly as a deployed system would.
Features that only ever grow (all-time counts, entity age) are avoided on purpose: their distribution drifts between the
train and test periods and hurt the tree baseline when it used them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENTITIES = ("user", "email", "device")
WINDOW = 7 * 86400
LABEL_WINDOW = 30 * 86400
LABEL_FREE = ["device_distinct_users_7d", "email_distinct_users_7d"]
LABEL_BASED = [f"{e}_lbl_{k}" for e in ENTITIES for k in ("n_known", "n_fraud", "rate")]
LABEL_WINDOWED = [f"{e}_lbl_rate_30d" for e in ENTITIES]   # the same rate over the 30 days before the embargo cut-off
LABEL_RATES = [f"{e}_lbl_rate" for e in ENTITIES]     # the counts only grow over time and hurt on validation; the rates do not


def _distinct_in_window(times, users, prior):
    """For each row of one entity (sorted by time): distinct users among strictly earlier rows within WINDOW seconds."""
    m = len(times)
    previous = np.full(m, -1)
    last = {}
    for j in range(m):
        u = users[j]
        if u >= 0:
            previous[j] = last.get(u, -1)
            last[u] = j
    counted = users >= 0
    start = np.searchsorted(times, times - WINDOW, side="left")
    out = np.zeros(m)
    for i in range(m):
        lo, hi = start[i], prior[i]
        if hi > lo:
            out[i] = np.count_nonzero(counted[lo:hi] & (previous[lo:hi] < lo))   # first occurrence of the user inside the window
    return out


def _block(codes, users, t, y, embargo, prior_rate, smoothing, distinct_users):
    n = len(t)
    out = {k: np.full(n, np.nan) for k in ("n_known", "n_fraud", "rate", "rate_30d", "distinct_users")}
    valid = np.flatnonzero(codes >= 0)
    if len(valid) == 0:
        return out
    order = valid[np.lexsort((t[valid], codes[valid]))]
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    for grp in np.split(order, cuts):
        times, fraud = t[grp], y[grp]
        prior = np.searchsorted(times, times, side="left")                  # strictly earlier rows
        known = np.searchsorted(times, times - embargo, side="right")       # rows whose label has arrived
        cum = np.concatenate([[0], np.cumsum(fraud)])
        out["n_known"][grp] = known
        out["n_fraud"][grp] = cum[known]
        out["rate"][grp] = (cum[known] + smoothing * prior_rate) / (known + smoothing)
        recent = np.searchsorted(times, times - embargo - LABEL_WINDOW, side="left")      # first row inside the 30-day window
        out["rate_30d"][grp] = (cum[known] - cum[recent] + smoothing * prior_rate) / (known - recent + smoothing)
        if distinct_users:
            out["distinct_users"][grp] = _distinct_in_window(times, users[grp], prior)
    return out


def graph_features(df: pd.DataFrame, embargo_seconds: float, prior_rate: float = 0.035, smoothing: float = 20.0) -> pd.DataFrame:
    """Columns LABEL_FREE + LABEL_BASED + LABEL_WINDOWED, indexed like `df`. Rows without an entity get NaN for that entity's columns."""
    if embargo_seconds <= 0:
        raise ValueError("the label-delay embargo must be positive: with no delay a row would see its own label")
    t = df["TransactionDT"].to_numpy(np.int64)
    y = df["isFraud"].to_numpy(np.int64)
    codes = {e: pd.factorize(df[e])[0] for e in ENTITIES}
    cols = {}
    for e in ENTITIES:
        b = _block(codes[e], codes["user"], t, y, embargo_seconds, prior_rate, smoothing, distinct_users=e != "user")
        cols[f"{e}_lbl_n_known"], cols[f"{e}_lbl_n_fraud"], cols[f"{e}_lbl_rate"] = b["n_known"], b["n_fraud"], b["rate"]
        cols[f"{e}_lbl_rate_30d"] = b["rate_30d"]
        if e != "user":
            cols[f"{e}_distinct_users_7d"] = b["distinct_users"]
    return pd.DataFrame(cols, index=df.index)[LABEL_FREE + LABEL_BASED + LABEL_WINDOWED]
