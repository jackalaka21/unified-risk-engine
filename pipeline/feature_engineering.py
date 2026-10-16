"""Per-transaction features. Every statistic is fitted on the train period only."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CAT_BASE = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
            "P_emaildomain", "R_emaildomain"]


def _cols(df: pd.DataFrame, prefix: str) -> list[str]:
    pat = re.compile(rf"^{prefix}(\d+)$")
    found = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(c))]
    return [c for _, c in sorted(found)]


SMALL_FRAME = 64


def _frequencies_small(series: pd.Series, freq: dict) -> np.ndarray:
    """Same result as `series.astype("string").fillna("nan").map(freq).fillna(0.0)`, without the pandas overhead."""
    out = np.empty(len(series))
    for i, v in enumerate(series.to_numpy(dtype=object)):
        missing = v is None or v is pd.NA or (isinstance(v, float) and v != v)
        out[i] = freq.get("nan" if missing else str(v), 0.0)
    return out


class FeatureTransformer:
    def __init__(self, max_v_columns: int = 100, v_max_missing: float = 0.5):
        self.max_v_columns = max_v_columns
        self.v_max_missing = v_max_missing

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        self.cat_cols = [c for c in CAT_BASE + _cols(df, "M") if c in df.columns]
        self.freq = {c: df[c].astype("string").fillna("nan").value_counts(normalize=True).to_dict()
                     for c in self.cat_cols}
        v_all = _cols(df, "V")
        missing = df[v_all].isna().mean() if v_all else pd.Series(dtype=float)
        keep = list(missing[missing <= self.v_max_missing].index)
        variances = df[keep].var().sort_values(ascending=False) if keep else pd.Series(dtype=float)
        self.v_cols = list(variances.index[: self.max_v_columns])
        self.num_cols = _cols(df, "C") + _cols(df, "D") + self.v_cols
        self.na_cols = [c for c in _cols(df, "D") + self.v_cols if df[c].isna().any()]
        log_amt = np.log1p(df["TransactionAmt"].astype(float))
        self.log_amt_mean = float(log_amt.mean())
        self.log_amt_std = float(log_amt.std()) or 1.0
        self.stats = {c: (float(df[c].mean()), float(df[c].std()) or 1.0) for c in self.num_cols}
        self.feature_names = (["log_amt", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
                              + [f"freq_{c}" for c in self.cat_cols] + list(self.num_cols)
                              + [f"na_{c}" for c in self.na_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        log_amt = np.log1p(df["TransactionAmt"].astype(float).to_numpy())
        cols.append((log_amt - self.log_amt_mean) / self.log_amt_std)
        hour = (df["TransactionDT"].to_numpy() // 3600) % 24
        dow = (df["TransactionDT"].to_numpy() // 86400) % 7
        cols += [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                 np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
        for c in self.cat_cols:
            if len(df) <= SMALL_FRAME:                       # single requests: plain dict look-ups beat pandas' string machinery
                cols.append(_frequencies_small(df[c], self.freq[c]))
            else:
                mapped = df[c].astype("string").fillna("nan").map(self.freq[c])
                cols.append(mapped.fillna(0.0).to_numpy(dtype=float))
        # numeric and missing-indicator blocks are built in one pass each (absent columns count as missing); this is
        # the serving hot path, where a column-by-column loop cost tens of milliseconds per request
        mean = np.array([self.stats[c][0] for c in self.num_cols], dtype=float)
        std = np.array([self.stats[c][1] for c in self.num_cols], dtype=float)
        values = df.reindex(columns=self.num_cols).to_numpy(dtype=float, na_value=np.nan)
        numeric = np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        missing = df.reindex(columns=self.na_cols).isna().to_numpy(dtype=float)
        return np.hstack([np.column_stack(cols), numeric, missing]).astype(np.float32)


def fit_on_train(df: pd.DataFrame, periods, cfg: dict) -> FeatureTransformer:
    train = df[df["TransactionDT"] <= periods.t_train]
    f = cfg["features"]
    return FeatureTransformer(f["max_v_columns"], f["v_max_missing"]).fit(train)
