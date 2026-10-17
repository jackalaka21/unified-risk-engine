"""A finer account identity for the tabular models, reconstructed from the D1 column.

`card1 + addr1` is a crude customer key. In IEEE-CIS the day of a transaction minus `D1` is (nearly) constant for one
account over time, so appending that start day separates accounts that share a card and address. The account always
nests inside one `user`, so the user-level train/validation/test split is never crossed. Everything here uses `D1`, which
is a raw field available when a transaction is scored, and only strictly earlier transactions; labels only after the
label-delay embargo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.graph_features import _block
from pipeline.history_features import history_features

HISTORY = ["n_1d", "n_7d", "amt_mean", "amt_std", "secs_since_prev"]         # no all-time counts: they only grow
ACCT_COLUMNS = [f"acct_{c}" for c in HISTORY] + ["acct_lbl_rate"]


def account_ids(df: pd.DataFrame) -> pd.Series:
    """user + start day; rows without D1 fall back to the user itself."""
    if "D1" not in df.columns:
        raise ValueError("the account identity needs the D1 column")
    start = np.floor(df["TransactionDT"].to_numpy(float) / 86400 - df["D1"].to_numpy(float))
    suffix = pd.Series(start, index=df.index).astype("Int64").astype("string")
    ids = (df["user"].astype("string") + "_" + suffix).where(~np.isnan(start), df["user"].astype("string"))
    return ids.astype(object)


def account_features(df: pd.DataFrame, embargo_seconds: float, prior_rate: float = 0.035, smoothing: float = 20.0) -> pd.DataFrame:
    ids = account_ids(df)
    h = history_features(df.assign(user=ids))
    out = pd.DataFrame({f"acct_{c}": h[f"user_{c}"].to_numpy() for c in HISTORY}, index=df.index)
    codes = pd.factorize(ids)[0]
    out["acct_lbl_rate"] = _block(codes, codes, df["TransactionDT"].to_numpy(np.int64), df["isFraud"].to_numpy(np.int64),
                                  embargo_seconds, prior_rate, smoothing, False)["rate"]
    return out[ACCT_COLUMNS]
