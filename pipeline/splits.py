"""Time periods with a label-delay embargo, user splits, account labels and leakage-safe masks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86400.0


@dataclass(frozen=True)
class Periods:
    t_train: float
    val_start: float
    val_end: float
    test_start: float
    embargo: float


def compute_periods(times, train_frac, val_end_frac, embargo_days) -> Periods:
    t = pd.Series(times)
    t_train = float(t.quantile(train_frac))
    val_end = float(t.quantile(val_end_frac))
    embargo = float(embargo_days) * DAY
    if val_end <= t_train + embargo:
        raise ValueError("embargo_days is too large: the validation period would be empty")
    if float(t.max()) <= val_end + embargo:
        raise ValueError("embargo_days is too large: the test period would be empty")
    return Periods(t_train, t_train + embargo, val_end, val_end + embargo, embargo)


def period_of(times, periods: Periods) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    out = np.full(len(t), "embargo", dtype=object)
    out[t <= periods.t_train] = "train"
    out[(t > periods.val_start) & (t <= periods.val_end)] = "val"
    out[t > periods.test_start] = "test"
    return out


def split_users(users, fractions, seed) -> pd.Series:
    ids = np.array(sorted(pd.unique(pd.Series(users))))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train = int(round(fractions[0] * len(ids)))
    n_val = int(round(fractions[1] * len(ids)))
    labels = np.empty(len(ids), dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"
    return pd.Series(labels, index=ids, name="split")


def account_table(df, periods, user_split, min_transactions, fraud_rate_threshold) -> pd.DataFrame:
    """Account-level proxy label from fraud history up to `t_train` (not credit data)."""
    history = df[df["TransactionDT"] <= periods.t_train]
    agg = history.groupby("user")["isFraud"].agg(n_tx="count", fraud_rate="mean")
    table = pd.DataFrame({"split": user_split}).join(agg)
    table["n_tx"] = table["n_tx"].fillna(0).astype(int)
    table["fraud_rate"] = table["fraud_rate"].fillna(0.0)
    table["eligible"] = table["n_tx"] >= min_transactions
    table["label"] = (table["eligible"] & (table["fraud_rate"] >= fraud_rate_threshold)).astype(int)
    return table


def check_positives(account: pd.DataFrame, min_positives: int) -> None:
    pos = account[account["eligible"]].groupby("split")["label"].sum()
    for split in ("train", "val", "test"):
        n = int(pos.get(split, 0))
        if n < min_positives:
            raise ValueError(
                f"account split '{split}' has {n} positive users, fewer than the required {min_positives}; "
                "adjust account.min_transactions or account.fraud_rate_threshold (by prevalence only)"
            )


def fraud_masks(df, periods, user_split) -> dict:
    period = period_of(df["TransactionDT"].to_numpy(), periods)
    split = df["user"].map(user_split).to_numpy()
    return {"train": (period == "train") & (split == "train"),
            "val": period == "val", "test": period == "test"}
