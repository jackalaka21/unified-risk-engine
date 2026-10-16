"""Tabular baselines on the same splits: naive, history-aware, user-aggregate and trivial account scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


class _RobustHGB(HistGradientBoostingClassifier):
    """HistGradientBoostingClassifier that ignores columns that are entirely NaN in the training data.

    sklearn 1.9.1 raises while binning such a column (for example the device history when no identity file is present).
    """

    def _clean(self, X):
        X = np.array(X, dtype=np.float32, copy=True)
        X[:, self.dead_columns_] = 0.0
        return X

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        self.dead_columns_ = np.isnan(X).all(axis=0)
        return super().fit(self._clean(X), y, **kwargs)

    def predict_proba(self, X):
        return super().predict_proba(self._clean(X))


def fit_hgb(X, y, seed, max_iter=200):
    return _RobustHGB(max_iter=max_iter, random_state=seed).fit(X, y)


def _tx_idx(prepared, split):
    return np.flatnonzero(prepared.graph["transaction"][f"{split}_mask"].numpy())


# Counts over a user's whole history only ever grow, so their distribution shifts between the train and test periods
# (on IEEE-CIS the median user_n_prior is several times larger in the test period). The "stable" baseline drops them.
CUMULATIVE = ("_n_prior", "_distinct_devices", "_distinct_emails")


def stable_history_columns(columns) -> list:
    return [c for c in columns if not c.endswith(CUMULATIVE)]


def _history_matrix(prepared, stable=False):
    hist = prepared.hist[stable_history_columns(prepared.hist.columns)] if stable else prepared.hist
    return np.hstack([prepared.x, hist.to_numpy(np.float32)])


def fraud_baselines(prepared, seed, max_iter=200) -> dict:
    y = prepared.df["isFraud"].to_numpy()
    tr, va, te = _tx_idx(prepared, "train"), _tx_idx(prepared, "val"), _tx_idx(prepared, "test")
    out = {}
    for name, X in (("naive_tabular", prepared.x), ("history_tabular", _history_matrix(prepared)),
                    ("history_stable_tabular", _history_matrix(prepared, stable=True))):
        model = fit_hgb(X[tr], y[tr], seed, max_iter)
        out[name] = {"val": model.predict_proba(X[va])[:, 1], "test": model.predict_proba(X[te])[:, 1],
                     "model": model}
    return out


def user_aggregates(df: pd.DataFrame, t_train: float) -> pd.DataFrame:
    h = df[df["TransactionDT"] <= t_train]
    g = h.groupby("user")
    agg = pd.DataFrame({
        "n_tx": g.size(), "amt_mean": g["TransactionAmt"].mean(), "amt_std": g["TransactionAmt"].std(),
        "amt_max": g["TransactionAmt"].max(), "n_devices": g["device"].nunique(),
        "n_emails": g["email"].nunique(),
        "span": g["TransactionDT"].max() - g["TransactionDT"].min(),
    })
    agg["tx_per_day"] = agg["n_tx"] / np.maximum(agg["span"] / 86400.0, 1.0)
    return agg


def _eligible_users(prepared):
    acc = prepared.account
    return {s: acc.index[(acc["split"] == s) & acc["eligible"]] for s in ("train", "val", "test")}


def _mean_fraud_score(prepared, seed, max_iter):
    """Per-user mean of out-of-fold fraud scores from the history-aware model (no in-sample label leakage)."""
    df, periods = prepared.df, prepared.periods
    y = df["isFraud"].to_numpy()
    X = _history_matrix(prepared)
    train_mask = prepared.graph["transaction"].train_mask.numpy()
    in_history = (df["TransactionDT"] <= periods.t_train).to_numpy()
    scores = np.full(len(df), np.nan)
    tr = np.flatnonzero(train_mask)
    groups = df["user"].to_numpy()[tr]
    for fit_i, pred_i in GroupKFold(n_splits=5).split(tr, groups=groups):
        model = fit_hgb(X[tr[fit_i]], y[tr[fit_i]], seed, max_iter)
        scores[tr[pred_i]] = model.predict_proba(X[tr[pred_i]])[:, 1]
    final = fit_hgb(X[tr], y[tr], seed, max_iter)
    other = np.flatnonzero(in_history & ~train_mask)  # train-period rows of held-out users
    scores[other] = final.predict_proba(X[other])[:, 1]
    return pd.Series(scores[in_history]).groupby(df["user"].to_numpy()[in_history]).mean()


def account_baselines(prepared, seed, max_iter=200) -> dict:
    users = _eligible_users(prepared)
    acc = prepared.account
    agg = user_aggregates(prepared.df, prepared.periods.t_train)
    model = fit_hgb(agg.loc[users["train"]].to_numpy(), acc.loc[users["train"], "label"].to_numpy(), seed, max_iter)
    hgb = {s: model.predict_proba(agg.loc[users[s]].to_numpy())[:, 1] for s in ("val", "test")}
    mean_score = _mean_fraud_score(prepared, seed, max_iter)
    trivial = {s: mean_score.reindex(users[s]).fillna(0.0).to_numpy() for s in ("val", "test")}
    return {"hgb_user_aggregates": hgb, "mean_fraud_score": trivial}
