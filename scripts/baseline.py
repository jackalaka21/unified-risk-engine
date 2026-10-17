"""Tabular baselines on the same splits: naive, history-aware, user-aggregate and trivial account scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pipeline.account_identity import ACCT_COLUMNS
from pipeline.graph_features import LABEL_FREE, LABEL_RATES, LABEL_WINDOWED
from pipeline.history_features import stable_history_columns  # noqa: F401  (also imported from here by callers and tests)


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


def _history_matrix(prepared, stable=False, graph=None):
    """Per-transaction features + history features (+ graph aggregates: None, "free" = shared users, "labels" = delayed fraud rates)."""
    hist = prepared.hist[stable_history_columns(prepared.hist.columns)] if stable else prepared.hist
    parts = [prepared.x, hist.to_numpy(np.float32)]
    if graph == "free":
        parts.append(prepared.gfeat[LABEL_FREE].to_numpy(np.float32))
    if graph in ("labels", "account", "final"):
        parts.append(prepared.gfeat[LABEL_RATES].to_numpy(np.float32))
    if graph in ("account", "final"):
        parts.append(prepared.acct[ACCT_COLUMNS].to_numpy(np.float32))
    if graph == "final":
        parts.append(prepared.gfeat[LABEL_WINDOWED].to_numpy(np.float32))
    return np.hstack(parts)


def fraud_baselines(prepared, seed, max_iter=200) -> dict:
    y = prepared.df["isFraud"].to_numpy()
    tr, va, te = _tx_idx(prepared, "train"), _tx_idx(prepared, "val"), _tx_idx(prepared, "test")
    out = {}
    for name, X in (("naive_tabular", prepared.x), ("history_tabular", _history_matrix(prepared)),
                    ("history_stable_tabular", _history_matrix(prepared, stable=True)),
                    ("graph_shared_tabular", _history_matrix(prepared, stable=True, graph="free")),
                    ("graph_labels_tabular", _history_matrix(prepared, stable=True, graph="labels")),
                    ("account_tabular", _history_matrix(prepared, stable=True, graph="account")),
                    ("final_tabular", _history_matrix(prepared, stable=True, graph="final"))):
        model = fit_hgb(X[tr], y[tr], seed, max_iter)
        out[name] = {"val": model.predict_proba(X[va])[:, 1], "test": model.predict_proba(X[te])[:, 1],
                     "model": model}
    return out


class BaggedHGB:
    """Average of several gradient-boosted trees that differ only in their random feature choices."""

    def __init__(self, params: dict, n: int = 5, max_features: float = 0.7, seed: int = 0):
        self.params, self.n, self.max_features, self.seed = dict(params), n, max_features, seed

    def fit(self, X, y):
        self.members = [_RobustHGB(**{**self.params, "max_features": self.max_features, "random_state": self.seed * 100 + i}).fit(X, y)
                        for i in range(self.n)]
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)
        return np.column_stack([1.0 - p, p])


def final_baselines(prepared, seed, params, n_bag=5) -> dict:
    """The best tree features with tuned parameters: single, bagged, and bagged then refit on train + validation.

    The refit model is the deployment setup (train on everything labelled when the test period starts). Every validation
    label is older than the embargo before the test period, so using all of it is legitimate; but its validation scores
    are in-sample, so it reports test scores only ("val" is None) and gets no validation-threshold metrics."""
    X = _history_matrix(prepared, stable=True, graph="final")
    y = prepared.df["isFraud"].to_numpy()
    tr, va, te = _tx_idx(prepared, "train"), _tx_idx(prepared, "val"), _tx_idx(prepared, "test")
    single = _RobustHGB(**{**params, "random_state": seed}).fit(X[tr], y[tr])
    bag = BaggedHGB(params, n=n_bag, seed=seed).fit(X[tr], y[tr])
    both = np.concatenate([tr, va])
    refit = BaggedHGB(params, n=n_bag, seed=seed).fit(X[both], y[both])
    return {"final_tuned_tabular": {"val": single.predict_proba(X[va])[:, 1], "test": single.predict_proba(X[te])[:, 1], "model": single},
            "final_bag_tabular": {"val": bag.predict_proba(X[va])[:, 1], "test": bag.predict_proba(X[te])[:, 1], "model": bag},
            "final_refit_tabular": {"val": None, "test": refit.predict_proba(X[te])[:, 1], "model": refit}}


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
