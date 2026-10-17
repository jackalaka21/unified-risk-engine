"""Stacking: the GNN's fraud score as one extra feature for the tree models.

A training row's GNN score must come from a model that never saw that row's user, otherwise the trees would learn to trust
an in-sample (overconfident) score. So training rows get *out-of-fold* scores: the train users are split into folds, one
GNN is trained per fold on the other folds, and it scores the held-out fold. Validation and test rows are scored by a GNN
trained on all train rows, exactly like the standalone `fraud_only` model.
"""
from __future__ import annotations

import numpy as np

from scripts.baseline import _history_matrix, _tx_idx, fit_hgb
from scripts.train import RunSpec, default_spec, predict_scores, train_variant


def fold_split(users, k: int, seed: int) -> list:
    """k (fit_positions, hold_positions) pairs over the rows; every user lands in exactly one hold-out fold."""
    unique, inverse = np.unique(np.asarray(users), return_inverse=True)
    if len(unique) < k:
        raise ValueError(f"need at least {k} users to make {k} folds, got {len(unique)}")
    fold_of_user = np.empty(len(unique), dtype=int)
    fold_of_user[np.random.default_rng(seed).permutation(len(unique))] = np.arange(len(unique)) % k
    fold = fold_of_user[inverse]
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def stacked_baselines(prepared, seed, device="cpu", folds=3, max_iter=200, variant="fraud_only") -> dict:
    spec0 = default_spec(prepared, variant, seed)
    train = spec0.fraud_train_idx
    users = prepared.df["user"].to_numpy()[train]
    oof = np.empty(len(train))
    for fit_pos, hold_pos in fold_split(users, folds, seed):
        spec = RunSpec(variant, seed, train[fit_pos], spec0.fraud_val_idx, spec0.account_train_idx, spec0.account_val_idx)
        res = train_variant(prepared, spec, device)
        oof[hold_pos] = _logit(predict_scores(prepared, res.model, res.heads, "fraud", train[hold_pos], device, variant=variant))
    full = train_variant(prepared, spec0, device)
    val, test = _tx_idx(prepared, "val"), _tx_idx(prepared, "test")
    score = np.full(len(prepared.df), np.nan, dtype=np.float32)
    score[train] = oof
    for idx in (val, test):
        score[idx] = _logit(predict_scores(prepared, full.model, full.heads, "fraud", idx, device, variant=variant))
    y = prepared.df["isFraud"].to_numpy()
    out = {}
    for name, graph in (("stacked_tabular", None), ("stacked_labels_tabular", "labels")):
        X = np.hstack([_history_matrix(prepared, stable=True, graph=graph), score[:, None]])
        model = fit_hgb(X[train], y[train], seed, max_iter)
        out[name] = {"val": model.predict_proba(X[val])[:, 1], "test": model.predict_proba(X[test])[:, 1], "model": model}
    return out
