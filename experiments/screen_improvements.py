"""Exploratory screening of ideas that might raise PR-AUC. NOT part of the tested pipeline.

Rules: only train and validation rows are ever scored; the test period is never touched. The validation period is split in
time into an earlier half A and a later half B. Anything that needs choosing (hyper-parameters, blend weight, a feature
variant) is chosen on A and then reported on B, so a lucky choice cannot flatter itself. Every idea is compared with the
current best trees (`graph_labels_tabular`) using a paired bootstrap on B. One seed, so differences of a few thousandths
are noise.

`SCREEN_ONLY=F,C` runs just those ideas (to resume after a crash); the baseline is always recomputed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import load_config
from pipeline.graph_features import LABEL_RATES, _block, graph_features
from pipeline.history_features import history_features, stable_history_columns
from pipeline.metrics import pr_auc
from pipeline.prepare import prepare
from scripts.baseline import _history_matrix, _RobustHGB, fit_hgb
from scripts.train import RunSpec, default_spec, predict_scores, train_variant

OUT = []
ONLY = set(filter(None, os.environ.get("SCREEN_ONLY", "").split(",")))


def want(part):
    return not ONLY or part in ONLY


def say(text=""):
    print(text, flush=True)
    OUT.append(text)


def main(device="cuda"):
    t0 = time.time()
    prep = prepare(load_config())
    df, g = prep.df, prep.graph
    y = df["isFraud"].to_numpy()
    times = df["TransactionDT"].to_numpy()
    tr = np.flatnonzero(g["transaction"].train_mask.numpy())
    va = np.flatnonzero(g["transaction"].val_mask.numpy())
    split_t = float(np.median(times[va]))
    in_a, in_b = times[va] <= split_t, times[va] > split_t
    embargo = prep.periods.embargo
    say("# Screening log (validation only)\n")
    say(f"Train rows {len(tr)}, validation rows {len(va)} (A: {int(in_a.sum())}, B: {int(in_b.sum())}), seed 0, embargo {embargo / 86400:.0f} days.\n")

    X0 = _history_matrix(prep, stable=True, graph="labels")
    base_scores = fit_hgb(X0[tr], y[tr], 0).predict_proba(X0[va])[:, 1]
    yb = y[va][in_b]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, in_b.sum(), in_b.sum()) for _ in range(300)]

    def pr(scores, mask):
        return pr_auc(y[va][mask], scores[mask])

    def ci(scores):
        d = [pr_auc(yb[i], scores[in_b][i]) - pr_auc(yb[i], base_scores[in_b][i]) for i in boots]
        return np.percentile(d, [2.5, 97.5])

    def row(name, scores, note="", a_score=True):
        lo, hi = ci(scores)
        a = f"{pr(scores, in_a):.3f}" if a_score else "n/a"
        say(f"| {name} | {a} | {pr(scores, in_b):.3f} | {pr(scores, in_b) - pr(base_scores, in_b):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {note} |")

    say("| idea | PR-AUC on A | PR-AUC on B | B minus baseline | paired 95% CI of that difference | note |")
    say("|---|---|---|---|---|---|")
    row("baseline: graph_labels_tabular (defaults)", base_scores)

    if want("A"):  # tune the trees: chosen on A, reported on B
        space = np.random.default_rng(1)
        trials = []
        for i in range(20):
            params = dict(learning_rate=float(np.exp(space.uniform(np.log(0.02), np.log(0.2)))),
                          max_leaf_nodes=int(space.choice([15, 31, 63, 127])), min_samples_leaf=int(space.choice([20, 50, 100, 200])),
                          l2_regularization=float(space.choice([0.0, 0.1, 1.0, 10.0])), max_features=float(space.choice([0.5, 0.8, 1.0])),
                          max_iter=600, early_stopping=True, n_iter_no_change=20, random_state=0)
            s = _RobustHGB(**params).fit(X0[tr], y[tr]).predict_proba(X0[va])[:, 1]
            trials.append((pr(s, in_a), params, s))
            say(f"<!-- tuning trial {i}: A={trials[-1][0]:.3f} {params} -->")
        best = max(trials, key=lambda t: t[0])
        row("A. tuned trees (best of 20 random trials on A)", best[2], f"learning_rate={best[1]['learning_rate']:.3f}, leaves={best[1]['max_leaf_nodes']}, "
            f"min_leaf={best[1]['min_samples_leaf']}, l2={best[1]['l2_regularization']}, max_features={best[1]['max_features']}")

    if want("B"):  # bagging
        bag = np.mean([_RobustHGB(max_iter=200, max_features=0.7, random_state=s).fit(X0[tr], y[tr]).predict_proba(X0[va])[:, 1]
                       for s in range(5)], axis=0)
        row("B. bag of 5 trees (max_features 0.7)", bag)

    if want("D"):  # refit on train + validation A, labels older than the embargo only
        rows = np.concatenate([tr, va[times[va] <= split_t - embargo]])
        refit = fit_hgb(X0[rows], y[rows], 0).predict_proba(X0[va])[:, 1]
        row("D. refit on train + validation A (up to B start minus embargo)", refit, "A is in the training data, so no A score", a_score=False)

    prior = float(y[times <= prep.periods.t_train].mean())
    if want("E"):  # more delayed-label features
        extra = {}
        for name, col in (("card1", "card1"), ("addr1", "addr1"), ("r_email", "R_emaildomain")):
            codes = pd.factorize(df[col])[0]
            extra[f"{name}_lbl_rate"] = _block(codes, codes, times.astype(np.int64), y, embargo, prior, 20.0, False)["rate"]
        window, wr = 30 * 86400, {}
        for name in ("user", "email", "device"):
            codes = pd.factorize(df[name])[0]
            out = np.full(len(df), np.nan)
            valid = np.flatnonzero(codes >= 0)
            order = valid[np.lexsort((times[valid], codes[valid]))]
            for grp in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1):
                tt, ff = times[grp], y[grp]
                known = np.searchsorted(tt, tt - embargo, side="right")
                lo = np.searchsorted(tt, tt - embargo - window, side="left")
                cum = np.concatenate([[0], np.cumsum(ff)])
                out[grp] = (cum[known] - cum[lo] + 20.0 * prior) / (known - lo + 20.0)
            wr[f"{name}_lbl_rate_30d"] = out
        variants = {"E1 + rates of card1, addr1, R_emaildomain": pd.DataFrame(extra), "E2 + 30-day windowed rates": pd.DataFrame(wr),
                    "E3 + both": pd.DataFrame({**extra, **wr})}
        scored = {}
        for k, v in variants.items():
            X = np.hstack([X0, v.to_numpy(np.float32)])
            scored[k] = fit_hgb(X[tr], y[tr], 0).predict_proba(X[va])[:, 1]
        pick = max(scored, key=lambda k: pr(scored[k], in_a))
        for name, s in scored.items():
            row(name, s, "chosen on A" if name == pick else "")

    if want("F"):  # a better account identity from D1 (start day = day of transaction minus D1)
        start = np.floor(times / 86400 - df["D1"].to_numpy(float))
        suffix = pd.Series(start).astype("Int64").astype("string")
        new_id = (df["card1"].astype("string").fillna("na") + "_" + df["addr1"].astype("string").fillna("na") + "_" + suffix).to_numpy(object)
        uid = np.where(np.isnan(start), df["user"].astype(str).to_numpy(object), new_id)
        df2 = df.assign(user=uid)
        h2 = history_features(df2)[stable_history_columns(list(prep.hist.columns))]
        g2 = graph_features(df2, embargo, prior_rate=prior)
        X_f = np.hstack([prep.x, h2.to_numpy(np.float32), g2[LABEL_RATES].to_numpy(np.float32)])
        row("F. user identity from card1+addr1+D1 start day (replaces user)", fit_hgb(X_f[tr], y[tr], 0).predict_proba(X_f[va])[:, 1],
            f"{pd.Series(uid).nunique()} users instead of {df['user'].nunique()}")
        X_g = np.hstack([X0, h2.add_prefix("uid2_").to_numpy(np.float32), g2[["user_lbl_rate"]].to_numpy(np.float32)])
        row("F+. same identity added as extra columns", fit_hgb(X_g[tr], y[tr], 0).predict_proba(X_g[va])[:, 1])

    if want("C"):  # blend with a GNN: weight chosen on A, GNN early-stopped on A only
        spec0 = default_spec(prep, "fraud_only", 0)
        spec = RunSpec("fraud_only", 0, spec0.fraud_train_idx, va[in_a], spec0.account_train_idx, spec0.account_val_idx)
        res = train_variant(prep, spec, device)
        gnn = predict_scores(prep, res.model, res.heads, "fraud", va, device, variant="fraud_only")
        rank = lambda s: pd.Series(s).rank(pct=True).to_numpy()  # noqa: E731
        say(f"\nGNN alone: A {pr(gnn, in_a):.3f}, B {pr(gnn, in_b):.3f} (early-stopped on A only, epoch {res.best_epoch})\n")
        blends = {w: (1 - w) * rank(base_scores) + w * rank(gnn) for w in (0.05, 0.1, 0.2, 0.3, 0.5)}
        wbest = max(blends, key=lambda w: pr(blends[w], in_a))
        for w, s in blends.items():
            row(f"C. rank blend, GNN weight {w}", s, "chosen on A" if w == wbest else "")
    say(f"\nTotal time {time.time() - t0:.0f} s.")
    Path(os.environ.get("SCREEN_OUT", "docs/screening.md")).write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
