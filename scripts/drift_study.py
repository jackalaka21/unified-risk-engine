"""How does performance decay after training, and does periodic retraining help?"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline.config import load_config, resolve_device, resolve_path
from pipeline.metrics import weekly_table
from pipeline.splits import DAY, period_of
from scripts.baseline import _history_matrix, fit_hgb
from scripts.train import RunSpec, predict_scores, train_variant


def retrain_cutoffs(periods, max_time, every_days):
    cuts, k = [], 1
    while periods.val_start + k * every_days * DAY < max_time:
        cuts.append(periods.val_start + k * every_days * DAY)
        k += 1
    return cuts


def combine_by_cutoff(times, cutoffs, score_sets):
    out = np.asarray(score_sets[0], dtype=float).copy()
    for k, cutoff in enumerate(cutoffs, start=1):
        out = np.where(np.asarray(times) > cutoff, score_sets[k], out)
    return out


def summarize_drift(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, schedule), g in table.groupby(["kind", "schedule"]):
        g = g.dropna(subset=["pr_auc"])
        slope = float(np.polyfit(g["week"], g["pr_auc"], 1)[0]) if len(g) >= 2 else float("nan")
        rows.append({"kind": kind, "schedule": schedule,
                     "mean_pr_auc": float(g["pr_auc"].mean()) if len(g) else float("nan"), "slope_per_week": slope})
    return pd.DataFrame(rows)


def _train_rows(prepared, cutoff):
    """Rows whose labels would be known at `cutoff`: older than the embargo, from train users only."""
    df, p = prepared.df, prepared.periods
    known = (df["TransactionDT"] <= cutoff - p.embargo).to_numpy()
    return np.flatnonzero(known & (df["user"].map(prepared.user_split) == "train").to_numpy())


TABULAR_KINDS = {"history_tabular": False, "history_stable_tabular": True}      # kind -> drop cumulative counts


def drift_study(prepared, seed=0, kinds=("history_tabular", "history_stable_tabular", "gnn_fraud_only"), device=None,
                log=print):
    cfg, df, p = prepared.cfg, prepared.df, prepared.periods
    device = resolve_device(device or cfg["train"]["device"])
    times, y = df["TransactionDT"].to_numpy(), df["isFraud"].to_numpy()
    eval_rows = np.flatnonzero(np.isin(period_of(times, p), ["val", "test"]))
    cutoffs = retrain_cutoffs(p, times.max(), cfg["drift"]["retrain_every_days"])
    n_boot = cfg["evaluation"]["bootstrap_samples"]
    out = []
    for kind in kinds:
        log(f"drift study: {kind}, {len(cutoffs)} retrain point(s)")
        scores = []
        if kind in TABULAR_KINDS:
            X = _history_matrix(prepared, stable=TABULAR_KINDS[kind])
            base_rows = np.flatnonzero(prepared.graph["transaction"].train_mask.numpy())
            scores.append(fit_hgb(X[base_rows], y[base_rows], seed).predict_proba(X[eval_rows])[:, 1])
            for c in cutoffs:
                rows = _train_rows(prepared, c)
                scores.append(fit_hgb(X[rows], y[rows], seed).predict_proba(X[eval_rows])[:, 1])
        elif kind == "gnn_fraud_only":
            epochs = cfg["drift"]["gnn_epochs"]                       # fixed: no early stopping on the evaluated weeks
            empty = np.zeros(0, dtype=int)
            base_rows = np.flatnonzero(prepared.graph["transaction"].train_mask.numpy())
            base = train_variant(prepared, RunSpec("fraud_only", seed, base_rows, empty, empty, empty, epochs=epochs), device)
            scores.append(predict_scores(prepared, base.model, base.heads, "fraud", eval_rows, device))
            for c in cutoffs:
                spec = RunSpec("fraud_only", seed, _train_rows(prepared, c), empty, empty, empty, epochs=epochs)
                m = train_variant(prepared, spec, device)
                scores.append(predict_scores(prepared, m.model, m.heads, "fraud", eval_rows, device))
        else:
            raise ValueError(f"unknown drift kind {kind!r}")
        for schedule, s in (("static", scores[0]), ("retrain", combine_by_cutoff(times[eval_rows], cutoffs, scores))):
            wk = weekly_table(times[eval_rows], y[eval_rows], s, start=p.val_start, n_boot=n_boot, seed=seed)
            wk.insert(0, "kind", kind)
            wk.insert(1, "schedule", schedule)
            out.append(wk)
    return pd.concat(out, ignore_index=True)


def main(argv=None):
    from pipeline.prepare import prepare

    ap = argparse.ArgumentParser(description="Drift and retraining study.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="plots")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    prepared = prepare(cfg)
    table = drift_study(prepared, seed=cfg["drift"]["seed"] if args.seed is None else args.seed)
    from pipeline.config import config_hash

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_drift(table)
    table.assign(config_hash=config_hash(cfg)).to_csv(out / "drift_weekly.csv", index=False)
    summary.assign(config_hash=config_hash(cfg)).to_csv(out / "drift_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
