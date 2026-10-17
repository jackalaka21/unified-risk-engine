"""Train every variant and baseline, evaluate on the test period and write the result tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.history_inputs import with_history_inputs
from pipeline.metrics import (alert_budget, bootstrap_metric, calibration, expected_cost, pr_auc,
                              recall_at_fpr, roc_auc, weekly_table)
from scripts.baseline import account_baselines, final_baselines, fraud_baselines
from scripts.stacking import stacked_baselines
from scripts.train import ACCOUNT_VARIANTS, FRAUD_VARIANTS, default_spec, predict_scores, train_variant


def _fraud_arrays(prepared):
    g, df = prepared.graph, prepared.df
    va = np.flatnonzero(g["transaction"].val_mask.numpy())
    te = np.flatnonzero(g["transaction"].test_mask.numpy())
    y, amount, times = df["isFraud"].to_numpy(), df["TransactionAmt"].to_numpy(float), df["TransactionDT"].to_numpy()
    return {"y_val": y[va], "a_val": amount[va], "y_test": y[te], "a_test": amount[te], "t_test": times[te]}


def evaluate_fraud(model, seed, p_val, p_test, prepared):
    cfg = prepared.cfg
    e, a = cfg["evaluation"], _fraud_arrays(prepared)
    y_val, y_test = a["y_val"], a["y_test"]
    lo, hi, kept = bootstrap_metric(pr_auc, y_test, p_test, e["bootstrap_samples"], seed, return_kept=True)
    row = {"model": model, "seed": seed, "head": "fraud", "config_hash": config_hash(cfg),
           "n_test": int(len(y_test)), "prevalence": float(y_test.mean()),
           "pr_auc": pr_auc(y_test, p_test), "pr_auc_lo": lo, "pr_auc_hi": hi, "pr_auc_boot_kept": kept,
           "roc_auc": roc_auc(y_test, p_test)}
    if p_val is None:                       # a refit model has no clean validation scores to choose thresholds on
        row.update(recall_at_fpr=float("nan"), realised_fpr=float("nan"))
    else:
        r = recall_at_fpr(y_val, p_val, y_test, p_test, e["fpr_target"])
        row.update(recall_at_fpr=r["recall"], realised_fpr=r["realised_fpr"])
    for b in e["alert_budgets"]:
        ab = alert_budget(y_test, p_test, a["a_test"], b)
        row.update({f"recall@{b:g}": ab["recall"], f"precision@{b:g}": ab["precision"],
                    f"amount_recall@{b:g}": ab["amount_recall"]})
    if p_val is None:
        row.update(cost=float("nan"), cost_flag_none=float(a["a_test"][y_test == 1].sum()),
                   cost_flag_all=float(e["review_cost"] * (y_test == 0).sum()))
    else:
        c = expected_cost(y_val, p_val, a["a_val"], y_test, p_test, a["a_test"], e["review_cost"])
        row.update(cost=c["cost"], cost_flag_none=c["cost_flag_none"], cost_flag_all=c["cost_flag_all"])
    cal, table = calibration(y_test, p_test)
    row.update(cal)
    weekly = weekly_table(a["t_test"], y_test, p_test, start=prepared.periods.test_start)
    for frame in (weekly, table):
        frame.insert(0, "model", model)
        frame.insert(1, "seed", seed)
        frame["config_hash"] = row["config_hash"]
    return row, weekly, table


def evaluate_account(model, seed, p_val, p_test, prepared):
    acc = prepared.account
    y_test = acc.loc[(acc["split"] == "test") & acc["eligible"], "label"].to_numpy()
    return {"model": model, "seed": seed, "head": "account", "config_hash": config_hash(prepared.cfg),
            "n_test": int(len(y_test)), "prevalence": float(y_test.mean()),
            "pr_auc": pr_auc(y_test, p_test), "roc_auc": roc_auc(y_test, p_test)}


HISTORY_SUFFIX = "_history"


def evaluate_all(prepared, out_dir, variants=("multitask", "fraud_only", "account_only", "no_graph"),
                 seeds=None, device=None, log=print, fixture=False, stacking=False, tuned_params=None) -> dict:
    cfg = prepared.cfg
    seeds = cfg["train"]["seeds"] if seeds is None else seeds
    device = resolve_device(device or cfg["train"]["device"])
    fraud_rows, account_rows, weekly, cals = [], [], [], []
    cache = {}

    def history_prepared():                                                # built once, only if a *_history variant is asked for
        if "h" not in cache:
            cache["h"] = with_history_inputs(prepared)
        return cache["h"]

    def add_fraud(name, seed, p_val, p_test):
        row, wk, cal = evaluate_fraud(name, seed, p_val, p_test, prepared)
        fraud_rows.append(row), weekly.append(wk), cals.append(cal)

    users = {s: prepared.account.index[(prepared.account["split"] == s) & prepared.account["eligible"]]
             for s in ("val", "test")}
    for seed in seeds:
        log(f"baselines seed={seed}")
        for name, res in fraud_baselines(prepared, seed).items():
            add_fraud(name, seed, res["val"], res["test"])
        if tuned_params:
            log(f"tuned trees seed={seed}")
            for name, res in final_baselines(prepared, seed, tuned_params).items():
                add_fraud(name, seed, res["val"], res["test"])
        if stacking:
            log(f"stacking seed={seed}")
            for name, res in stacked_baselines(prepared, seed, device).items():
                add_fraud(name, seed, res["val"], res["test"])
        for name, res in account_baselines(prepared, seed).items():
            account_rows.append(evaluate_account(name, seed, res["val"], res["test"], prepared))
        for name in variants:
            log(f"training {name} seed={seed}")
            variant = name.removesuffix(HISTORY_SUFFIX)                  # "fraud_only_history" is fraud_only on wider inputs
            data = history_prepared() if name != variant else prepared
            spec = default_spec(data, variant, seed)
            res = train_variant(data, spec, device)
            if variant in FRAUD_VARIANTS:
                va = np.flatnonzero(data.graph["transaction"].val_mask.numpy())
                te = np.flatnonzero(data.graph["transaction"].test_mask.numpy())
                add_fraud(name, seed,
                          predict_scores(data, res.model, res.heads, "fraud", va, device, variant=variant),
                          predict_scores(data, res.model, res.heads, "fraud", te, device, variant=variant))
            if variant in ACCOUNT_VARIANTS:
                index = data.entity_index["user"]
                pos = {user: i for i, user in enumerate(index)}
                p = {s: predict_scores(data, res.model, res.heads, "account",
                                       np.array([pos[u] for u in users[s]]), device) for s in ("val", "test")}
                account_rows.append(evaluate_account(name, seed, p["val"], p["test"], prepared))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"fraud": out_dir / "results_fraud.csv", "account": out_dir / "results_account.csv",
             "weekly": out_dir / "results_weekly.csv", "calibration": out_dir / "calibration.csv"}
    pd.DataFrame(fraud_rows).assign(fixture=fixture).to_csv(paths["fraud"], index=False)
    pd.DataFrame(account_rows).assign(fixture=fixture).to_csv(paths["account"], index=False)
    pd.concat(weekly, ignore_index=True).assign(fixture=fixture).to_csv(paths["weekly"], index=False)
    pd.concat(cals, ignore_index=True).assign(fixture=fixture).to_csv(paths["calibration"], index=False)
    return paths


def main(argv=None):
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Evaluate every variant and baseline.")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="plots")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args = p.parse_args(argv)
    prepared = prepare(load_config(args.config))
    paths = evaluate_all(prepared, resolve_path(args.out), seeds=args.seeds)
    print(pd.read_csv(paths["fraud"])[["model", "seed", "pr_auc", "recall@0.01", "amount_recall@0.01"]].to_string(index=False))


if __name__ == "__main__":
    main()
