"""Test PR-AUC of one checkpoint under the training neighbour rule and the serving neighbour rule."""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.metrics import pr_auc
from scripts.train import load_checkpoint, predict_scores


def sampling_rule_table(prepared, checkpoint, device="cpu") -> pd.DataFrame:
    model, heads, meta = load_checkpoint(checkpoint, prepared, device)
    test = np.flatnonzero(prepared.graph["transaction"].test_mask.numpy())
    y = prepared.df["isFraud"].to_numpy()[test]
    rows = []
    for rule in ("uniform", "last"):
        variant_prep = copy.copy(prepared)
        variant_prep.cfg = copy.deepcopy(prepared.cfg)
        variant_prep.cfg["train"]["temporal_strategy"] = rule
        scores = predict_scores(variant_prep, model, heads, "fraud", test, device, variant=meta["variant"])
        rows.append({"rule": rule, "pr_auc": pr_auc(y, scores), "variant": meta["variant"], "seed": meta["seed"],
                     "temporal": bool(prepared.cfg["train"]["temporal"]), "config_hash": config_hash(prepared.cfg)})
    return pd.DataFrame(rows)


def main(argv=None):
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Compare the uniform and last neighbour rules for one checkpoint.")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/multitask_seed0.pt")
    p.add_argument("--out", default="plots")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    table = sampling_rule_table(prepare(cfg), resolve_path(a.checkpoint), resolve_device(cfg["train"]["device"]))
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results_sampling_rule.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
