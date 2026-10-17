"""Random search over the tree hyper-parameters. Fits on train rows only and scores validation rows only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.config import config_hash, load_config, resolve_path
from pipeline.metrics import pr_auc
from scripts.baseline import _history_matrix, _RobustHGB, _tx_idx

SPACE = {"learning_rate": lambda r: float(np.exp(r.uniform(np.log(0.02), np.log(0.2)))),
         "max_leaf_nodes": lambda r: int(r.choice([15, 31, 63, 127])),
         "min_samples_leaf": lambda r: int(r.choice([20, 50, 100, 200])),
         "l2_regularization": lambda r: float(r.choice([0.0, 0.1, 1.0, 10.0])),
         "max_features": lambda r: float(r.choice([0.5, 0.8, 1.0]))}
FIXED = {"max_iter": 600, "early_stopping": True, "n_iter_no_change": 20, "random_state": 0}


def tune_trees(X, y, train_idx, val_idx, trials=20, seed=1) -> dict:
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    if set(train_idx.tolist()) & set(val_idx.tolist()):
        raise ValueError("the training and validation rows overlap: tuning on rows it was fitted on would be meaningless")
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(trials):
        params = {**{k: draw(rng) for k, draw in SPACE.items()}, **FIXED}
        model = _RobustHGB(**params).fit(X[train_idx], y[train_idx])
        results.append({"params": params, "validation_pr_auc": pr_auc(y[val_idx], model.predict_proba(X[val_idx])[:, 1])})
    best = max(results, key=lambda r: r["validation_pr_auc"])
    return {"params": best["params"], "trials": results, "best_validation_pr_auc": best["validation_pr_auc"]}


def save_params(path, result: dict, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"config_hash": config_hash(cfg), **result}, indent=2), encoding="utf-8")


def load_params(path, cfg: dict):
    path = Path(path)
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["config_hash"] != config_hash(cfg):
        raise ValueError(f"{path} was tuned under a different configuration; delete it or re-run scripts.tune")
    return saved["params"]


def tune_or_load(prepared, out_dir, log=print) -> dict:
    """Tuned parameters for the best tree features: reused when tuned under the same configuration, else searched once."""
    path = Path(out_dir) / "tuned_params.json"
    params = load_params(path, prepared.cfg)
    if params is not None:
        log(f"using tuned parameters from {path}")
        return params
    log("tuning the tree hyper-parameters on the validation period")
    X = _history_matrix(prepared, stable=True, graph="final")
    result = tune_trees(X, prepared.df["isFraud"].to_numpy(), _tx_idx(prepared, "train"), _tx_idx(prepared, "val"),
                        trials=prepared.cfg["tune"]["trials"])
    save_params(path, result, prepared.cfg)
    return result["params"]


def main(argv=None):
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Tune the tree hyper-parameters on the validation period.")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="plots")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    params = tune_or_load(prepare(cfg), resolve_path(a.out))
    print(params)


if __name__ == "__main__":
    main()
