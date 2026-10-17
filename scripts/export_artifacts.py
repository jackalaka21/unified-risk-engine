"""Turn a trained checkpoint and the history before the test period into a self-contained scoring bundle."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np

from pipeline.config import PROJECT_ROOT, config_hash, load_config, resolve_path
from scripts.train import load_checkpoint
from serving.bundle import save_bundle
from serving.graph_index import ENTITIES, GraphIndex
from serving.scorer import payload_of, raw_kinds_of


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def export_artifacts(prepared, checkpoint, out_dir, history_until="test_start", device="cpu", fixture=False, n_samples=50):
    if prepared.history_inputs is not None:
        raise ValueError("this model uses history inputs, which the online scorer does not compute yet; "
                         "export a model trained on the per-transaction features")
    model, heads, ckpt = load_checkpoint(checkpoint, prepared, device)
    if ckpt["variant"] not in ("multitask", "fraud_only"):
        raise ValueError(f"checkpoint variant {ckpt['variant']!r} has no fraud head to serve")
    cut = {"test_start": prepared.periods.test_start, "val_end": prepared.periods.val_end, "all": np.inf}[history_until]
    upto = int((prepared.df["TransactionDT"] < cut).sum())           # rows are sorted by time
    index = GraphIndex.from_prepared(prepared, upto)
    m = prepared.cfg["model"]
    model_cfg = dict(tx_in=prepared.x.shape[1], edge_types=list(prepared.graph.edge_types), hidden=m["hidden"],
                     heads=m["heads"], layers=m["layers"], dropout=m["dropout"], conv=m["conv"])
    chash = config_hash(prepared.cfg)
    meta = {"version": f"{ckpt['variant']}-seed{ckpt['seed']}-{chash}", "variant": ckpt["variant"], "seed": ckpt["seed"],
            "config_hash": chash, "git_commit": _git_commit(), "indexed_transactions": upto,
            "history_until": history_until, "fixture": bool(fixture),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    samples = [payload_of(prepared.df, i) for i in range(upto, min(upto + n_samples, len(prepared.df)))]
    entities = [nt for nt in ENTITIES if len(prepared.entity_index[nt]) > 0]       # types that had nodes (and edges) in training
    return save_bundle(out_dir, model_cfg, model, heads, prepared.transformer, index, raw_kinds_of(prepared.df),
                       prepared.hist.columns, meta, samples, entities=entities)


def main(argv=None):
    from pipeline.prepare import prepare
    from scripts.train import default_spec, train_variant

    p = argparse.ArgumentParser(description="Export the scoring bundle.")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/multitask_seed0.pt")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--history-until", default="test_start", choices=["test_start", "val_end", "all"])
    p.add_argument("--fixture", action="store_true", help="train a one-epoch model on generated data (CI and demos)")
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        prepared = prepare(load_config(a.config, overrides=fixture_overrides()), df=make_fixture()[0])
        ckpt = resolve_path(a.out) / "fixture_checkpoint.pt"
        train_variant(prepared, default_spec(prepared, "multitask", 0), "cpu", ckpt)
    else:
        prepared, ckpt = prepare(load_config(a.config)), resolve_path(a.checkpoint)
    paths = export_artifacts(prepared, ckpt, resolve_path(a.out), a.history_until, fixture=a.fixture)
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
