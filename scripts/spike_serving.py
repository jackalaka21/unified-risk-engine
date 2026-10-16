"""Serving spike: what does the stock NeighborLoader cost per single request, against the custom index?"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pipeline.config import PROJECT_ROOT
from scripts.train import make_loader
from serving.graph_index import GraphIndex


def time_stock_loader(prep, rows) -> list:
    """Milliseconds per request when each request builds a batch-size-1 loader (what the stock path needs online)."""
    out = []
    for r in rows:
        t0 = time.perf_counter()
        next(iter(make_loader(prep, "transaction", np.array([int(r)]), 1, False)))
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def time_index(prep, rows) -> list:
    df = prep.df
    index = GraphIndex.from_prepared(prep, int(rows[0]))
    out = []
    for r in rows:
        r = int(r)
        keys = (df["user"].iat[r], df["email"].iat[r], df["device"].iat[r])
        t0 = time.perf_counter()
        index.subgraph(keys, int(df["TransactionDT"].iat[r]), prep.x[r])
        out.append((time.perf_counter() - t0) * 1e3)
        index.add(int(df["TransactionDT"].iat[r]), float(df["TransactionAmt"].iat[r]), keys, prep.x[r])
    return out


def main(argv=None):
    from pipeline.config import load_config
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Compare per-request neighbourhood cost: stock loader vs custom index.")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        cfg = load_config(overrides={**fixture_overrides(), "train": {**fixture_overrides()["train"], "temporal_strategy": "last"}})
        prep = prepare(cfg, df=make_fixture()[0])
    else:
        prep = prepare(load_config(overrides={"train": {"temporal_strategy": "last"}}))
    rows = np.arange(int(len(prep.df) * 0.85), int(len(prep.df) * 0.85) + a.n)
    stock, custom = np.array(time_stock_loader(prep, rows)), np.array(time_index(prep, rows))
    text = ("# Serving spike\n\n" + ("> Fixture data: indicative only.\n\n" if a.fixture else "")
            + "Per-request cost of building the sampled neighbourhood, milliseconds, one machine, CPU:\n\n"
            "| method | p50 | p95 |\n|---|---|---|\n"
            f"| stock `NeighborLoader`, batch size 1 (loader built per request) | {np.percentile(stock, 50):.2f} | {np.percentile(stock, 95):.2f} |\n"
            f"| custom time-sorted index | {np.percentile(custom, 50):.3f} | {np.percentile(custom, 95):.3f} |\n\n"
            "Decision: serve with the custom index regardless of the gap. The serving image then needs no sampler "
            "dependency (pyg-lib), the index is append-only so new transactions become neighbours immediately, and the "
            "parity tests pin it to the stock sampler's `\"last\"` rule.\n")
    path = PROJECT_ROOT / "docs" / "serving-spike.md"
    path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
