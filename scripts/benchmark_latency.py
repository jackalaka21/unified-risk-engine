"""Latency and throughput of the online scorers: in process and over HTTP, with a component breakdown."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import platform
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.config import config_hash, resolve_path
from scripts.baseline import fit_hgb
from scripts.spike_serving import time_stock_loader
from scripts.train import load_checkpoint, predict_scores
from serving.graph_index import GraphIndex
from serving.scorer import OnlineScorer, TabularOnlineScorer, payload_of, raw_kinds_of

DEFAULT_CHECKPOINT = "models/multitask_seed0.pt"      # what scripts.run_all writes; the fraud head is the one served
COMPONENTS = ("feature_transform", "neighbour_lookup", "model_forward", "index_update")


class ParityError(RuntimeError):
    """The online scorer disagrees with the offline batch scores: no latency number may be reported."""


def percentiles(ms) -> dict:
    a = np.asarray(ms, dtype=float)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


def _batches(payloads, size, warmup):
    batches = [payloads[i:i + size] for i in range(0, len(payloads), size)]
    batches = [b for b in batches if len(b) == size] or batches
    if len(batches) <= warmup:                                   # too little data to warm up and measure: measure everything
        return [], batches
    return batches[:warmup], batches[warmup:]


def check_parity(prepared, make_scorer, n=200, atol=1e-4) -> dict:
    cfg = copy.deepcopy(prepared.cfg)
    if not cfg["train"]["temporal"]:
        return {"checked": False, "max_abs_diff": None, "n_rows": 0, "reason": "temporal sampling is disabled in the config"}
    try:
        import pyg_lib  # noqa: F401
    except ImportError:
        return {"checked": False, "max_abs_diff": None, "n_rows": 0, "reason": "pyg-lib is not installed"}
    cfg["train"]["temporal_strategy"] = "last"                   # the rule the online index implements
    offline_prep = copy.copy(prepared)
    offline_prep.cfg = cfg
    scorer = make_scorer()
    start = len(scorer.index)
    rows = np.arange(start, min(start + n, len(prepared.df)))
    online = np.array(scorer.score_batch([payload_of(prepared.df, int(i)) for i in rows]))
    offline = predict_scores(offline_prep, scorer.model, scorer.heads, "fraud", rows, scorer.device)
    diff = float(np.abs(online - offline).max()) if len(rows) else 0.0
    if diff > atol:
        raise ParityError(f"online and offline scores differ by {diff:.2e} (> {atol}); fix train/serve skew before benchmarking")
    return {"checked": True, "max_abs_diff": diff, "n_rows": int(len(rows)), "reason": ""}


def bench_inprocess(make_scorer, payloads, batch_size, warmup_batches=3) -> dict:
    scorer = make_scorer()
    warm, measured = _batches(payloads, batch_size, warmup_batches)
    for b in warm:
        scorer.score_batch(b)
    lat, comp, n_req = [], dict.fromkeys(COMPONENTS, 0.0), 0
    start = time.perf_counter()
    for b in measured:
        tm: dict = {}
        t0 = time.perf_counter()
        scorer.score_batch(b, timings=tm)
        lat.append((time.perf_counter() - t0) * 1e3)
        n_req += len(b)
        for c in COMPONENTS:
            comp[c] += tm.get(c, 0.0)
    wall = time.perf_counter() - start
    return {**percentiles(lat), "n_requests": n_req, "throughput_rps": n_req / wall,
            **{f"{c}_ms": comp[c] * 1e3 / len(lat) for c in COMPONENTS}}


def start_server(app, host="127.0.0.1", timeout=30.0):
    import uvicorn

    with socket.socket() as s:
        s.bind((host, 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("benchmark server did not start")
        time.sleep(0.05)
    return server, thread, f"http://{host}:{port}"


def bench_http(make_scorer, payloads, batch_size, concurrency, warmup_batches=3) -> dict:
    import httpx

    from serving.app import create_app

    scorer = make_scorer()
    server, thread, url = start_server(create_app(scorer))
    try:
        warm, measured = _batches(payloads, batch_size, warmup_batches)

        def send(client, b):
            if batch_size == 1:
                r = client.post("/score", json=b[0])
            else:
                r = client.post("/score_batch", json={"transactions": b})
            if r.status_code != 200:
                raise RuntimeError(f"benchmark request failed: {r.status_code} {r.text}")

        with httpx.Client(base_url=url, timeout=60) as client:
            for b in warm:
                send(client, b)

        def worker(chunk):
            lat = []
            with httpx.Client(base_url=url, timeout=60) as c:
                for b in chunk:
                    t0 = time.perf_counter()
                    send(c, b)
                    lat.append((time.perf_counter() - t0) * 1e3)
            return lat

        chunks = [measured[i::concurrency] for i in range(concurrency)]
        start = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            lat = [x for part in pool.map(worker, chunks) for x in part]
        wall = time.perf_counter() - start
        n_req = sum(len(b) for b in measured)
        return {**percentiles(lat), "n_requests": n_req, "throughput_rps": n_req / wall}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def hardware_info() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def run_benchmark(prepared, checkpoint, out_dir, devices=None, batch_sizes=(1, 8, 32, 128), n_requests=2000,
                  budget_ms=100.0, skip_parity=False, include_http=True, include_tabular=True, concurrent_clients=8,
                  fixture=False, repeats=1, log=print) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    devices = devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, heads, ckpt = load_checkpoint(checkpoint, prepared, "cpu")
    df = prepared.df
    upto = int((df["TransactionDT"] < prepared.periods.test_start).sum())
    base_index = GraphIndex.from_prepared(prepared, upto)
    kinds = raw_kinds_of(df)
    payloads = [payload_of(df, i) for i in range(upto, min(upto + n_requests, len(df)))]

    def gnn_factory(device, late="reject"):
        return lambda: OnlineScorer(copy.deepcopy(model), copy.deepcopy(heads), prepared.transformer,
                                    copy.deepcopy(base_index), kinds, device=device, late=late)

    parity = {"checked": False, "max_abs_diff": None, "n_rows": 0, "reason": "skipped by request"}
    if not skip_parity:
        parity = check_parity(prepared, gnn_factory("cpu"))           # raises ParityError: nothing is written
        log(f"parity: {parity}")

    rows = []

    def add(model_name, device, mode, batch, conc, run, notes=""):
        """Run the measurement `repeats` times: timings on a laptop vary from run to run, so one number would overstate precision."""
        for r in range(repeats):
            result = run()
            rows.append({"model": model_name, "device": device, "mode": mode, "batch_size": batch, "concurrency": conc,
                         "repeat": r, **result, "p99_within_budget": result.get("p99_ms", np.nan) <= budget_ms, "notes": notes})

    for device in devices:
        for b in batch_sizes:
            log(f"gnn {device} in-process batch={b}")
            add("gnn", device, "inprocess", b, 1, lambda: bench_inprocess(gnn_factory(device), payloads, b))
    if include_http:
        for b in sorted({1} | ({32} & set(batch_sizes))):
            log(f"gnn cpu http batch={b}")
            add("gnn", "cpu", "http", b, 1, lambda: bench_http(gnn_factory("cpu"), payloads, b, 1))
        log(f"gnn cpu http batch=1 clients={concurrent_clients}")
        add("gnn", "cpu", "http", 1, concurrent_clients,
            lambda: bench_http(gnn_factory("cpu", late="clamp"), payloads, 1, concurrent_clients),
            notes="late events clamped to the newest indexed time; scoring is serialised by one lock")
    if include_tabular:
        X = np.hstack([prepared.x, prepared.hist.to_numpy(np.float32)])
        train_rows = np.flatnonzero(prepared.graph["transaction"].train_mask.numpy())
        hgb = fit_hgb(X[train_rows], df["isFraud"].to_numpy()[train_rows], seed=0)

        def tab_factory(late="reject"):
            return lambda: TabularOnlineScorer(hgb, prepared.transformer, copy.deepcopy(base_index), kinds,
                                               hist_columns=list(prepared.hist.columns), late=late)

        for b in batch_sizes:
            log(f"history_tabular cpu in-process batch={b}")
            add("history_tabular", "cpu", "inprocess", b, 1, lambda: bench_inprocess(tab_factory(), payloads, b),
                notes="neighbour_lookup_ms is the history-feature computation")
        if include_http:
            add("history_tabular", "cpu", "http", 1, 1, lambda: bench_http(tab_factory(), payloads, 1, 1))
            add("history_tabular", "cpu", "http", 1, concurrent_clients,
                lambda: bench_http(tab_factory("clamp"), payloads, 1, concurrent_clients), notes="late events clamped")
    if parity["checked"]:
        sample = np.arange(upto, min(upto + 100, len(df)))
        add("stock_neighbor_loader", "cpu", "lookup_only", 1, 1, lambda: percentiles(time_stock_loader(prepared, sample)),
            notes="neighbourhood sampling only, loader built per request; no features, model or HTTP")
    table = pd.DataFrame(rows)
    for c in COMPONENTS:
        if f"{c}_ms" not in table:
            table[f"{c}_ms"] = np.nan
    meta = {"hardware": hardware_info(), "budget_ms": budget_ms, "budget_note": "illustrative assumption, not a requirement",
            "config_hash": config_hash(prepared.cfg), "fixture": bool(fixture), "parity": parity, "variant": ckpt["variant"],
            "indexed_transactions": upto, "index_pickle_mb": len(pickle.dumps(base_index)) / 1e6,
            "n_requests": len(payloads), "repeats": repeats, "notes": "one machine replaying historical data; not a production service-level claim"}
    paths = {"csv": out_dir / "latency.csv", "meta": out_dir / "latency_meta.json"}
    table.to_csv(paths["csv"], index=False)
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def main(argv=None):
    from pipeline.config import load_config
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Benchmark online scoring latency.")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--out", default="plots")
    p.add_argument("--n-requests", type=int, default=2000)
    p.add_argument("--budget-ms", type=float, default=100.0)
    p.add_argument("--repeats", type=int, default=3, help="run every measurement this many times")
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        cfg = load_config(a.config, overrides=fixture_overrides())
        prepared = prepare(cfg, df=make_fixture()[0])
    else:
        prepared = prepare(load_config(a.config))
    paths = run_benchmark(prepared, resolve_path(a.checkpoint), resolve_path(a.out), n_requests=a.n_requests,
                          budget_ms=a.budget_ms, fixture=a.fixture, repeats=a.repeats)
    print(pd.read_csv(paths["csv"])[["model", "device", "mode", "batch_size", "concurrency", "p50_ms", "p95_ms",
                                      "p99_ms", "throughput_rps"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
