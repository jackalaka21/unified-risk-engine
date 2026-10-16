import json

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_latency import COMPONENTS, ParityError, check_parity, percentiles, run_benchmark
from scripts.train import default_spec, train_variant
from serving.scorer import OnlineScorer, payload_of
from tests.serving_helpers import HAS_TEMPORAL, last_rule_prepared, make_scorer


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


@pytest.fixture(scope="module")
def ckpt(prep, tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "m.pt"
    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", path)
    return path


def test_percentiles_by_hand():
    p = percentiles(list(range(1, 101)))
    assert p["n"] == 100 and p["p50_ms"] == pytest.approx(50.5) and p["p99_ms"] == pytest.approx(99.01)
    assert p["max_ms"] == 100 and p["p50_ms"] <= p["p95_ms"] <= p["p99_ms"] <= p["max_ms"]


def test_replay_payloads_survive_strict_json(prep):
    for i in range(0, len(prep.df), 500):
        json.dumps(payload_of(prep.df, i), allow_nan=False)


def test_benchmark_report_has_every_field_and_never_asserts_on_timing(prep, ckpt, tmp_path):
    paths = run_benchmark(prep, ckpt, tmp_path, devices=["cpu"], batch_sizes=(1, 4), n_requests=40, include_http=False,
                          fixture=True, log=lambda *_: None)
    df = pd.read_csv(paths["csv"])
    meta = json.loads(paths["meta"].read_text())
    assert {"gnn", "history_tabular"} <= set(df["model"])
    assert {"model", "device", "mode", "batch_size", "concurrency", "n", "p50_ms", "p95_ms", "p99_ms", "throughput_rps",
            "p99_within_budget"} <= set(df.columns)
    assert all(f"{c}_ms" in df.columns for c in COMPONENTS)
    inproc = df[df["mode"] == "inprocess"]
    assert (inproc["p50_ms"] <= inproc["p95_ms"]).all() and (inproc["p95_ms"] <= inproc["p99_ms"]).all()
    assert set(inproc["batch_size"]) == {1, 4}
    assert meta["budget_ms"] == 100.0 and meta["fixture"] is True and "hardware" in meta and "index_pickle_mb" in meta
    assert meta["parity"]["checked"] is HAS_TEMPORAL


@pytest.mark.skipif(not HAS_TEMPORAL, reason="parity needs temporal sampling")
def test_parity_passes_for_the_real_scorer_and_fails_for_a_skewed_one(prep):
    p = len(prep.df) // 2
    ok = check_parity(prep, lambda: make_scorer(prep, p), n=30)
    assert ok["checked"] and ok["max_abs_diff"] < 1e-4 and ok["n_rows"] > 0

    class Skewed(OnlineScorer):
        def score_batch(self, payloads, timings=None):
            return [min(1.0, s + 0.2) for s in super().score_batch(payloads, timings)]

    def skewed():
        s = make_scorer(prep, p)
        return Skewed(s.model, s.heads, s.transformer, s.index, s.raw_kinds)

    with pytest.raises(ParityError):
        check_parity(prep, skewed, n=30)


def test_parity_is_reported_as_unchecked_without_temporal_sampling(prep):
    if HAS_TEMPORAL:
        pytest.skip("temporal sampling is available here")
    assert check_parity(prep, lambda: make_scorer(prep, 100), n=10)["checked"] is False


@pytest.mark.slow
def test_http_benchmark_runs_end_to_end(prep, ckpt, tmp_path):
    paths = run_benchmark(prep, ckpt, tmp_path, devices=["cpu"], batch_sizes=(1,), n_requests=30, include_http=True,
                          concurrent_clients=2, include_tabular=False, skip_parity=True, fixture=True, log=lambda *_: None)
    df = pd.read_csv(paths["csv"])
    http = df[df["mode"] == "http"]
    assert set(http["concurrency"]) == {1, 2} and http["throughput_rps"].gt(0).all()
