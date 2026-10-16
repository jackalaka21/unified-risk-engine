import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.train import (default_spec, load_checkpoint, make_loader, predict_scores, train_variant)
from tests.fixtures import fixture_overrides


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


@pytest.fixture(scope="module")
def prep_last(raw_frames):
    """Prediction is stochastic under the "uniform" neighbour rule; the "last" rule is deterministic."""
    overrides = fixture_overrides()
    overrides["train"]["temporal_strategy"] = "last"
    return prepare(load_config(overrides=overrides), df=raw_frames[0])


@pytest.mark.parametrize("variant", ["multitask", "fraud_only", "account_only", "no_graph"])
def test_every_variant_trains_one_epoch(prep, variant):
    res = train_variant(prep, default_spec(prep, variant, 0), "cpu")
    assert len(res.history) == 1 and np.isfinite(res.history[0]["train_loss"])


def test_checkpoint_round_trip_reproduces_predictions(prep_last, tmp_path):
    spec = default_spec(prep_last, "multitask", 0)
    res = train_variant(prep_last, spec, "cpu", tmp_path / "m.pt")
    idx = spec.fraud_val_idx[:64]
    before = predict_scores(prep_last, res.model, res.heads, "fraud", idx, "cpu")
    model, heads, meta = load_checkpoint(tmp_path / "m.pt", prep_last, "cpu")
    after = predict_scores(prep_last, model, heads, "fraud", idx, "cpu")
    assert np.allclose(before, after, atol=1e-5) and meta["variant"] == "multitask"


def test_predictions_are_probabilities_aligned_with_the_requested_rows(prep):
    spec = default_spec(prep, "multitask", 0)
    res = train_variant(prep, spec, "cpu")
    idx = spec.fraud_val_idx[:40]
    p = predict_scores(prep, res.model, res.heads, "fraud", idx, "cpu")
    assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all()
    u = predict_scores(prep, res.model, res.heads, "account", spec.account_val_idx, "cpu")
    assert u.shape == (len(spec.account_val_idx),)


def test_fixed_epochs_skip_validation(prep):
    spec = default_spec(prep, "fraud_only", 0)
    spec.epochs = 2
    res = train_variant(prep, spec, "cpu")
    assert [h["epoch"] for h in res.history] == [1, 2] and res.best_epoch == 2
    assert all(h.get("val_score") is None for h in res.history)


def test_training_is_reproducible_for_a_seed(prep):
    a = train_variant(prep, default_spec(prep, "fraud_only", 3), "cpu")
    b = train_variant(prep, default_spec(prep, "fraud_only", 3), "cpu")
    assert a.history[0]["train_loss"] == pytest.approx(b.history[0]["train_loss"], rel=1e-4)


def test_sampled_neighbours_never_come_from_the_future(prep):
    if not prep.cfg["train"]["temporal"]:
        pytest.skip("temporal sampling disabled (Option A fallback)")
    idx = np.flatnonzero(prep.graph["transaction"].val_mask.numpy())[:64]
    batch = next(iter(make_loader(prep, "transaction", idx, 64, False)))
    seed_time = batch["transaction"].time[: batch["transaction"].batch_size]
    for nt in batch.node_types:
        assert (batch[nt].time <= seed_time[batch[nt].batch]).all(), nt
    users = np.flatnonzero(prep.graph["user"].val_mask.numpy())[:16]
    ub = next(iter(make_loader(prep, "user", users, 16, False)))
    assert (ub["transaction"].time <= prep.graph.t_train).all()


def test_empty_seed_set_gives_empty_predictions(prep):
    spec = default_spec(prep, "fraud_only", 0)
    res = train_variant(prep, spec, "cpu")
    assert predict_scores(prep, res.model, res.heads, "fraud", np.array([], dtype=int), "cpu").shape == (0,)
