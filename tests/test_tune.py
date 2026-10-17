import json

import numpy as np
import pytest

from pipeline.config import config_hash, load_config
from pipeline.prepare import prepare
from scripts.baseline import BaggedHGB
from scripts.tune import SPACE, load_params, save_params, tune_trees
from tests.fixtures import fixture_overrides


def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 1.0).astype(int)
    return X, y


def test_tuning_fits_only_on_train_rows_and_scores_only_validation_rows(monkeypatch):
    import scripts.tune as tune

    X, y = toy()
    train, val = np.arange(0, 400), np.arange(400, 600)
    seen = []
    real_fit = tune._RobustHGB.fit

    def spy(self, Xf, yf, **kw):
        seen.append(len(Xf))
        return real_fit(self, Xf, yf, **kw)

    monkeypatch.setattr(tune._RobustHGB, "fit", spy)
    out = tune_trees(X, y, train, val, trials=3, seed=1)
    assert seen == [400, 400, 400]                                      # never a validation or test row in a fit
    assert len(out["trials"]) == 3 and set(out["params"]) <= set(SPACE) | {"max_iter", "early_stopping", "n_iter_no_change", "random_state"}
    assert max(t["validation_pr_auc"] for t in out["trials"]) == out["best_validation_pr_auc"]


def test_overlapping_train_and_validation_rows_are_refused():
    X, y = toy()
    with pytest.raises(ValueError, match="overlap"):
        tune_trees(X, y, np.arange(0, 400), np.arange(300, 600), trials=1)


def test_tuning_is_deterministic_for_a_seed():
    X, y = toy()
    a = tune_trees(X, y, np.arange(400), np.arange(400, 600), trials=3, seed=2)
    b = tune_trees(X, y, np.arange(400), np.arange(400, 600), trials=3, seed=2)
    assert a["params"] == b["params"] and a["best_validation_pr_auc"] == b["best_validation_pr_auc"]


def test_saved_params_are_tied_to_the_configuration(tmp_path):
    cfg = load_config()
    path = tmp_path / "tuned.json"
    save_params(path, {"params": {"learning_rate": 0.1}, "trials": []}, cfg)
    assert load_params(path, cfg) == {"learning_rate": 0.1}
    other = load_config(overrides={"seed": 7})
    with pytest.raises(ValueError, match="configuration"):
        load_params(path, other)
    assert json.loads(path.read_text())["config_hash"] == config_hash(cfg)
    assert load_params(tmp_path / "missing.json", cfg) is None


def test_a_bag_averages_members_that_differ_only_in_their_random_choices():
    X, y = toy()
    bag = BaggedHGB({"max_iter": 30, "learning_rate": 0.1}, n=3, max_features=0.6).fit(X[:400], y[:400])
    assert len(bag.members) == 3 and len({m.random_state for m in bag.members}) == 3
    expected = np.mean([m.predict_proba(X[400:])[:, 1] for m in bag.members], axis=0)
    got = bag.predict_proba(X[400:])
    assert np.allclose(got[:, 1], expected) and np.allclose(got.sum(axis=1), 1.0)
    assert bag.n_features_in_ == 6


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_final_baselines_refit_only_on_labels_older_than_the_embargo(prep, monkeypatch):
    import scripts.baseline as bl

    sizes = []
    real_fit = bl._RobustHGB.fit

    def spy(self, Xf, yf, **kw):
        sizes.append(len(Xf))
        return real_fit(self, Xf, yf, **kw)

    monkeypatch.setattr(bl._RobustHGB, "fit", spy)
    params = {"max_iter": 20, "learning_rate": 0.1}
    out = bl.final_baselines(prep, seed=0, params=params, n_bag=2)
    assert set(out) == {"final_tuned_tabular", "final_bag_tabular", "final_refit_tabular"}
    train = int(prep.graph["transaction"].train_mask.sum())
    val = int(prep.graph["transaction"].val_mask.sum())
    assert sizes[0] == train and sizes[1:3] == [train, train] and sizes[3:5] == [train + val] * 2
    assert out["final_refit_tabular"]["val"] is None                      # no clean validation scores for a refit model
    te = int(prep.graph["transaction"].test_mask.sum())
    assert out["final_refit_tabular"]["test"].shape == (te,) and out["final_tuned_tabular"]["val"].shape == (val,)
    # every validation label is older than the embargo before the test period starts
    times = prep.df["TransactionDT"].to_numpy()
    assert times[prep.graph["transaction"].val_mask.numpy()].max() <= prep.periods.test_start - prep.periods.embargo
