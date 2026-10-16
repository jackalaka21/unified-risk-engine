import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.metrics import pr_auc
from pipeline.prepare import prepare
from scripts.baseline import account_baselines, fraud_baselines, user_aggregates
from tests.fixtures import fixture_overrides


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_fraud_baselines_return_aligned_probabilities(prep):
    out = fraud_baselines(prep, seed=0, max_iter=30)
    val = np.flatnonzero(prep.graph["transaction"].val_mask.numpy())
    test = np.flatnonzero(prep.graph["transaction"].test_mask.numpy())
    assert set(out) == {"naive_tabular", "history_tabular"}
    for res in out.values():
        assert res["val"].shape == (len(val),) and res["test"].shape == (len(test),)
        assert ((res["test"] >= 0) & (res["test"] <= 1)).all()


def test_the_planted_fixture_signal_is_learnable(prep):
    out = fraud_baselines(prep, seed=0, max_iter=60)
    y = prep.df["isFraud"].to_numpy()[np.flatnonzero(prep.graph["transaction"].test_mask.numpy())]
    assert pr_auc(y, out["naive_tabular"]["test"]) > y.mean()  # better than chance on synthetic data only


def test_user_aggregates_use_only_history_up_to_t_train(prep):
    agg = user_aggregates(prep.df, prep.periods.t_train)
    early = prep.df[prep.df["TransactionDT"] <= prep.periods.t_train]
    assert agg["n_tx"].sum() == len(early)
    assert {"n_tx", "amt_mean", "n_devices", "n_emails", "tx_per_day"} <= set(agg.columns)


def test_account_baselines_cover_every_eligible_val_and_test_user(prep):
    out = account_baselines(prep, seed=0, max_iter=30)
    acc = prep.account
    for split in ("val", "test"):
        n = int(((acc["split"] == split) & acc["eligible"]).sum())
        for name in ("hgb_user_aggregates", "mean_fraud_score"):
            assert out[name][split].shape == (n,)
    assert np.isfinite(out["mean_fraud_score"]["test"]).all()


def test_an_all_nan_training_column_does_not_break_the_tree_baseline():
    """sklearn 1.9.1 crashes when binning a column that is entirely NaN (e.g. no identity file, so no device history)."""
    from scripts.baseline import fit_hgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    X[:, 3] = np.nan
    model = fit_hgb(X, y, seed=0, max_iter=10)
    other = X.copy()
    other[:, 3] = 5.0                                   # a value the model never saw in that column is ignored
    assert np.allclose(model.predict_proba(X)[:, 1], model.predict_proba(other)[:, 1])
