import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.stacking import fold_split, stacked_baselines
from tests.fixtures import fixture_overrides


def test_folds_partition_the_rows_and_never_split_a_user():
    users = np.array(["a", "b", "a", "c", "b", "d", "e", "a", "f", "c"])
    folds = fold_split(users, k=3, seed=0)
    assert len(folds) == 3
    seen = np.concatenate([hold for _, hold in folds])
    assert sorted(seen.tolist()) == list(range(len(users)))                      # every row is held out exactly once
    for fit, hold in folds:
        assert set(fit).isdisjoint(hold) and len(fit) + len(hold) == len(users)
        assert set(users[fit]).isdisjoint(users[hold])                           # the model never saw a held-out user


def test_folds_are_deterministic_for_a_seed_and_differ_between_seeds():
    users = np.array([f"u{i % 40}" for i in range(400)])
    a, b = fold_split(users, 3, seed=1), fold_split(users, 3, seed=1)
    assert all((x[1] == y[1]).all() for x, y in zip(a, b))
    c = fold_split(users, 3, seed=2)
    assert not all((x[1] == y[1]).all() for x, y in zip(a, c))


def test_folds_are_roughly_balanced():
    users = np.array([f"u{i % 90}" for i in range(900)])
    sizes = [len(hold) for _, hold in fold_split(users, 3, seed=0)]
    assert min(sizes) > 200 and max(sizes) < 400


def test_too_few_users_for_the_folds_is_an_error():
    with pytest.raises(ValueError, match="users"):
        fold_split(np.array(["a", "a", "b"]), k=3, seed=0)


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_stacked_baselines_return_aligned_probabilities(prep):
    out = stacked_baselines(prep, seed=0, device="cpu", folds=3, max_iter=20)
    assert set(out) == {"stacked_tabular", "stacked_labels_tabular"}
    va = np.flatnonzero(prep.graph["transaction"].val_mask.numpy())
    te = np.flatnonzero(prep.graph["transaction"].test_mask.numpy())
    for res in out.values():
        assert res["val"].shape == (len(va),) and res["test"].shape == (len(te),)
        assert ((res["test"] >= 0) & (res["test"] <= 1)).all()


def test_the_gnn_score_is_one_extra_column_and_the_models_are_trained_on_out_of_fold_scores(prep, monkeypatch):
    import scripts.stacking as st

    specs = []
    real = st.train_variant

    def spy(prepared, spec, device="cpu", checkpoint_path=None):
        specs.append(spec)
        return real(prepared, spec, device, checkpoint_path)

    monkeypatch.setattr(st, "train_variant", spy)
    out = st.stacked_baselines(prep, seed=0, device="cpu", folds=3, max_iter=20)
    assert len(specs) == 4                                                       # three fold models and one full model
    train_users = prep.df["user"].to_numpy()
    full = specs[-1]
    for spec in specs[:-1]:
        held = np.setdiff1d(full.fraud_train_idx, spec.fraud_train_idx)
        assert len(held) > 0 and set(train_users[held]).isdisjoint(train_users[spec.fraud_train_idx])
        assert (spec.fraud_val_idx == full.fraud_val_idx).all()                   # early stopping looks at the later period only
    assert out["stacked_tabular"]["model"].n_features_in_ == out["stacked_labels_tabular"]["model"].n_features_in_ - 3
