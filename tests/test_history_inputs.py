import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.history_inputs import HistoryInputs, with_history_inputs
from pipeline.prepare import prepare
from scripts.baseline import stable_history_columns
from scripts.train import default_spec, train_variant
from tests.fixtures import fixture_overrides


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_inputs_are_log_scaled_standardised_on_the_fit_rows_with_nan_indicators():
    hist = pd.DataFrame({"user_n_1d": [0.0, 3.0, 8.0, 100.0], "email_amt_mean": [10.0, np.nan, 30.0, 50.0]})
    hi = HistoryInputs().fit(hist, np.array([True, True, True, False]))              # the last row is not a fit row
    out = hi.transform(hist)
    assert out.shape == (4, 3) and out.dtype == np.float32                            # two columns + one NaN indicator
    assert out[:3, 0].mean() == pytest.approx(0.0, abs=1e-6) and out[:3, 0].std() == pytest.approx(1.0, rel=0.05)
    assert out[1, 1] == 0.0 and out[1, 2] == 1.0 and out[0, 2] == 0.0                # NaN -> 0 plus an indicator
    assert hi.names == ["hist_user_n_1d", "hist_email_amt_mean", "hist_na_email_amt_mean"]
    assert np.isfinite(out).all()


def test_the_fit_uses_only_the_selected_rows():
    hist = pd.DataFrame({"user_n_1d": [1.0, 2.0, 3.0, 4.0]})
    a = HistoryInputs().fit(hist, np.array([True, True, False, False]))
    b = HistoryInputs().fit(hist.assign(user_n_1d=[1.0, 2.0, 1e6, 1e6]), np.array([True, True, False, False]))
    np.testing.assert_allclose(a.transform(hist)[:2], b.transform(hist)[:2])


def test_cumulative_counts_are_never_model_inputs(prep):
    hp = with_history_inputs(prep)
    assert not any(n.endswith(("_n_prior", "n_distinct_devices", "_distinct_emails")) for n in hp.history_inputs.names)
    assert hp.history_inputs.names[0].startswith("hist_")


def test_the_history_copy_is_wider_and_leaves_the_original_untouched(prep):
    width = prep.graph["transaction"].x.shape[1]
    hp = with_history_inputs(prep)
    extra = len(hp.history_inputs.names)
    assert prep.graph["transaction"].x.shape[1] == width and prep.history_inputs is None
    assert hp.graph["transaction"].x.shape[1] == width + extra == hp.x.shape[1] == len(hp.feature_names)
    assert len(prep.feature_names) == width
    for key in ("train_mask", "val_mask", "test_mask", "y", "time"):
        assert (hp.graph["transaction"][key] == prep.graph["transaction"][key]).all()
    assert set(hp.graph.edge_types) == set(prep.graph.edge_types)
    n_stable = len(stable_history_columns(list(prep.hist.columns)))
    assert 0 < extra <= 2 * n_stable                                                  # columns plus at most one indicator each


def test_a_gnn_trains_on_the_wider_inputs(prep):
    hp = with_history_inputs(prep)
    res = train_variant(hp, default_spec(hp, "fraud_only", 0), "cpu")
    assert len(res.history) == 1 and np.isfinite(res.history[0]["train_loss"])
