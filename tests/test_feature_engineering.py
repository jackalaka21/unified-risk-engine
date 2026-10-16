import numpy as np
import pytest

from pipeline.data_loading import finalize
from pipeline.feature_engineering import FeatureTransformer, fit_on_train
from pipeline.splits import compute_periods


@pytest.fixture()
def df(raw_frames):
    return finalize(raw_frames[0])


@pytest.fixture()
def periods(df, cfg):
    s = cfg["splits"]
    return compute_periods(df["TransactionDT"], s["train_frac"], s["val_end_frac"], s["embargo_days"])


def test_statistics_come_from_train_rows_only(df, periods, cfg):
    train = df[df["TransactionDT"] <= periods.t_train]
    ft = fit_on_train(df, periods, cfg)
    assert ft.log_amt_mean == pytest.approx(np.log1p(train["TransactionAmt"]).mean())
    poisoned = df.copy()
    later = poisoned["TransactionDT"] > periods.t_train
    poisoned.loc[later, "TransactionAmt"] = poisoned.loc[later, "TransactionAmt"] * 1000
    ft2 = fit_on_train(poisoned, periods, cfg)
    assert ft2.log_amt_mean == ft.log_amt_mean and ft2.feature_names == ft.feature_names


def test_transform_shape_dtype_and_finiteness(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    x = ft.transform(df)
    assert x.shape == (len(df), len(ft.feature_names)) and x.dtype == np.float32
    assert np.isfinite(x).all()


def test_v_columns_respect_the_missing_rate_and_the_cap(df, periods, cfg):
    ft = FeatureTransformer(max_v_columns=5, v_max_missing=0.5).fit(df[df["TransactionDT"] <= periods.t_train])
    assert len(ft.v_cols) == 5
    assert not {f"V{j}" for j in range(31, 41)} & set(ft.v_cols)  # those are 80% missing


def test_unseen_category_gets_zero_frequency(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    row = df.iloc[[0]].copy()
    row["P_emaildomain"] = "never-seen.example"
    x = ft.transform(row)
    assert x[0, ft.feature_names.index("freq_P_emaildomain")] == 0.0


def test_train_features_are_centred(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    x = ft.transform(df[df["TransactionDT"] <= periods.t_train])
    assert abs(x[:, ft.feature_names.index("C1")].mean()) < 1e-3


def test_missing_values_get_indicators_not_nans(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    assert any(name.startswith("na_D") for name in ft.feature_names)
    x = ft.transform(df.head(50))
    assert np.isfinite(x).all()
