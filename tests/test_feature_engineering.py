import numpy as np
import pandas as pd
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


def _reference_transform(ft, df):
    """The straightforward column-by-column implementation, kept as the oracle for the vectorised transform."""
    cols = []
    log_amt = np.log1p(df["TransactionAmt"].astype(float).to_numpy())
    cols.append((log_amt - ft.log_amt_mean) / ft.log_amt_std)
    hour = (df["TransactionDT"].to_numpy() // 3600) % 24
    dow = (df["TransactionDT"].to_numpy() // 86400) % 7
    cols += [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
             np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
    for c in ft.cat_cols:
        cols.append(df[c].astype("string").fillna("nan").map(ft.freq[c]).fillna(0.0).to_numpy(dtype=float))
    for c in ft.num_cols:
        mean, std = ft.stats[c]
        values = df[c].astype(float).to_numpy() if c in df.columns else np.full(len(df), np.nan)
        cols.append(np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0))
    for c in ft.na_cols:
        values = df[c].to_numpy() if c in df.columns else np.full(len(df), np.nan)
        cols.append(pd.isna(values).astype(float))
    return np.column_stack(cols).astype(np.float32)


def test_the_vectorised_transform_matches_the_column_by_column_oracle(raw_frames):
    df = finalize(raw_frames[0])
    ft = FeatureTransformer(max_v_columns=20, v_max_missing=0.5).fit(df)
    np.testing.assert_array_equal(ft.transform(df), _reference_transform(ft, df))
    small = df.iloc[:7].drop(columns=["V3", "D2"]).copy()                     # absent columns count as missing
    small.loc[small.index[0], "C1"] = np.inf
    np.testing.assert_array_equal(ft.transform(small), _reference_transform(ft, small))
    nullable = df.iloc[:7].copy()
    nullable["card1"] = nullable["card1"].astype("Int64")
    nullable.loc[nullable.index[2], "card1"] = pd.NA
    np.testing.assert_array_equal(ft.transform(nullable), _reference_transform(ft, nullable))
