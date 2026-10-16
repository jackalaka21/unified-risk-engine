import numpy as np
import pandas as pd

from pipeline.data_loading import finalize
from pipeline.history_features import history_features


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "email", "device", "TransactionDT", "TransactionAmt"])


def test_hand_checked_example():
    df = frame([("u", "e", "A", 0, 10.0), ("u", "e", "A", 100, 20.0), ("u", "e", "B", 100_000, 30.0)])
    h = history_features(df)
    assert h["user_n_prior"].tolist() == [0, 1, 2]
    assert np.isnan(h["user_amt_mean"].iloc[0]) and h["user_amt_mean"].tolist()[1:] == [10.0, 15.0]
    assert h["user_n_1d"].tolist() == [0, 1, 0]
    assert h["user_n_7d"].tolist() == [0, 1, 2]
    assert np.isnan(h["user_secs_since_prev"].iloc[0]) and h["user_secs_since_prev"].tolist()[1:] == [100.0, 99_900.0]
    assert h["user_distinct_devices"].tolist() == [0, 1, 1]


def test_equal_timestamps_do_not_count_each_other():
    df = frame([("u", "e", "A", 50, 1.0), ("u", "e", "A", 100, 2.0), ("u", "e", "A", 100, 3.0)])
    assert history_features(df)["user_n_prior"].tolist() == [0, 1, 1]


def test_features_never_depend_on_later_transactions(raw_frames):
    df = finalize(raw_frames[0])
    cutoff = df["TransactionDT"].quantile(0.5)
    base = history_features(df)
    perturbed = df.copy()
    later = perturbed["TransactionDT"] > cutoff
    perturbed.loc[later, "TransactionAmt"] = perturbed.loc[later, "TransactionAmt"] * 10
    again = history_features(perturbed)
    early = (df["TransactionDT"] <= cutoff).to_numpy()
    pd.testing.assert_frame_equal(base[early], again[early])


def test_features_do_not_use_labels(raw_frames):
    df = finalize(raw_frames[0])
    flipped = df.copy()
    flipped["isFraud"] = 1 - flipped["isFraud"]
    pd.testing.assert_frame_equal(history_features(df), history_features(flipped))


def test_missing_entities_give_nan_not_crashes():
    df = frame([("u", None, None, 0, 5.0), ("u", "e", None, 10, 6.0), ("v", None, "D", 20, 7.0)])
    h = history_features(df)
    assert h["email_n_prior"].isna().tolist() == [True, False, True]
    assert h["device_n_prior"].isna().tolist() == [True, True, False]
    assert h["user_n_prior"].notna().all()
