import numpy as np
import pandas as pd
import pytest

from pipeline.data_loading import finalize
from pipeline.graph_features import LABEL_BASED, LABEL_FREE, LABEL_RATES, graph_features

DAY = 86400


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "email", "device", "TransactionDT", "isFraud"])


def test_label_based_features_by_hand_with_the_label_delay():
    df = frame([("u", "e", "A", 0, 1), ("u", "e", "A", 500, 0), ("u", "e", "A", 2000, 0), ("u", "e", "A", 3000, 1)])
    g = graph_features(df, embargo_seconds=1000, prior_rate=0.1, smoothing=20.0)
    assert g["user_lbl_n_known"].tolist() == [0, 0, 2, 3]        # labels older than the embargo only
    assert g["user_lbl_n_fraud"].tolist() == [0, 0, 1, 1]
    assert g["user_lbl_rate"].iloc[0] == pytest.approx(0.1)       # nothing known: the prior
    assert g["user_lbl_rate"].iloc[2] == pytest.approx((1 + 2) / (2 + 20))
    assert g["user_lbl_rate"].iloc[3] == pytest.approx((1 + 2) / (3 + 20))


def test_shared_entities_by_hand_inside_a_seven_day_window():
    df = frame([("u1", "e", "A", 0, 0), ("u2", "e", "A", 100, 0), ("u1", "e", "A", 200, 0), ("u3", "f", "B", 300, 0)])
    g = graph_features(df, embargo_seconds=1000)
    assert g["device_distinct_users_7d"].tolist() == [0, 1, 2, 0]    # distinct users seen on the device before this row
    assert g["email_distinct_users_7d"].tolist() == [0, 1, 2, 0]
    assert list(g.columns) == LABEL_FREE + LABEL_BASED and not any("age" in c for c in g.columns)


def test_users_older_than_seven_days_fall_out_of_the_window():
    df = frame([("u1", "e", "A", 0, 0), ("u2", "e", "A", 8 * DAY, 0), ("u3", "e", "A", 8 * DAY + 100, 0), ("u2", "e", "A", 8 * DAY + 200, 0)])
    assert graph_features(df, embargo_seconds=1000)["device_distinct_users_7d"].tolist() == [0, 0, 1, 2]


def test_a_user_counted_twice_in_the_window_is_one_user():
    df = frame([("u1", "e", "A", 0, 0), ("u1", "e", "A", 10, 0), ("u1", "e", "A", 20, 0), ("u2", "e", "A", 30, 0)])
    assert graph_features(df, embargo_seconds=5)["device_distinct_users_7d"].tolist() == [0, 1, 1, 1]


def test_ties_do_not_see_each_other():
    df = frame([("u1", "e", "A", 50, 0), ("u2", "e", "A", 100, 0), ("u3", "e", "A", 100, 0)])
    assert graph_features(df, embargo_seconds=10)["device_distinct_users_7d"].tolist() == [0, 1, 1]


def test_features_never_use_the_label_of_a_row_younger_than_the_embargo(raw_frames):
    df = finalize(raw_frames[0])
    embargo = DAY
    cutoff = df["TransactionDT"].quantile(0.5)
    base = graph_features(df, embargo_seconds=embargo)
    flipped = df.copy()
    late = flipped["TransactionDT"] > cutoff
    flipped.loc[late, "isFraud"] = 1 - flipped.loc[late, "isFraud"]
    again = graph_features(flipped, embargo_seconds=embargo)
    safe = (df["TransactionDT"] <= cutoff + embargo).to_numpy()   # these rows only look at labels no younger than the cutoff
    assert safe.sum() > 100
    pd.testing.assert_frame_equal(base[safe], again[safe])
    assert not base[~safe].equals(again[~safe])                   # and the flip does reach rows that are allowed to see it


def test_features_never_depend_on_later_transactions(raw_frames):
    df = finalize(raw_frames[0])
    cutoff = df["TransactionDT"].quantile(0.5)
    base = graph_features(df, embargo_seconds=DAY)
    changed = df.copy()
    later = changed["TransactionDT"] > cutoff
    changed.loc[later, "user"] = "someone_else"
    changed.loc[later, "device"] = "another_device"
    changed.loc[later, "isFraud"] = 1
    again = graph_features(changed, embargo_seconds=DAY)
    early = (df["TransactionDT"] <= cutoff).to_numpy()
    pd.testing.assert_frame_equal(base[early], again[early])


def test_missing_entities_give_nan_not_crashes():
    df = frame([("u", None, None, 0, 0), ("u", "e", None, 10, 1), ("v", None, "D", 20, 0)])
    g = graph_features(df, embargo_seconds=5)
    assert g["email_lbl_rate"].isna().tolist() == [True, False, True]
    assert g["device_distinct_users_7d"].isna().tolist() == [True, True, False]
    assert g["user_lbl_rate"].notna().all()


def test_column_groups_partition_the_output_and_the_embargo_must_be_positive(raw_frames):
    df = finalize(raw_frames[0])
    g = graph_features(df.iloc[:500], embargo_seconds=DAY)
    assert list(g.columns) == LABEL_FREE + LABEL_BASED and set(LABEL_FREE).isdisjoint(LABEL_BASED)
    with pytest.raises(ValueError, match="embargo"):
        graph_features(df.iloc[:10], embargo_seconds=0)


def test_label_free_features_do_not_change_when_every_label_changes(raw_frames):
    df = finalize(raw_frames[0]).iloc[:2000]
    flipped = df.assign(isFraud=1 - df["isFraud"])
    pd.testing.assert_frame_equal(graph_features(df, DAY)[LABEL_FREE], graph_features(flipped, DAY)[LABEL_FREE])


def test_the_rate_columns_are_the_drift_robust_subset_of_the_label_based_ones():
    assert LABEL_RATES == ["user_lbl_rate", "email_lbl_rate", "device_lbl_rate"] and set(LABEL_RATES) <= set(LABEL_BASED)
