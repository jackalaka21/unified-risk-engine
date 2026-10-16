import numpy as np
import pandas as pd
import pytest

from pipeline.data_loading import finalize
from pipeline.splits import (DAY, account_table, check_positives, compute_periods, fraud_masks,
                             period_of, split_users)


@pytest.fixture()
def df(raw_frames):
    return finalize(raw_frames[0])


@pytest.fixture()
def periods(df, cfg):
    s = cfg["splits"]
    return compute_periods(df["TransactionDT"], s["train_frac"], s["val_end_frac"], s["embargo_days"])


@pytest.fixture()
def users(df, cfg):
    return split_users(df["user"], cfg["splits"]["user_fractions"], cfg["seed"])


def test_periods_are_ordered_and_separated_by_the_embargo(periods):
    assert periods.t_train < periods.val_start < periods.val_end < periods.test_start
    assert periods.val_start - periods.t_train == periods.embargo
    assert periods.test_start - periods.val_end == periods.embargo


def test_embargo_that_empties_a_period_is_rejected(df):
    with pytest.raises(ValueError, match="embargo_days"):
        compute_periods(df["TransactionDT"], 0.7, 0.85, embargo_days=60)


def test_split_users_is_deterministic_and_complete(df, cfg):
    a = split_users(df["user"], cfg["splits"]["user_fractions"], 0)
    b = split_users(df["user"], cfg["splits"]["user_fractions"], 0)
    assert a.equals(b) and set(a.index) == set(df["user"]) and set(a) == {"train", "val", "test"}
    assert 0.6 < (a == "train").mean() < 0.8


def test_no_fraud_train_row_belongs_to_a_held_out_user(df, periods, users):
    train = fraud_masks(df, periods, users)["train"]
    assert train.any() and set(df.loc[train, "user"].map(users)) == {"train"}


def test_every_fraud_train_row_is_inside_the_train_period(df, periods, users):
    train = fraud_masks(df, periods, users)["train"]
    assert (df.loc[train, "TransactionDT"] <= periods.t_train).all()


def test_no_evaluation_row_falls_inside_an_embargo_window(df, periods, users):
    masks = fraud_masks(df, periods, users)
    t = df["TransactionDT"]
    embargo1 = (t > periods.t_train) & (t <= periods.val_start)
    embargo2 = (t > periods.val_end) & (t <= periods.test_start)
    for name in ("val", "test"):
        assert not (masks[name] & (embargo1 | embargo2)).any()
    assert masks["val"].any() and masks["test"].any()
    assert set(period_of(t.to_numpy(), periods)) == {"train", "embargo", "val", "test"}


def test_account_labels_ignore_labels_after_t_train(df, periods, users, cfg):
    a = cfg["account"]
    base = account_table(df, periods, users, a["min_transactions"], a["fraud_rate_threshold"])
    shuffled = df.copy()
    later = shuffled["TransactionDT"] > periods.t_train
    rng = np.random.default_rng(0)
    shuffled.loc[later, "isFraud"] = rng.permutation(shuffled.loc[later, "isFraud"].to_numpy())
    again = account_table(shuffled, periods, users, a["min_transactions"], a["fraud_rate_threshold"])
    assert base.equals(again) and base["label"].sum() > 0


def test_users_without_train_period_history_are_ineligible():
    periods = compute_periods(pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 0.5, 0.8, 0.0)
    rows = [("A", 60, 1), ("A", 70, 1), ("A", 80, 1),          # only after t_train
            ("B", 1, 1), ("B", 2, 1),                          # fewer than min_transactions
            ("C", 3, 1), ("C", 4, 1), ("C", 5, 0)]             # eligible, fraud rate 2/3
    df = pd.DataFrame(rows, columns=["user", "TransactionDT", "isFraud"])
    users = pd.Series({"A": "train", "B": "train", "C": "train"}, name="split")
    t = account_table(df, periods, users, min_transactions=3, fraud_rate_threshold=0.2)
    assert t.loc["A", ["eligible", "label"]].tolist() == [False, 0]
    assert t.loc["B", ["eligible", "label"]].tolist() == [False, 0]
    assert t.loc["C", ["eligible", "label"]].tolist() == [True, 1]


def test_too_few_positives_raises_with_the_split_named():
    t = pd.DataFrame({"split": ["train"] * 5 + ["val"] * 5 + ["test"] * 5, "eligible": True,
                      "label": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]})
    with pytest.raises(ValueError, match="val"):
        check_positives(t, min_positives=2)
    check_positives(t, min_positives=1)
