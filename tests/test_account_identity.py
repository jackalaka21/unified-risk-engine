import numpy as np
import pandas as pd
import pytest

from pipeline.account_identity import ACCT_COLUMNS, account_features, account_ids
from pipeline.data_loading import finalize

DAY = 86400


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "email", "device", "TransactionDT", "TransactionAmt", "isFraud", "D1"])


def test_ids_by_hand_the_start_day_is_the_transaction_day_minus_d1():
    df = frame([("u", "e", "A", 10 * DAY, 5.0, 0, 2.0),      # started on day 8
                ("u", "e", "A", 12 * DAY, 5.0, 0, 4.0),      # started on day 8: the same account
                ("u", "e", "A", 12 * DAY, 5.0, 0, 3.0),      # started on day 9: another account behind the same card
                ("u", "e", "A", 13 * DAY, 5.0, 0, np.nan)])  # no D1: falls back to the card-level user
    ids = account_ids(df).tolist()
    assert ids[0] == ids[1] and ids[0] != ids[2] and ids[3] == "u"


def test_every_account_belongs_to_exactly_one_user(raw_frames):
    df = finalize(raw_frames[0])
    ids = account_ids(df)
    assert ids.nunique() > df["user"].nunique()                                  # the identity really is finer
    per_account = df.groupby(ids.to_numpy())["user"].nunique()
    assert (per_account == 1).all()                                              # so no account can straddle two user splits


def test_account_features_have_the_documented_columns(raw_frames):
    df = finalize(raw_frames[0]).iloc[:800]
    out = account_features(df, embargo_seconds=DAY)
    assert list(out.columns) == ACCT_COLUMNS and len(out) == len(df) and out.index.equals(df.index)
    assert not any(c.endswith(("n_prior", "distinct_devices", "distinct_emails")) for c in out.columns)   # nothing that only grows


def test_features_never_depend_on_later_transactions(raw_frames):
    df = finalize(raw_frames[0])
    cutoff = df["TransactionDT"].quantile(0.5)
    base = account_features(df, DAY)
    changed = df.copy()
    later = changed["TransactionDT"] > cutoff
    changed.loc[later, "TransactionAmt"] = changed.loc[later, "TransactionAmt"] * 7
    changed.loc[later, "D1"] = 0.0
    changed.loc[later, "isFraud"] = 1
    early = (df["TransactionDT"] <= cutoff).to_numpy()
    pd.testing.assert_frame_equal(base[early], account_features(changed, DAY)[early])


def test_labels_are_used_only_after_the_embargo(raw_frames):
    df = finalize(raw_frames[0])
    cutoff = df["TransactionDT"].quantile(0.5)
    flipped = df.copy()
    late = flipped["TransactionDT"] > cutoff
    flipped.loc[late, "isFraud"] = 1 - flipped.loc[late, "isFraud"]
    safe = (df["TransactionDT"] <= cutoff + DAY).to_numpy()
    pd.testing.assert_frame_equal(account_features(df, DAY)[safe], account_features(flipped, DAY)[safe])


def test_a_missing_d1_column_is_an_error_naming_it():
    df = frame([("u", "e", "A", 0, 1.0, 0, 1.0)]).drop(columns="D1")
    with pytest.raises(ValueError, match="D1"):
        account_ids(df)
