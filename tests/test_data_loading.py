import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.data_loading import finalize, load_raw
from tests.fixtures import write_fixture


def cfg_for(tmp_path):
    return load_config(overrides={"data": {"dir": str(tmp_path)}})


def test_load_raw_merges_identity_sorts_and_adds_keys(tmp_path):
    write_fixture(tmp_path, n_tx=1500, n_users=80, days=30)
    df = load_raw(cfg_for(tmp_path))
    assert list(df["TransactionDT"]) == sorted(df["TransactionDT"])
    assert {"user", "email", "device", "DeviceInfo"} <= set(df.columns)
    assert df["user"].str.contains("_").all()
    assert df["device"].isna().any() and df["device"].notna().any()  # identity covers only some rows


def test_identity_file_is_optional(tmp_path):
    write_fixture(tmp_path, n_tx=800, n_users=40, days=20, with_identity=False)
    df = load_raw(cfg_for(tmp_path))
    assert df["device"].isna().all()


def test_missing_transaction_file_names_the_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_transaction.csv"):
        load_raw(cfg_for(tmp_path))


def test_missing_columns_are_named(tmp_path):
    write_fixture(tmp_path, n_tx=300, n_users=20, days=10)
    path = tmp_path / "train_transaction.csv"
    pd.read_csv(path).drop(columns=["isFraud"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="isFraud"):
        load_raw(cfg_for(tmp_path))


def test_missing_addr1_and_email_do_not_break_keys(raw_frames):
    tx, _ = raw_frames
    df = finalize(tx)
    assert df["user"].notna().all()  # a sentinel keeps users with no addr1 usable
    assert df["email"].isna().any()
