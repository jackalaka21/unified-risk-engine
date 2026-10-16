import pandas as pd

from pipeline.config import load_config
from scripts.check_data import check_data
from tests.fixtures import write_fixture


def cfg_for(tmp_path):
    return load_config(overrides={"data": {"dir": str(tmp_path)}})


def test_missing_files_give_the_download_instruction(tmp_path):
    problems = check_data(cfg_for(tmp_path))
    assert any("train_transaction.csv" in p for p in problems)
    assert any("kaggle competitions download -c ieee-fraud-detection" in p for p in problems)


def test_a_small_but_valid_file_is_reported_as_not_the_full_dataset(tmp_path):
    write_fixture(tmp_path, n_tx=500, n_users=30, days=10)
    problems = check_data(cfg_for(tmp_path))
    assert any("rows" in p for p in problems) and not any("missing columns" in p for p in problems)


def test_missing_columns_are_named(tmp_path):
    write_fixture(tmp_path, n_tx=500, n_users=30, days=10)
    path = tmp_path / "train_transaction.csv"
    pd.read_csv(path).drop(columns=["card1"]).to_csv(path, index=False)
    assert any("card1" in p for p in check_data(cfg_for(tmp_path)))
