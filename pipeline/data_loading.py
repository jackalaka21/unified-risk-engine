"""Load and normalise the IEEE-CIS CSV files."""
from __future__ import annotations

import pandas as pd

from pipeline.config import resolve_path

REQUIRED_TX = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
               "card1", "addr1", "P_emaildomain"]


def finalize(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    if "DeviceInfo" not in tx.columns:
        tx["DeviceInfo"] = pd.NA
    tx["isFraud"] = tx["isFraud"].astype(int)
    tx["TransactionDT"] = tx["TransactionDT"].astype("int64")
    tx = tx.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    tx["user"] = (tx["card1"].fillna(-1).astype("int64").astype(str) + "_"
                  + tx["addr1"].fillna(-1).astype("int64").astype(str))
    tx["email"] = tx["P_emaildomain"]
    tx["device"] = tx["DeviceInfo"]
    return tx


def load_raw(cfg: dict) -> pd.DataFrame:
    directory = resolve_path(cfg["data"]["dir"])
    tx_path = directory / cfg["data"]["transaction_file"]
    if not tx_path.exists():
        raise FileNotFoundError(
            f"Expected {tx_path}. Download train_transaction.csv from the Kaggle competition "
            "'ieee-fraud-detection' (a Kaggle account and acceptance of the rules are required) "
            "and place it in data/ (see README)."
        )
    tx = pd.read_csv(tx_path)
    missing = [c for c in REQUIRED_TX if c not in tx.columns]
    if missing:
        raise ValueError(f"{tx_path} is missing required columns: {missing}")
    id_path = directory / cfg["data"]["identity_file"]
    if id_path.exists():
        tx = tx.merge(pd.read_csv(id_path), on="TransactionID", how="left")
    return finalize(tx)
