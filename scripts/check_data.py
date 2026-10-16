"""Check that the Kaggle files are in place before a real run. Never touches credentials."""
from __future__ import annotations

import pandas as pd

from pipeline.config import load_config, resolve_path
from pipeline.data_loading import REQUIRED_TX

EXPECTED_ROWS = 590_540
HOW_TO = ("Download the data yourself (it needs your own Kaggle login and acceptance of the competition rules): "
          "`kaggle competitions download -c ieee-fraud-detection`, unzip train_transaction.csv "
          "(and optionally train_identity.csv) into the data directory.")


def check_data(cfg) -> list[str]:
    folder = resolve_path(cfg["data"]["dir"])
    path = folder / cfg["data"]["transaction_file"]
    if not path.exists():
        return [f"{path} not found. {HOW_TO}"]
    problems = []
    head = pd.read_csv(path, nrows=5)
    missing = [c for c in REQUIRED_TX if c not in head.columns]
    if missing:
        problems.append(f"{path.name} is missing columns {missing}")
    rows = sum(1 for _ in open(path, "rb")) - 1
    if rows < EXPECTED_ROWS * 0.9:
        problems.append(f"{path.name} has {rows} rows; the full IEEE-CIS training file has about {EXPECTED_ROWS}. "
                        "Real results need the full file.")
    if not (folder / cfg["data"]["identity_file"]).exists():
        problems.append(f"{cfg['data']['identity_file']} not found: device edges will be absent (allowed, but the graph is smaller).")
    return problems


if __name__ == "__main__":
    out = check_data(load_config())
    print("\n".join(out) if out else "data looks complete")
