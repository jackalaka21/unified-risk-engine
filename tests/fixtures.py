"""IEEE-CIS-shaped synthetic data with a planted fraud signal, for tests only (never real results)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

EMAILS = np.array(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "protonmail.com", None], dtype=object)
P_LEGIT = np.array([0.45, 0.20, 0.15, 0.10, 0.02, 0.08])
P_FRAUD = np.array([0.15, 0.10, 0.10, 0.25, 0.30, 0.10])


def make_fixture(n_users=500, n_tx=9000, days=90, seed=0, with_identity=True):
    rng = np.random.default_rng(seed)
    horizon = days * 86400
    times = np.sort(rng.choice(horizon, size=n_tx, replace=False)).astype(np.int64)  # unique timestamps
    user = rng.integers(0, n_users, size=n_tx)
    compromised = rng.random(n_users) < 0.12
    start = rng.uniform(0.15, 0.7, size=n_users) * horizon
    in_window = compromised[user] & (times >= start[user])
    y = (rng.random(n_tx) < np.where(in_window, 0.45, 0.01)).astype(int)

    amount = np.exp(rng.normal(3.5, 0.8, size=n_tx) + 0.9 * y)
    email_idx = np.where(y == 1, rng.choice(6, size=n_tx, p=P_FRAUD), rng.choice(6, size=n_tx, p=P_LEGIT))
    email = EMAILS[email_idx]
    tx = pd.DataFrame({
        "TransactionID": 2_987_000 + np.arange(n_tx),
        "isFraud": y,
        "TransactionDT": times,
        "TransactionAmt": amount.round(2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n_tx, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        "card1": 1000 + user,
        "card2": rng.choice([111.0, 222.0, 333.0, np.nan], size=n_tx),
        "card3": 150.0,
        "card4": rng.choice(["visa", "mastercard", "discover"], size=n_tx),
        "card5": rng.choice([226.0, 224.0, 166.0], size=n_tx),
        "card6": rng.choice(["debit", "credit"], size=n_tx),
        "addr1": np.where(rng.random(n_tx) < 0.08, np.nan, 100 + user % 50),
        "addr2": 87.0,
        "dist1": np.where(rng.random(n_tx) < 0.7, np.nan, rng.exponential(50, size=n_tx)),
        "P_emaildomain": email,
        "R_emaildomain": np.where(rng.random(n_tx) < 0.7, None, "gmail.com"),
    })
    for j in range(1, 15):
        tx[f"C{j}"] = rng.poisson(1 + 3 * y).astype(float)
    for j in range(1, 16):
        tx[f"D{j}"] = np.where(rng.random(n_tx) < 0.4, np.nan, rng.exponential(30, size=n_tx))
    for j in range(1, 10):
        tx[f"M{j}"] = rng.choice(["T", "F", None], size=n_tx, p=[0.5, 0.3, 0.2])
    for j in range(1, 41):
        shift = 0.6 * y if j <= 10 else 0.0
        tx[f"V{j}"] = np.where(rng.random(n_tx) < (0.8 if j > 30 else 0.1), np.nan, rng.normal(0, 1, size=n_tx) + shift)

    identity = None
    if with_identity:
        has = rng.random(n_tx) < 0.6
        shared = rng.random(n_tx) < np.where(y == 1, 0.6, 0.02)
        device = np.where(shared, [f"shared_{k}" for k in rng.integers(0, 8, size=n_tx)],
                          [f"dev_{u}_{k}" for u, k in zip(user, rng.integers(0, 2, size=n_tx))])
        identity = pd.DataFrame({
            "TransactionID": tx["TransactionID"][has].to_numpy(),
            "DeviceType": rng.choice(["desktop", "mobile"], size=int(has.sum())),
            "DeviceInfo": device[has],
            "id_01": rng.normal(0, 5, size=int(has.sum())),
        })
    return tx, identity


def write_fixture(directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tx, identity = make_fixture(**kwargs)
    tx.to_csv(directory / "train_transaction.csv", index=False)
    if identity is not None:
        identity.to_csv(directory / "train_identity.csv", index=False)
    return directory


def fixture_overrides():
    return {
        "splits": {"embargo_days": 1},
        "account": {"min_positives": 3},
        "model": {"hidden": 16, "heads": 2},
        "train": {"temporal": importlib.util.find_spec("pyg_lib") is not None, "num_neighbors": [5, 3], "batch_fraud": 256, "batch_account": 128, "epochs": 1,
                  "patience": 2, "device": "cpu", "max_steps_per_epoch": 4},
        "evaluation": {"bootstrap_samples": 50},
        "explain": {"epochs": 10, "fidelity_sample": 5, "n_explained": 2},
    }
