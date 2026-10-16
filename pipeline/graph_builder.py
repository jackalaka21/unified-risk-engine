"""Build the time-stamped heterogeneous graph."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from pipeline.splits import fraud_masks

I64MAX = np.iinfo(np.int64).max
EDGE_TYPES = [
    ("user", "owns", "transaction"), ("transaction", "rev_owns", "user"),
    ("transaction", "uses_email", "email_domain"), ("email_domain", "rev_uses_email", "transaction"),
    ("transaction", "uses_device", "device"), ("device", "rev_uses_device", "transaction"),
]
ENTITIES = (("user", "user", "owns", "rev_owns", True),
            ("email_domain", "email", "uses_email", "rev_uses_email", False),
            ("device", "device", "uses_device", "rev_uses_device", False))


def build_graph(df, x, periods, user_split, account, entity_codes=None):
    n = len(df)
    times = df["TransactionDT"].to_numpy(np.int64)
    data = HeteroData()
    data["transaction"].x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    data["transaction"].y = torch.as_tensor(df["isFraud"].to_numpy(np.int64))
    data["transaction"].time = torch.as_tensor(times)
    masks = fraud_masks(df, periods, user_split)
    for name, mask in masks.items():
        data["transaction"][f"{name}_mask"] = torch.as_tensor(mask)

    rows_all = np.arange(n)
    index = {}
    for node_type, col, fwd_name, rev_name, owns in ENTITIES:
        codes, uniques = pd.factorize(df[col])
        index[node_type] = pd.Index(uniques)
        n_ent = len(uniques)
        data[node_type].num_nodes = n_ent
        valid = codes >= 0
        first = np.full(n_ent, I64MAX, dtype=np.int64)
        np.minimum.at(first, codes[valid], times[valid])
        data[node_type].time = torch.as_tensor(first)
        rows = rows_all[valid]
        ent = torch.as_tensor(codes[valid], dtype=torch.long)
        tx = torch.as_tensor(rows, dtype=torch.long)
        if owns:
            data[node_type, fwd_name, "transaction"].edge_index = torch.stack([ent, tx])
            data["transaction", rev_name, node_type].edge_index = torch.stack([tx, ent])
        else:
            data["transaction", fwd_name, node_type].edge_index = torch.stack([tx, ent])
            data[node_type, rev_name, "transaction"].edge_index = torch.stack([ent, tx])

    table = account.reindex(index["user"])
    data["user"].y = torch.as_tensor(table["label"].to_numpy(np.int64))
    eligible = table["eligible"].to_numpy(bool)
    for name in ("train", "val", "test"):
        data["user"][f"{name}_mask"] = torch.as_tensor((table["split"].to_numpy() == name) & eligible)
    data.t_train = int(periods.t_train)
    return data, index
