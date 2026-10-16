import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.graph_builder import EDGE_TYPES
from pipeline.prepare import prepare


@pytest.fixture(scope="module")
def prep(raw_frames):
    from pipeline.config import load_config
    from tests.fixtures import fixture_overrides

    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_node_counts(prep):
    g, df = prep.graph, prep.df
    assert g["transaction"].num_nodes == len(df)
    assert g["user"].num_nodes == df["user"].nunique()
    assert g["email_domain"].num_nodes == df["email"].nunique()
    assert g["device"].num_nodes == df["device"].nunique()


def test_times_are_transaction_times_and_first_seen_for_entities(prep):
    g, df = prep.graph, prep.df
    assert torch.equal(g["transaction"].time, torch.as_tensor(df["TransactionDT"].to_numpy(np.int64)))
    first = df.groupby("user")["TransactionDT"].min()
    names = prep.entity_index["user"]
    for node in (0, 5, 17):
        assert int(g["user"].time[node]) == int(first[names[node]])


def test_rows_without_entities_have_no_edges(prep):
    g, df = prep.graph, prep.df
    ei = g["transaction", "uses_email", "email_domain"].edge_index
    assert ei.shape[1] == df["email"].notna().sum()
    no_email = set(np.flatnonzero(df["email"].isna().to_numpy()))
    assert not no_email & set(ei[0].tolist())
    dv = g["transaction", "uses_device", "device"].edge_index
    assert dv.shape[1] == df["device"].notna().sum() and df["device"].isna().any()


def test_every_row_has_a_user_edge_and_reverse_edges_mirror(prep):
    g = prep.graph
    fwd = g["user", "owns", "transaction"].edge_index
    rev = g["transaction", "rev_owns", "user"].edge_index
    assert fwd.shape[1] == g["transaction"].num_nodes
    assert torch.equal(fwd.flip(0), rev)
    assert set(g.edge_types) == set(EDGE_TYPES)


def test_masks_follow_the_splits(prep):
    g, df = prep.graph, prep.df
    train = g["transaction"].train_mask.numpy()
    assert set(df.loc[train, "user"].map(prep.user_split)) == {"train"}
    assert (df.loc[train, "TransactionDT"] <= prep.periods.t_train).all()
    for name in ("train", "val", "test"):
        m = g["user"][f"{name}_mask"].numpy()
        assert set(prep.account.loc[prep.entity_index["user"][m], "split"]) <= {name}
        assert prep.account.loc[prep.entity_index["user"][m], "eligible"].all()
    assert g.t_train == int(prep.periods.t_train)


def test_features_and_history_are_aligned_with_rows(prep):
    assert prep.x.shape[0] == len(prep.df) == prep.hist.shape[0]
    assert prep.graph["transaction"].x.shape[1] == len(prep.transformer.feature_names)
