import numpy as np
import pytest
import torch

from pipeline.graph_builder import EDGE_TYPES
from pipeline.prepare import prepare


@pytest.fixture(scope="module")
def prep(raw_frames):
    from pipeline.config import load_config
    from tests.fixtures import fixture_overrides, make_fixture

    tx, identity = make_fixture()                                   # merged, so device nodes and edges really exist
    return prepare(load_config(overrides=fixture_overrides()), df=tx.merge(identity, on="TransactionID", how="left"))


def test_node_counts(prep):
    g, df = prep.graph, prep.df
    assert g["transaction"].num_nodes == len(df)
    assert g["user"].num_nodes == df["user"].nunique()
    assert g["email_domain"].num_nodes == df["email"].nunique()
    assert g["device"].num_nodes == df["device"].nunique()


def test_graph_time_is_arrival_order_so_equal_timestamps_never_see_later_rows(prep):
    g, df = prep.graph, prep.df
    assert torch.equal(g["transaction"].time, torch.arange(len(df)))             # strictly increasing: ties broken by row order
    first_row = df.reset_index().groupby("user")["index"].min()
    names = prep.entity_index["user"]
    for node in (0, 5, 17):
        assert int(g["user"].time[node]) == int(first_row[names[node]])
    assert g.t_train_rank == int((df["TransactionDT"] <= prep.periods.t_train).sum()) - 1


def test_rank_time_breaks_ties_between_equal_timestamps(raw_frames):
    from pipeline.config import load_config
    from tests.fixtures import fixture_overrides

    tx = raw_frames[0].copy()
    tx["TransactionDT"] = (tx["TransactionDT"] // 3600) * 3600                  # many transactions share a timestamp
    p = prepare(load_config(overrides=fixture_overrides()), df=tx)
    assert p.df["TransactionDT"].duplicated().any()
    assert (p.graph["transaction"].time[1:] > p.graph["transaction"].time[:-1]).all()


def test_rows_without_entities_have_no_edges(prep):
    g, df = prep.graph, prep.df
    ei = g["transaction", "uses_email", "email_domain"].edge_index
    assert ei.shape[1] == df["email"].notna().sum()
    no_email = set(np.flatnonzero(df["email"].isna().to_numpy()))
    assert not no_email & set(ei[0].tolist())
    dv = g["transaction", "uses_device", "device"].edge_index
    assert df["device"].notna().any() and df["device"].isna().any()      # a mix of rows with and without a device
    assert dv.shape[1] == df["device"].notna().sum()
    no_device = set(np.flatnonzero(df["device"].isna().to_numpy()))
    assert not no_device & set(dv[0].tolist())


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


def test_graph_features_are_aligned_with_rows_and_use_the_configured_embargo(prep):
    from pipeline.account_identity import ACCT_COLUMNS
    from pipeline.graph_features import LABEL_BASED, LABEL_FREE, LABEL_WINDOWED

    assert list(prep.gfeat.columns) == LABEL_FREE + LABEL_BASED + LABEL_WINDOWED and len(prep.gfeat) == len(prep.df)
    assert list(prep.acct.columns) == ACCT_COLUMNS and prep.acct.index.equals(prep.df.index)
    assert prep.gfeat.index.equals(prep.df.index)
    # nothing known in the first embargo window: every labelled count is zero for the very first rows
    assert (prep.gfeat["user_lbl_n_known"].iloc[:3] == 0).all()
