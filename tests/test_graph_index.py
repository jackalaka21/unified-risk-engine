import numpy as np
import pytest

from pipeline.graph_builder import EDGE_TYPES
from serving.graph_index import EDGES, GraphIndex, OutOfOrderError

F = 3
Z = np.zeros(F, np.float32)


def idx_with(rows, k2=3):
    ix = GraphIndex(F, fanouts=(15, k2))
    for t, a, u, e, d in rows:
        ix.add(t, a, (u, e, d), np.full(F, float(t), np.float32))
    return ix


def test_edge_table_matches_the_graph_builder():
    assert {et for pair in EDGES.values() for et in pair} == set(EDGE_TYPES)


def test_lookup_returns_the_most_recent_k_up_to_t():
    ix = idx_with([(10, 1, "u", None, None), (20, 1, "u", None, None), (30, 1, "u", None, None), (40, 1, "u", None, None)])
    assert ix.lookup("user", "u", 40, 3) == [1, 2, 3]       # at or before t, most recent three
    assert ix.lookup("user", "u", 25, 3) == [0, 1]
    assert ix.lookup("user", "u", 5, 3) == []
    assert ix.lookup("user", "nobody", 99, 3) == []


def test_subgraph_adds_the_new_transaction_as_the_most_recent_neighbour():
    ix = idx_with([(10, 1, "u", "e", None), (20, 1, "u", None, None)])
    sg = ix.subgraph(("u", "e", None), 30, Z)
    assert sg.tx_rows == [-1, 0, 1] and sg.x.shape == (3, F) and sg.x[1, 0] == 10 and sg.x[2, 0] == 20
    assert sg.entity_keys == {"user": ["u"], "email_domain": ["e"], "device": []}
    assert sg.edges[("user", "owns", "transaction")] == [(0, 0)]
    assert sg.edges[("transaction", "rev_owns", "user")] == [(1, 0), (2, 0), (0, 0)]
    assert sg.edges[("email_domain", "rev_uses_email", "transaction")] == [(0, 0)]
    assert sg.edges[("transaction", "uses_email", "email_domain")] == [(1, 0), (0, 0)]
    assert not sg.edges.get(("device", "rev_uses_device", "transaction"))


def test_only_k_minus_one_old_neighbours_are_kept():
    ix = idx_with([(t, 1, "u", None, None) for t in range(10, 100, 10)])      # nine rows, fan-out 3
    sg = ix.subgraph(("u", None, None), 100, Z)
    assert sg.tx_rows == [-1, 7, 8]                                          # two old ones and the new one


def test_unseen_and_missing_keys():
    ix = idx_with([(10, 1, "u", "e", "d")])
    sg = ix.subgraph(("new_user", None, np.nan), 20, Z)
    assert sg.entity_keys == {"user": ["new_user"], "email_domain": [], "device": []}
    assert sg.tx_rows == [-1]
    assert sg.edges[("user", "owns", "transaction")] == [(0, 0)] and sg.edges[("transaction", "rev_owns", "user")] == [(0, 0)]
    assert not sg.edges.get(("transaction", "uses_email", "email_domain"))
    assert not sg.edges.get(("transaction", "uses_device", "device"))


def test_out_of_order_arrivals_are_rejected_but_equal_times_are_fine():
    ix = idx_with([(10, 1, "u", None, None)])
    with pytest.raises(OutOfOrderError):
        ix.add(5, 1, ("u", None, None), Z)
    ix.add(10, 1, ("u", None, None), Z)
    assert len(ix) == 2


def test_history_by_hand_with_strictly_earlier_semantics():
    ix = idx_with([(0, 10.0, "u", "e", "A"), (100, 20.0, "u", "e", "A")])
    h = ix.history(("u", "e", "A"), 200)
    assert h["user_n_prior"] == 2 and h["user_n_1d"] == 2 and h["user_amt_mean"] == 15.0
    assert h["user_amt_std"] == pytest.approx(50 ** 0.5) and h["user_secs_since_prev"] == 100
    assert h["user_distinct_devices"] == 1 and h["user_distinct_emails"] == 1
    tie = ix.history(("u", "e", "A"), 100)                                    # the row at t=100 does not count itself
    assert tie["user_n_prior"] == 1 and np.isnan(tie["user_amt_std"]) and tie["user_secs_since_prev"] == 100


def test_history_for_unseen_and_missing_entities():
    ix = idx_with([(0, 10.0, "u", "e", "A")])
    h = ix.history(("zzz", None, None), 50)
    assert h["user_n_prior"] == 0 and np.isnan(h["user_amt_mean"]) and h["user_distinct_devices"] == 0
    assert np.isnan(h["email_n_prior"]) and np.isnan(h["device_n_prior"])


def test_pickling_does_not_carry_unused_capacity():
    import pickle

    ix = idx_with([(1, 1, "u", None, None)])
    again = pickle.loads(pickle.dumps(ix))
    assert again._x.shape[0] == 1
    again.add(2, 1, ("u", None, None), Z)                            # growth still works after a round trip
    assert len(again) == 2 and again.lookup("user", "u", 9, 3) == [0, 1]
