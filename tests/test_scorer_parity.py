import numpy as np
import pandas as pd
import pytest

from scripts.train import make_loader, predict_scores
from serving.graph_index import GraphIndex, OutOfOrderError, keys_from_frame
from serving.scorer import TabularOnlineScorer, payload_of, raw_kinds_of
from tests.serving_helpers import HAS_TEMPORAL, last_rule_prepared, make_scorer

needs_temporal = pytest.mark.skipif(not HAS_TEMPORAL, reason="temporal sampling needs pyg-lib")


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


def label_offline(prep, nt, node_id):
    return int(node_id) if nt == "transaction" else prep.entity_index[nt][int(node_id)]


@needs_temporal
def test_index_subgraph_equals_the_neighbor_loader_neighbourhood(prep):
    df, p = prep.df, len(prep.df) // 2
    rows = np.arange(p, p + 40)
    batch = next(iter(make_loader(prep, "transaction", rows, len(rows), False)))
    offline = {i: set() for i in range(len(rows))}
    for et in batch.edge_types:
        s, _, d = et
        ei = batch[et].edge_index
        for seed, a, b in zip(batch[d].batch[ei[1]].tolist(), batch[s].n_id[ei[0]].tolist(), batch[d].n_id[ei[1]].tolist()):
            offline[seed].add((et, label_offline(prep, s, a), label_offline(prep, d, b)))
    index = GraphIndex.from_prepared(prep, p)
    n_checked = 0
    for j, row in enumerate(rows):
        keys = (df["user"].iat[row], df["email"].iat[row], df["device"].iat[row])
        sg = index.subgraph(keys, int(df["TransactionDT"].iat[row]), prep.x[row])
        online = set()
        for et, pairs in sg.edges.items():
            for a, b in pairs:
                lab = lambda nt, i: row if nt == "transaction" and sg.tx_rows[i] == -1 else (
                    sg.tx_rows[i] if nt == "transaction" else sg.entity_keys[nt][i])
                online.add((et, lab(et[0], a), lab(et[2], b)))
        assert online == offline[j], f"row {row}: only online {online - offline[j]}, only offline {offline[j] - online}"
        n_checked += len(online)
        index.add(int(df["TransactionDT"].iat[row]), float(df["TransactionAmt"].iat[row]), keys, prep.x[row])
    assert n_checked > 100


@needs_temporal
def test_online_scores_equal_offline_scores_under_the_last_rule(prep):
    p = len(prep.df) // 2
    rows = np.arange(p, p + 60)
    scorer = make_scorer(prep, p)
    online = np.array(scorer.score_batch([payload_of(prep.df, int(i)) for i in rows]))
    offline = predict_scores(prep, scorer.model, scorer.heads, "fraud", rows, "cpu")
    np.testing.assert_allclose(online, offline, atol=1e-4)


def test_batch_scoring_equals_one_by_one(prep):
    p = len(prep.df) // 2
    payloads = [payload_of(prep.df, i) for i in range(p, p + 20)]
    batch = make_scorer(prep, p).score_batch(payloads)
    single = make_scorer(prep, p)
    assert np.allclose(batch, [single.score(x) for x in payloads], atol=1e-5)


def test_payload_order_does_not_change_which_score_goes_with_which_request(prep):
    p = len(prep.df) // 2
    payloads = [payload_of(prep.df, i) for i in range(p, p + 10)]
    forward = make_scorer(prep, p).score_batch(payloads)
    shuffled = make_scorer(prep, p).score_batch(payloads[::-1])
    assert np.allclose(forward, shuffled[::-1], atol=1e-5)


def test_payload_number_types_do_not_change_the_entity_keys(prep):
    scorer = make_scorer(prep, 100)
    pay = payload_of(prep.df, 150)
    variants = [pay, {**pay, "card1": float(pay["card1"])}, {**pay, "addr1": None if pay["addr1"] is None else int(pay["addr1"])}]
    users = [scorer._frame([v])["user"].iat[0] for v in variants]
    assert len(set(users)) == 1


def test_a_card_never_seen_with_no_email_or_device_scores_cleanly(prep):
    scorer = make_scorer(prep, len(prep.df) // 2)
    pay = {"TransactionDT": int(prep.df["TransactionDT"].max()) + 10, "TransactionAmt": 25.0, "ProductCD": "W", "card1": 999999}
    assert 0.0 <= scorer.score(pay) <= 1.0


def test_old_requests_are_rejected_without_changing_the_index(prep):
    p = len(prep.df) // 2
    scorer = make_scorer(prep, p)
    n = len(scorer.index)
    with pytest.raises(OutOfOrderError):
        scorer.score(payload_of(prep.df, 0))
    with pytest.raises(OutOfOrderError):
        scorer.score_batch([payload_of(prep.df, p + 5), payload_of(prep.df, 0)])
    assert len(scorer.index) == n


def test_late_requests_can_be_clamped_to_the_newest_time(prep):
    scorer = make_scorer(prep, len(prep.df) // 2, late="clamp")
    before = scorer.index.last_time
    assert 0.0 <= scorer.score(payload_of(prep.df, 0)) <= 1.0
    assert scorer.index.last_time == before


def test_missing_required_fields_are_named(prep):
    scorer = make_scorer(prep, 100)
    with pytest.raises(ValueError, match="TransactionDT"):
        scorer.score({"TransactionAmt": 5.0})
    with pytest.raises(ValueError, match="numbers"):
        scorer.score({"TransactionDT": "soon", "TransactionAmt": 5.0})


def test_tabular_online_history_matches_the_offline_history_features(prep):
    p = len(prep.df) // 2
    index = GraphIndex.from_prepared(prep, p)
    df = prep.df
    for j in range(p, p + 80):
        keys = (df["user"].iat[j], df["email"].iat[j], df["device"].iat[j])
        got = pd.Series(index.history(keys, int(df["TransactionDT"].iat[j])))[prep.hist.columns]
        np.testing.assert_allclose(got.to_numpy(float), prep.hist.iloc[j].to_numpy(float), rtol=1e-6, equal_nan=True)
        index.add(int(df["TransactionDT"].iat[j]), float(df["TransactionAmt"].iat[j]), keys, prep.x[j])


def test_tabular_scorer_matches_the_offline_baseline_predictions(prep):
    from scripts.baseline import fit_hgb

    p = len(prep.df) // 2
    X = np.hstack([prep.x, prep.hist.to_numpy(np.float32)])
    model = fit_hgb(X[:p], prep.df["isFraud"].to_numpy()[:p], seed=0, max_iter=20)
    scorer = TabularOnlineScorer(model, prep.transformer, GraphIndex.from_prepared(prep, p), raw_kinds_of(prep.df),
                                 hist_columns=list(prep.hist.columns))
    rows = list(range(p, p + 40))
    online = scorer.score_batch([payload_of(prep.df, i) for i in rows])
    np.testing.assert_allclose(online, model.predict_proba(X[rows])[:, 1], atol=1e-5)
