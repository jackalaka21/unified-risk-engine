import threading

import pytest
from fastapi.testclient import TestClient

from serving.app import create_app
from serving.scorer import payload_of
from tests.serving_helpers import last_rule_prepared, make_scorer


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


@pytest.fixture()
def client_and_payloads(prep):
    p = len(prep.df) // 2
    scorer = make_scorer(prep, p, version="test-v1")
    return TestClient(create_app(scorer)), [payload_of(prep.df, i) for i in range(p, p + 40)]


def test_health_and_version(client_and_payloads):
    client, _ = client_and_payloads
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/version").json()["model_version"] == "test-v1"


def test_score_returns_a_probability_and_grows_the_index(client_and_payloads):
    client, payloads = client_and_payloads
    n0 = client.get("/health").json()["indexed_transactions"]
    r = client.post("/score", json=payloads[0])
    assert r.status_code == 200 and 0.0 <= r.json()["fraud_probability"] <= 1.0 and r.json()["model_version"] == "test-v1"
    assert client.get("/health").json()["indexed_transactions"] == n0 + 1


def test_batch_returns_one_probability_per_transaction(client_and_payloads):
    client, payloads = client_and_payloads
    r = client.post("/score_batch", json={"transactions": payloads[:8]})
    assert r.status_code == 200 and len(r.json()["fraud_probabilities"]) == 8


@pytest.mark.parametrize("body", [
    {"TransactionAmt": 5.0},                                  # missing time
    {"TransactionDT": 100},                                    # missing amount
    {"TransactionDT": 100, "TransactionAmt": -1.0},            # negative amount
    {"TransactionDT": "soon", "TransactionAmt": 5.0},          # wrong type
    {},
])
def test_malformed_transactions_get_422(client_and_payloads, body):
    client, _ = client_and_payloads
    assert client.post("/score", json=body).status_code == 422


@pytest.mark.parametrize("body", [{"transactions": []}, {"transactions": [{"TransactionDT": 1, "TransactionAmt": 1.0}] * 257}, {}])
def test_bad_batches_get_422(client_and_payloads, body):
    client, _ = client_and_payloads
    assert client.post("/score_batch", json=body).status_code == 422


def test_unknown_extra_fields_are_ignored(client_and_payloads):
    client, payloads = client_and_payloads
    assert client.post("/score", json={**payloads[0], "not_a_column": 1}).status_code == 200


def test_old_requests_get_409_and_do_not_change_the_index(client_and_payloads, prep):
    client, payloads = client_and_payloads
    n0 = client.get("/health").json()["indexed_transactions"]
    r = client.post("/score", json=payload_of(prep.df, 0))
    assert r.status_code == 409 and "older" in r.json()["detail"]
    assert client.get("/health").json()["indexed_transactions"] == n0


def test_service_without_a_model_returns_503():
    client = TestClient(create_app(scorer=None))              # no lifespan: nothing is loaded
    assert client.get("/health").status_code == 503
    assert client.post("/score", json={"TransactionDT": 1, "TransactionAmt": 1.0}).status_code == 503


def test_concurrent_requests_are_serialised_safely(client_and_payloads):
    client, payloads = client_and_payloads
    n0 = client.get("/health").json()["indexed_transactions"]
    same_time = [{**payloads[-1]} for _ in range(16)]         # equal times never count as out of order
    codes = []
    threads = [threading.Thread(target=lambda b=b: codes.append(client.post("/score", json=b).status_code)) for b in same_time]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes == [200] * 16 and client.get("/health").json()["indexed_transactions"] == n0 + 16


@pytest.mark.parametrize("amount", ["Infinity", "-Infinity", "NaN"])
def test_non_finite_numbers_get_422_and_leave_the_service_healthy(client_and_payloads, amount):
    client, payloads = client_and_payloads
    n0 = client.get("/health").json()["indexed_transactions"]
    body = ('{"TransactionDT": %d, "TransactionAmt": %s}' % (payloads[0]["TransactionDT"], amount)).encode()
    r = client.post("/score", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert client.get("/health").json()["indexed_transactions"] == n0
    assert client.post("/score", json=payloads[0]).status_code == 200
