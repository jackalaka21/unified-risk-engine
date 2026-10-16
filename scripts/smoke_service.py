"""Smoke test for a running scoring service (works with httpx.Client and FastAPI's TestClient)."""
from __future__ import annotations

import argparse
import json
import time


def smoke(client, payloads) -> dict:
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "ok", f"health: {health.status_code}"
    assert client.get("/version").json()["model_version"]
    scored = 0
    for body in payloads[:5]:
        r = client.post("/score", json=body)
        assert r.status_code == 200, f"/score returned {r.status_code}: {r.text}"
        assert 0.0 <= r.json()["fraud_probability"] <= 1.0
        scored += 1
    r = client.post("/score_batch", json={"transactions": payloads[5:15]})
    assert r.status_code == 200 and len(r.json()["fraud_probabilities"]) == len(payloads[5:15])
    scored += len(payloads[5:15])
    malformed = client.post("/score", json={"TransactionAmt": 1.0}).status_code == 422
    late = client.post("/score", json={**payloads[0], "TransactionDT": 0}).status_code == 409
    assert malformed, "malformed request was not rejected with 422"
    assert late, "request older than the index was not rejected with 409"
    return {"scored": scored, "malformed_rejected": malformed, "late_rejected": late}


def main(argv=None):
    import httpx

    p = argparse.ArgumentParser(description="Smoke-test a running scoring service.")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--samples", default="artifacts/sample_requests.json")
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /health")
    a = p.parse_args(argv)
    payloads = json.load(open(a.samples, encoding="utf-8"))
    with httpx.Client(base_url=a.url, timeout=30) as client:
        deadline = time.time() + a.wait
        while True:
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.time() > deadline:
                raise SystemExit(f"service at {a.url} did not become healthy within {a.wait}s")
            time.sleep(1)
        print(smoke(client, payloads))


if __name__ == "__main__":
    main()
