import json

import pytest
from fastapi.testclient import TestClient

from scripts.smoke_service import smoke
from serving.app import create_app
from serving.scorer import payload_of
from tests.serving_helpers import last_rule_prepared, make_scorer


def test_smoke_passes_against_a_healthy_service_and_fails_against_a_broken_one():
    prep = last_rule_prepared()
    p = len(prep.df) // 2
    payloads = [payload_of(prep.df, i) for i in range(p, p + 15)]
    result = smoke(TestClient(create_app(make_scorer(prep, p))), payloads)
    assert result["scored"] >= 14 and result["malformed_rejected"] and result["late_rejected"]
    with pytest.raises(AssertionError):
        smoke(TestClient(create_app(scorer=None)), payloads)
