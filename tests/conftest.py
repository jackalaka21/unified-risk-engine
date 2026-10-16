import pytest

from pipeline.config import load_config
from tests.fixtures import fixture_overrides, make_fixture


@pytest.fixture(scope="session")
def raw_frames():
    return make_fixture()


@pytest.fixture()
def cfg():
    return load_config(overrides=fixture_overrides())
