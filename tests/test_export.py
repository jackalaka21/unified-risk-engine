import json

import numpy as np
import pytest
import torch

from scripts.export_artifacts import export_artifacts
from scripts.train import default_spec, load_checkpoint, train_variant
from serving.bundle import load_bundle
from serving.graph_index import GraphIndex
from serving.scorer import OnlineScorer, raw_kinds_of
from tests.serving_helpers import last_rule_prepared


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


@pytest.fixture(scope="module")
def exported(prep, tmp_path_factory):
    d = tmp_path_factory.mktemp("export")
    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", d / "m.pt")
    train_variant(prep, default_spec(prep, "account_only", 0), "cpu", d / "acct.pt")
    return d, export_artifacts(prep, d / "m.pt", d / "artifacts", fixture=True)


def test_export_writes_every_file_and_meta(exported):
    d, paths = exported
    for key in ("weights", "bundle", "meta", "samples"):
        assert paths[key].exists(), key
    meta = json.loads(paths["meta"].read_text())
    assert meta["fixture"] is True and meta["variant"] == "fraud_only" and meta["config_hash"] and meta["version"]


def test_loaded_bundle_scores_like_the_in_memory_model(prep, exported):
    d, paths = exported
    samples = json.loads(paths["samples"].read_text())
    assert len(samples) > 5
    loaded = load_bundle(d / "artifacts")
    model, heads, _ = load_checkpoint(d / "m.pt", prep, "cpu")
    upto = int((prep.df["TransactionDT"] < prep.periods.test_start).sum())
    direct = OnlineScorer(model, heads, prep.transformer, GraphIndex.from_prepared(prep, upto), raw_kinds_of(prep.df))
    assert np.allclose(loaded.score_batch(samples), direct.score_batch(samples), atol=1e-5)


def test_bundle_weights_load_without_pickle_execution(exported):
    d, paths = exported
    state = torch.load(paths["weights"], map_location="cpu", weights_only=True)
    assert set(state) == {"model", "heads"}


def test_a_checkpoint_without_a_fraud_head_is_refused(prep, exported):
    d, _ = exported
    with pytest.raises(ValueError, match="fraud"):
        export_artifacts(prep, d / "acct.pt", d / "bad", fixture=True)


def test_a_tampered_bundle_is_refused_before_it_is_unpickled(exported, tmp_path):
    import shutil

    d, paths = exported
    copy = tmp_path / "tampered"
    shutil.copytree(d / "artifacts", copy)
    with open(copy / "bundle.pkl", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(copy)
    shutil.copy(d / "artifacts" / "bundle.pkl", copy / "bundle.pkl")
    assert load_bundle(copy)                                       # the untouched copy still loads


def test_the_bundle_records_which_entity_types_the_model_trained_on(tmp_path):
    from pipeline.config import load_config
    from pipeline.prepare import prepare
    from tests.fixtures import fixture_overrides, make_fixture

    prep = prepare(load_config(overrides=fixture_overrides()), df=make_fixture()[0])        # no identity file: no devices
    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", tmp_path / "m.pt")
    paths = export_artifacts(prep, tmp_path / "m.pt", tmp_path / "a", fixture=True)
    assert json.loads(paths["meta"].read_text())["entity_types"] == ["user", "email_domain"]
    assert load_bundle(tmp_path / "a").entities == ("user", "email_domain")
