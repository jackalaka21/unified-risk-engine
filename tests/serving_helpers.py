import importlib.util

import torch

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.train import build_modules, warm_up
from serving.graph_index import GraphIndex
from serving.scorer import OnlineScorer, raw_kinds_of
from tests.fixtures import fixture_overrides, make_fixture

HAS_TEMPORAL = importlib.util.find_spec("pyg_lib") is not None


def last_rule_prepared():
    overrides = fixture_overrides()
    overrides["train"]["temporal_strategy"] = "last"
    tx, identity = make_fixture()
    frame = tx.merge(identity, on="TransactionID", how="left")      # device nodes and edges need the identity file
    return prepare(load_config(overrides=overrides), df=frame)


def make_scorer(prep, upto, seed=0, **kwargs):
    """A scorer with a seeded, untrained model: parity does not need training, only identical weights."""
    torch.manual_seed(seed)
    model, heads = build_modules(prep, "cpu")
    warm_up(model, heads, prep, "cpu")
    index = GraphIndex.from_prepared(prep, upto)
    return OnlineScorer(model, heads, prep.transformer, index, raw_kinds_of(prep.df), **kwargs)
