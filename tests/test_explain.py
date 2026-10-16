import numpy as np
import pytest
import torch

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.explain import (deletion_fidelity, explain_transaction, explain_user, occlusion_importance,
                             pyg_fidelity, top_edges, top_features, topk_overlap)
from scripts.train import default_spec, train_variant
from tests.fixtures import fixture_overrides


class Linear(torch.nn.Module):
    """Toy 'model': logit of node i is x[i] @ w, so occlusion importance is exactly |x_j * w_j|."""

    def __init__(self, w):
        super().__init__()
        self.w = torch.tensor(w, dtype=torch.float32)

    def forward(self, x_dict, edge_index_dict):
        return x_dict["transaction"] @ self.w


def test_occlusion_importance_matches_a_linear_model_exactly():
    x = {"transaction": torch.tensor([[1.0, 2.0, 3.0]])}
    imp = occlusion_importance(Linear([0.5, -1.0, 0.1]), x, {})
    assert np.allclose(imp, [0.5, 2.0, 0.3])


def test_top_features_orders_by_importance():
    assert top_features(np.array([0.1, 0.9, 0.5]), ["a", "b", "c"], 2) == [("b", 0.9), ("c", 0.5)]


def test_topk_overlap_is_one_for_identical_and_lower_for_different_rankings():
    a = np.array([5.0, 4.0, 3.0, 0.1, 0.0])
    b = np.array([0.0, 0.1, 3.0, 4.0, 5.0])
    assert topk_overlap([a, a, a], k=2) == 1.0 and topk_overlap([a, b], k=2) == 0.0


def test_deleting_the_top_features_hurts_more_than_deleting_random_ones():
    x = {"transaction": torch.tensor([[3.0, 2.0, 0.1, 0.1, 0.1, 0.1]])}
    model = Linear([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    imp = occlusion_importance(model, x, {})
    f = deletion_fidelity(model, x, {}, imp, k=2)
    assert f["drop_top_k"] > f["drop_random_k"]


@pytest.fixture(scope="module")
def trained(raw_frames):
    prep = prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])
    return prep, train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu")


@pytest.mark.slow
def test_explaining_a_transaction_returns_one_importance_per_feature(trained):
    prep, res = trained
    mode = prep.cfg["explain"]["mode"]
    row = int(np.flatnonzero(prep.graph["transaction"].test_mask.numpy())[0])
    out = explain_transaction(prep, res.model, res.heads, row, mode, seed=0, device="cpu")
    assert out["importance"].shape == (len(prep.transformer.feature_names),)
    assert np.isfinite(out["importance"]).all()


@pytest.mark.slow
def test_hetero_mode_reports_edges_pyg_fidelity_and_a_user_explanation(trained):
    prep, res = trained
    if prep.cfg["explain"]["mode"] != "hetero":
        pytest.skip("the spike chose occlusion mode")
    row = int(np.flatnonzero(prep.graph["transaction"].test_mask.numpy())[0])
    out = explain_transaction(prep, res.model, res.heads, row, "hetero", seed=0, device="cpu")
    edges = top_edges(out, 5, prep.entity_index)
    assert 0 < len(edges) <= 5 and all(len(e) == 4 and np.isfinite(e[3]) for e in edges)
    fid = pyg_fidelity(out)
    assert fid is None or (np.isfinite(fid["fidelity_plus"]) and np.isfinite(fid["fidelity_minus"]))
    user = int(np.flatnonzero(prep.graph["user"].test_mask.numpy())[0])
    assert len(explain_user(prep, res.model, res.heads, user, seed=0, device="cpu")) > 0


def test_occlusion_mode_always_works_on_the_trained_model(trained):
    prep, res = trained
    row = int(np.flatnonzero(prep.graph["transaction"].test_mask.numpy())[0])
    out = explain_transaction(prep, res.model, res.heads, row, "occlusion", seed=0, device="cpu")
    assert out["importance"].shape == (len(prep.transformer.feature_names),) and np.isfinite(out["logit"])
