import pytest
import torch
import torch.nn.functional as F

from models.gnn_backbone import HeteroGNN
from models.losses import focal_loss
from models.multi_task_heads import TaskHeads
from pipeline.graph_builder import EDGE_TYPES


def tiny_batch(n_tx=6, n_user=3, n_email=2, n_device=0, f=5):
    x = {"transaction": torch.randn(n_tx, f)}
    tx = torch.arange(n_tx)
    user = tx % n_user
    email = tx % n_email
    edges = {
        ("user", "owns", "transaction"): torch.stack([user, tx]),
        ("transaction", "rev_owns", "user"): torch.stack([tx, user]),
        ("transaction", "uses_email", "email_domain"): torch.stack([tx, email]),
        ("email_domain", "rev_uses_email", "transaction"): torch.stack([email, tx]),
    }
    return x, edges, {"transaction": n_tx, "user": n_user, "email_domain": n_email, "device": n_device}


@pytest.mark.parametrize("conv", ["gatv2", "sage"])
def test_forward_shapes_for_each_conv_including_an_empty_node_type(conv):
    x, edges, nn_ = tiny_batch()
    model = HeteroGNN(5, EDGE_TYPES, hidden=16, heads=2, layers=2, dropout=0.0, conv=conv)
    h = model(x, edges, nn_)
    assert h["transaction"].shape == (6, 16) and h["user"].shape == (3, 16)
    assert h["device"].shape == (0, 16)
    assert torch.isfinite(h["transaction"]).all()


def test_gradients_flow_to_the_transaction_projection_and_type_vectors():
    x, edges, nn_ = tiny_batch()
    model = HeteroGNN(5, EDGE_TYPES, hidden=8, heads=2, dropout=0.0)
    h = model(x, edges, nn_)
    h["transaction"].sum().backward()
    assert model.tx_proj.weight.grad is not None and model.type_vec["user"].grad is not None


def test_tabular_path_ignores_edges_and_has_the_same_width():
    x, _, _ = tiny_batch()
    model = HeteroGNN(5, EDGE_TYPES, hidden=8, heads=2, dropout=0.0)
    assert model.encode_tabular(x["transaction"]).shape == (6, 8)


def test_heads_return_one_logit_per_node():
    heads = TaskHeads(8, dropout=0.0)
    assert heads.fraud_logits(torch.randn(4, 8)).shape == (4,)
    assert heads.account_logits(torch.randn(3, 8)).shape == (3,)


def test_focal_loss_reduces_to_scaled_cross_entropy_at_gamma_zero():
    logits, y = torch.randn(50), (torch.rand(50) > 0.5).float()
    got = focal_loss(logits, y, alpha=0.5, gamma=0.0)
    assert torch.allclose(got, 0.5 * F.binary_cross_entropy_with_logits(logits, y))


def test_alpha_weights_positives_and_negatives():
    logits = torch.tensor([0.3, 0.3])
    pos = focal_loss(logits[:1], torch.tensor([1.0]), alpha=0.75, gamma=0.0)
    neg = focal_loss(logits[1:], torch.tensor([0.0]), alpha=0.75, gamma=0.0)
    assert pos == pytest.approx(0.75 * F.binary_cross_entropy_with_logits(logits[:1], torch.tensor([1.0])).item())
    assert neg == pytest.approx(0.25 * F.binary_cross_entropy_with_logits(logits[1:], torch.tensor([0.0])).item())


def test_gamma_downweights_easy_examples_and_none_reduction_keeps_shape():
    logits, y = torch.tensor([6.0, -6.0]), torch.tensor([1.0, 0.0])  # both confidently correct
    assert focal_loss(logits, y, 0.75, 2.0).item() < focal_loss(logits, y, 0.75, 0.0).item()
    assert focal_loss(logits, y, 0.75, 2.0, reduction="none").shape == (2,)
