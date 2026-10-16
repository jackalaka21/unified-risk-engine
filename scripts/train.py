"""Train the GNN variants with alternating transaction/user mini-batches and temporal neighbour sampling."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from models.gnn_backbone import HeteroGNN
from models.losses import focal_loss
from models.multi_task_heads import TaskHeads
from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.metrics import pr_auc

FRAUD_VARIANTS = ("multitask", "fraud_only", "no_graph")
ACCOUNT_VARIANTS = ("multitask", "account_only")


@dataclass
class RunSpec:
    variant: str
    seed: int
    fraud_train_idx: np.ndarray
    fraud_val_idx: np.ndarray
    account_train_idx: np.ndarray
    account_val_idx: np.ndarray
    epochs: int | None = None


@dataclass
class TrainResult:
    model: torch.nn.Module
    heads: torch.nn.Module
    best_epoch: int
    best_score: float
    history: list


def default_spec(prepared, variant: str, seed: int) -> RunSpec:
    g = prepared.graph

    def idx(node_type, name):
        return np.flatnonzero(g[node_type][f"{name}_mask"].numpy())

    return RunSpec(variant, seed, idx("transaction", "train"), idx("transaction", "val"),
                   idx("user", "train"), idx("user", "val"))


def _times(prepared, node_type, idx):
    if node_type == "transaction":
        return prepared.graph["transaction"].time.numpy()[idx]
    return np.full(len(idx), int(prepared.graph.t_train_rank), dtype=np.int64)


def make_loader(prepared, node_type, idx, batch_size, shuffle):
    t = prepared.cfg["train"]
    idx = np.asarray(idx)
    kwargs = dict(data=prepared.graph, num_neighbors=list(t["num_neighbors"]),
                  input_nodes=(node_type, torch.as_tensor(idx, dtype=torch.long)),
                  batch_size=batch_size, shuffle=shuffle)
    if t["temporal"]:
        kwargs.update(time_attr="time", temporal_strategy=t["temporal_strategy"],
                      input_time=torch.as_tensor(_times(prepared, node_type, idx), dtype=torch.long))
    return NeighborLoader(**kwargs)


def build_modules(prepared, device):
    m, g = prepared.cfg["model"], prepared.graph
    model = HeteroGNN(g["transaction"].x.shape[1], list(g.edge_types), m["hidden"], m["heads"],
                      m["layers"], m["dropout"], m["conv"])
    return model.to(device), TaskHeads(m["hidden"], m["dropout"]).to(device)


def _graph_logits(model, heads, batch, node_type, kind, device):
    batch = batch.to(device)
    num_nodes = {nt: batch[nt].num_nodes for nt in batch.node_types}
    h = model(batch.x_dict, batch.edge_index_dict, num_nodes)
    bs = batch[node_type].batch_size
    z = h[node_type][:bs]
    logits = heads.fraud_logits(z) if kind == "fraud" else heads.account_logits(z)
    return logits, batch[node_type].y[:bs].float()


def warm_up(model, heads, prepared, device):
    """PyG's lazy layers need one forward pass before parameters exist (optimiser, load_state_dict)."""
    g = prepared.graph
    idx = np.flatnonzero(g["transaction"].train_mask.numpy())[:8]
    if len(idx) == 0:
        idx = np.arange(min(8, g["transaction"].num_nodes))
    with torch.no_grad():
        _graph_logits(model, heads, next(iter(make_loader(prepared, "transaction", idx, 8, False))),
                      "transaction", "fraud", device)


def _cycle(loader):
    while True:
        yield from loader


def _tabular_batches(x, y, idx, batch_size, shuffle, rng):
    order = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(order), batch_size):
        sel = order[i:i + batch_size]
        yield x[sel], y[sel]


@torch.no_grad()
def predict_scores(prepared, model, heads, kind, idx, device, batch_size=None, variant="multitask"):
    idx = np.asarray(idx)
    if len(idx) == 0:
        return np.zeros(0)
    model.eval()
    heads.eval()
    out = []
    if variant == "no_graph":
        x = prepared.graph["transaction"].x
        for i in range(0, len(idx), 4096):
            z = model.encode_tabular(x[idx[i:i + 4096]].to(device))
            out.append(torch.sigmoid(heads.fraud_logits(z)).cpu().numpy())
        return np.concatenate(out)
    node_type = "transaction" if kind == "fraud" else "user"
    t = prepared.cfg["train"]
    bs = batch_size or (t["batch_fraud"] if kind == "fraud" else t["batch_account"])
    for batch in make_loader(prepared, node_type, idx, bs, shuffle=False):
        logits, _ = _graph_logits(model, heads, batch, node_type, kind, device)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def _validation_score(prepared, model, heads, spec, device, use_fraud, use_account):
    g, scores = prepared.graph, []
    if use_fraud and len(spec.fraud_val_idx):
        p = predict_scores(prepared, model, heads, "fraud", spec.fraud_val_idx, device, variant=spec.variant)
        scores.append(pr_auc(g["transaction"].y.numpy()[spec.fraud_val_idx], p))
    if use_account and len(spec.account_val_idx):
        p = predict_scores(prepared, model, heads, "account", spec.account_val_idx, device)
        scores.append(pr_auc(g["user"].y.numpy()[spec.account_val_idx], p))
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else -1.0


def train_variant(prepared, spec: RunSpec, device="cpu", checkpoint_path=None) -> TrainResult:
    cfg, g = prepared.cfg, prepared.graph
    t = cfg["train"]
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    use_fraud = spec.variant in FRAUD_VARIANTS
    use_account = spec.variant in ACCOUNT_VARIANTS
    w_f = t["w_fraud"] if use_fraud else 0.0
    w_a = t["w_account"] if use_account else 0.0
    alpha, gamma = t["focal_alpha"], t["focal_gamma"]
    graph_mode = spec.variant != "no_graph"
    model, heads = build_modules(prepared, device)
    if graph_mode:
        warm_up(model, heads, prepared, device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()),
                            lr=t["lr"], weight_decay=t["weight_decay"])

    fraud_loader = (make_loader(prepared, "transaction", spec.fraud_train_idx, t["batch_fraud"], True)
                    if use_fraud and graph_mode else None)
    account_loader = (make_loader(prepared, "user", spec.account_train_idx, t["batch_account"], True)
                      if use_account else None)
    x_tx, y_tx = g["transaction"].x, g["transaction"].y
    cap = t["max_steps_per_epoch"]
    best, best_epoch, bad, best_state, history = -np.inf, 0, 0, None, []

    for epoch in range(1, (spec.epochs or t["epochs"]) + 1):
        model.train()
        heads.train()
        losses = []
        if not graph_mode:
            for step, (xb, yb) in enumerate(_tabular_batches(x_tx, y_tx, spec.fraud_train_idx, t["batch_fraud"], True, rng)):
                if cap is not None and step >= cap:
                    break
                logits = heads.fraud_logits(model.encode_tabular(xb.to(device)))
                loss = w_f * focal_loss(logits, yb.float().to(device), alpha, gamma)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        else:
            main, kind, node_type, w_main = ((fraud_loader, "fraud", "transaction", w_f) if fraud_loader is not None
                                             else (account_loader, "account", "user", w_a))
            side = _cycle(account_loader) if fraud_loader is not None and account_loader is not None else None
            for step, batch in enumerate(main):
                if cap is not None and step >= cap:
                    break
                logits, y = _graph_logits(model, heads, batch, node_type, kind, device)
                loss = w_main * focal_loss(logits, y, alpha, gamma)
                if side is not None:
                    l2, y2 = _graph_logits(model, heads, next(side), "user", "account", device)
                    loss = loss + w_a * focal_loss(l2, y2, alpha, gamma)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan")}
        if spec.epochs is not None:  # fixed epochs: no validation, keep the last model
            history.append(entry)
            best_epoch = epoch
            continue
        entry["val_score"] = _validation_score(prepared, model, heads, spec, device, use_fraud, use_account)
        history.append(entry)
        if entry["val_score"] > best + 1e-9:
            best, best_epoch, bad = entry["val_score"], epoch, 0
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(heads.state_dict()))
        else:
            bad += 1
            if bad >= t["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state[0])
        heads.load_state_dict(best_state[1])
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "heads": heads.state_dict(), "variant": spec.variant,
                    "seed": spec.seed, "best_epoch": best_epoch, "config_hash": config_hash(cfg)}, checkpoint_path)
    return TrainResult(model, heads, best_epoch, float(best) if best_state is not None else float("nan"), history)


def load_checkpoint(path, prepared, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model, heads = build_modules(prepared, device)
    warm_up(model, heads, prepared, device)
    model.load_state_dict(ckpt["model"])
    heads.load_state_dict(ckpt["heads"])
    return model.eval(), heads.eval(), {k: ckpt[k] for k in ("variant", "seed", "best_epoch", "config_hash")}


def main(argv=None):
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Train one GNN variant.")
    p.add_argument("--variant", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    variant = args.variant or cfg["train"]["variant"]
    device = resolve_device(args.device or cfg["train"]["device"])
    prepared = prepare(cfg)
    path = resolve_path("models") / f"{variant}_seed{args.seed}.pt"
    res = train_variant(prepared, default_spec(prepared, variant, args.seed), device, path)
    print(f"variant={variant} seed={args.seed} best_epoch={res.best_epoch} val_score={res.best_score:.4f} saved={path}")


if __name__ == "__main__":
    main()
