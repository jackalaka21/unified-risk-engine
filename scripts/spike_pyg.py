"""Spike: verify the PyG features the design depends on. Prints PASS/FAIL/INFO lines and writes nothing."""
from __future__ import annotations

import time
import traceback

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv, HeteroConv

I64MAX = np.iinfo(np.int64).max


def synthetic_graph(n_tx=5000, n_users=300, n_emails=40, entity_time=True, seed=0):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.choice(n_tx * 50, size=n_tx, replace=False)).astype(np.int64)
    user = rng.integers(0, n_users, n_tx)
    email = rng.integers(0, n_emails, n_tx)
    d = HeteroData()
    d["transaction"].x = torch.randn(n_tx, 8)
    d["transaction"].y = torch.as_tensor((rng.random(n_tx) < 0.05).astype(np.int64))
    d["transaction"].time = torch.as_tensor(times)
    for name, codes, n in (("user", user, n_users), ("email_domain", email, n_emails)):
        first = np.full(n, I64MAX, dtype=np.int64)
        np.minimum.at(first, codes, times)
        d[name].num_nodes = n
        if entity_time:
            d[name].time = torch.as_tensor(first)
    tx = torch.arange(n_tx)
    u, e = torch.as_tensor(user), torch.as_tensor(email)
    d["user", "owns", "transaction"].edge_index = torch.stack([u, tx])
    d["transaction", "rev_owns", "user"].edge_index = torch.stack([tx, u])
    d["transaction", "uses_email", "email_domain"].edge_index = torch.stack([tx, e])
    d["email_domain", "rev_uses_email", "transaction"].edge_index = torch.stack([e, tx])
    return d


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def check_temporal(data, strategy="uniform"):
    idx = torch.arange(data["transaction"].num_nodes)[-512:]
    loader = NeighborLoader(
        data, num_neighbors=[15, 10], input_nodes=("transaction", idx),
        input_time=data["transaction"].time[idx], time_attr="time",
        temporal_strategy=strategy, batch_size=128,
    )
    batch = next(iter(loader))
    bs = batch["transaction"].batch_size
    seed_time = batch["transaction"].time[:bs]
    has_vec = all(hasattr(batch[nt], "batch") for nt in batch.node_types)
    violations = 0
    if has_vec:
        for nt in batch.node_types:
            if hasattr(batch[nt], "time"):
                violations += int((batch[nt].time > seed_time[batch[nt].batch]).sum())
    counts = {nt: batch[nt].num_nodes for nt in batch.node_types}
    edges = {str(et): int(batch[et].edge_index.shape[1]) for et in batch.edge_types}
    return {"batch_size": bs, "has_batch_vector": has_vec, "violations": violations,
            "node_counts": counts, "edge_counts": edges, "n_id_tx": batch["transaction"].n_id.clone()}


def check_equal_timestamp():
    d = HeteroData()
    d["transaction"].x = torch.zeros(3, 2)
    d["transaction"].time = torch.tensor([5, 5, 9])
    d["user"].num_nodes = 1
    d["user"].time = torch.tensor([5])
    tx = torch.arange(3)
    u = torch.zeros(3, dtype=torch.long)
    d["user", "owns", "transaction"].edge_index = torch.stack([u, tx])
    d["transaction", "rev_owns", "user"].edge_index = torch.stack([tx, u])
    loader = NeighborLoader(d, num_neighbors=[5, 5], input_nodes=("transaction", torch.tensor([0])),
                            input_time=torch.tensor([5]), time_attr="time", batch_size=1)
    batch = next(iter(loader))
    ids = batch["transaction"].n_id.tolist()
    return "earlier_or_equal" if 1 in ids else "strictly_earlier", ids


class TinyHetero(torch.nn.Module):
    def __init__(self, edge_types, hidden=64, heads=4):
        super().__init__()
        self.tx = torch.nn.Linear(8, hidden)
        self.vec = torch.nn.ParameterDict({"user": torch.nn.Parameter(torch.randn(hidden)),
                                           "email_domain": torch.nn.Parameter(torch.randn(hidden))})
        self.convs = torch.nn.ModuleList([
            HeteroConv({et: GATv2Conv((-1, -1), hidden, heads=heads, concat=False, add_self_loops=False)
                        for et in edge_types}, aggr="sum") for _ in range(2)])
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict, num_nodes):
        h = {"transaction": torch.relu(self.tx(x_dict["transaction"]))}
        for nt, p in self.vec.items():
            h[nt] = p.unsqueeze(0).repeat(num_nodes.get(nt, 0), 1)
        for conv in self.convs:
            o = conv(h, edge_index_dict)
            h = {nt: torch.relu(o[nt]) + h[nt] if nt in o else h[nt] for nt in h}
        return h


def check_speed(device, batch_sizes=(256, 1024, 2048), n_tx=200_000):
    data = synthetic_graph(n_tx=n_tx, n_users=15_000, n_emails=120)
    model = TinyHetero(list(data.edge_types)).to(device)
    # lazy init
    warm = next(iter(NeighborLoader(data, num_neighbors=[15, 10], input_nodes=("transaction", torch.arange(64)),
                                    input_time=data["transaction"].time[:64], time_attr="time", batch_size=64))).to(device)
    model(warm.x_dict, warm.edge_index_dict, {nt: warm[nt].num_nodes for nt in warm.node_types})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    seeds = torch.arange(n_tx)[-100_000:]
    rows = []
    for bs in batch_sizes:
        loader = NeighborLoader(data, num_neighbors=[15, 10], input_nodes=("transaction", seeds),
                                input_time=data["transaction"].time[seeds], time_attr="time",
                                batch_size=bs, shuffle=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0, n = time.perf_counter(), 0
        for batch in loader:
            batch = batch.to(device)
            h = model(batch.x_dict, batch.edge_index_dict, {nt: batch[nt].num_nodes for nt in batch.node_types})
            logits = model.out(h["transaction"][: batch["transaction"].batch_size]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["transaction"].y[: batch["transaction"].batch_size].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            n += 1
            if n == 6:
                break
        if device == "cuda":
            torch.cuda.synchronize()
        per_batch = (time.perf_counter() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
        train_seeds = 413_000  # about 70% of the real data's 590k transactions
        rows.append((bs, per_batch, peak, per_batch * train_seeds / bs / 60))
    return rows


def check_explainer():
    from torch_geometric.explain import Explainer, GNNExplainer

    data = synthetic_graph(n_tx=300, n_users=30, n_emails=6)
    net = TinyHetero(list(data.edge_types))
    nn_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    x_dict = {"transaction": data["transaction"].x}
    net(x_dict, data.edge_index_dict, nn_dict)  # lazy init

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x_dict, edge_index_dict):
            h = self.net(x_dict, edge_index_dict, nn_dict)
            return self.net.out(h["transaction"]).squeeze(-1)

    explainer = Explainer(
        model=Wrapper(), algorithm=GNNExplainer(epochs=20), explanation_type="model",
        node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )
    expl = explainer(x_dict, data.edge_index_dict, index=torch.tensor([0, 1]))
    info = {"type": type(expl).__name__,
            "tx_node_mask": tuple(expl["transaction"].node_mask.shape),
            "edge_masks": {str(et): tuple(expl[et].edge_mask.shape) for et in expl.edge_types if hasattr(expl[et], "edge_mask")}}
    try:
        from torch_geometric.explain.metric import fidelity

        info["fidelity"] = tuple(float(v) for v in fidelity(explainer, expl))
    except Exception as exc:  # recorded, not fatal
        info["fidelity_error"] = f"{type(exc).__name__}: {exc}"
    return info


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import pyg_lib

    report("env", True, f"torch {torch.__version__}, pyg_lib {pyg_lib.__version__}, device {device}")
    data = synthetic_graph()
    try:
        r = check_temporal(data)
        report("temporal sampling, time on every node type",
               r["violations"] == 0 and r["has_batch_vector"],
               f"violations={r['violations']} batch_vector={r['has_batch_vector']} nodes={r['node_counts']} edges={r['edge_counts']}")
    except Exception:
        report("temporal sampling, time on every node type", False, traceback.format_exc(limit=2))
    try:
        rule, ids = check_equal_timestamp()
        report("equal-timestamp rule", True, f"{rule} (sampled transaction ids {ids})")
    except Exception:
        report("equal-timestamp rule", False, traceback.format_exc(limit=2))
    try:
        check_temporal(synthetic_graph(entity_time=False))
        report("entity nodes without time", True, "sampling worked without entity time")
    except Exception as exc:
        report("entity nodes without time", False, f"{type(exc).__name__}: {exc}")
    try:
        a = check_temporal(data, "last")["n_id_tx"]
        b = check_temporal(data, "last")["n_id_tx"]
        report("'last' strategy deterministic", bool(torch.equal(a, b)))
    except Exception:
        report("'last' strategy deterministic", False, traceback.format_exc(limit=2))
    try:
        for bs, sec, peak, epoch_min in check_speed(device):
            print(f"INFO speed batch_size={bs}: {sec:.2f}s/batch, peak {peak:.2f} GiB, estimated epoch {epoch_min:.1f} min", flush=True)
    except Exception:
        report("speed and memory", False, traceback.format_exc(limit=3))
    try:
        report("hetero GNNExplainer", True, str(check_explainer()))
    except Exception:
        report("hetero GNNExplainer", False, traceback.format_exc(limit=3))


if __name__ == "__main__":
    main()
