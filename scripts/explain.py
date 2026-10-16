"""GNNExplainer (or an occlusion fallback) with stability and fidelity checks."""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pipeline.config import resolve_device, resolve_path
from scripts.train import load_checkpoint, make_loader, predict_scores


class FraudLogitWrapper(torch.nn.Module):
    def __init__(self, model, heads, num_nodes_dict):
        super().__init__()
        self.model, self.heads, self.num_nodes = model, heads, num_nodes_dict

    def forward(self, x_dict, edge_index_dict):
        h = self.model(x_dict, edge_index_dict, self.num_nodes)
        return self.heads.fraud_logits(h["transaction"])


def top_features(importance, names, k):
    order = np.argsort(-np.asarray(importance), kind="stable")[:k]
    return [(names[i], float(importance[i])) for i in order]


def topk_overlap(importances, k) -> float:
    sets = [set(np.argsort(-np.asarray(v), kind="stable")[:k].tolist()) for v in importances]
    pairs = list(itertools.combinations(sets, 2))
    if not pairs:
        return 1.0
    return float(np.mean([len(a & b) / len(a | b) for a, b in pairs]))


@torch.no_grad()
def occlusion_importance(wrapper, x_dict, edge_index_dict, row=0):
    wrapper.eval()
    base = wrapper(x_dict, edge_index_dict)[row]
    x = x_dict["transaction"]
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        x2 = x.clone()
        x2[row, j] = 0.0  # features are standardised on the train period, so 0 is the train mean
        out[j] = float((base - wrapper({**x_dict, "transaction": x2}, edge_index_dict)[row]).abs())
    return out


@torch.no_grad()
def deletion_fidelity(wrapper, x_dict, edge_index_dict, importance, k, row=0, seed=0):
    wrapper.eval()
    x = x_dict["transaction"]
    base = float(wrapper(x_dict, edge_index_dict)[row])

    def drop(cols):
        x2 = x.clone()
        x2[row, list(cols)] = 0.0
        return abs(base - float(wrapper({**x_dict, "transaction": x2}, edge_index_dict)[row]))

    top = np.argsort(-np.asarray(importance), kind="stable")[:k]
    rng = np.random.default_rng(seed)
    rand = [drop(rng.choice(x.shape[1], size=k, replace=False)) for _ in range(20)]
    return {"drop_top_k": drop(top), "drop_random_k": float(np.mean(rand))}


def _seed_batch(prepared, tx_row, device):
    return next(iter(make_loader(prepared, "transaction", np.array([tx_row]), 1, False))).to(device)


class AccountLogitWrapper(torch.nn.Module):
    def __init__(self, model, heads, num_nodes_dict):
        super().__init__()
        self.model, self.heads, self.num_nodes = model, heads, num_nodes_dict

    def forward(self, x_dict, edge_index_dict):
        return self.heads.account_logits(self.model(x_dict, edge_index_dict, self.num_nodes)["user"])


def _nonempty(edge_index_dict):
    """GNNExplainer crashes on an edge type with no edges (e.g. a seed with no email or device), so drop those."""
    return {et: ei for et, ei in edge_index_dict.items() if ei.shape[1] > 0}


def _hetero_explainer(wrapper, epochs):
    from torch_geometric.explain import Explainer, GNNExplainer

    return Explainer(model=wrapper, algorithm=GNNExplainer(epochs=epochs), explanation_type="model",
                     node_mask_type="attributes", edge_mask_type="object",
                     model_config=dict(mode="binary_classification", task_level="node", return_type="raw"))


def explain_transaction(prepared, model, heads, tx_row, mode, seed, device):
    batch = _seed_batch(prepared, tx_row, device)
    wrapper = FraudLogitWrapper(model, heads, {nt: batch[nt].num_nodes for nt in batch.node_types}).to(device).eval()
    x_dict, edge_index_dict = batch.x_dict, _nonempty(batch.edge_index_dict)
    with torch.no_grad():
        logit = float(wrapper(x_dict, edge_index_dict)[0])
    common = {"logit": logit, "wrapper": wrapper, "x_dict": x_dict, "edge_index_dict": edge_index_dict, "batch": batch}
    if mode == "occlusion":
        return {"importance": occlusion_importance(wrapper, x_dict, edge_index_dict), **common}
    torch.manual_seed(seed)
    explainer = _hetero_explainer(wrapper, prepared.cfg["explain"]["epochs"])
    expl = explainer(x_dict, edge_index_dict, index=torch.tensor([0], device=device))
    importance = expl["transaction"].node_mask[0].detach().cpu().numpy()
    return {"importance": importance, "explainer": explainer, "explanation": expl, **common}


def top_edges(out, k, entity_index=None):
    """The k most influential sampled edges of a hetero explanation: [(edge_type, src, dst, importance)].

    Transactions are labelled by row; entity nodes by key when `entity_index` is given, else by node id."""
    expl, batch = out["explanation"], out["batch"]

    def label(node_type, node_id):
        node_id = int(node_id)
        return node_id if node_type == "transaction" or entity_index is None else entity_index[node_type][node_id]

    rows = []
    for et in expl.edge_types:
        store = expl[et]
        if "edge_mask" not in store:
            continue
        src_t, _, dst_t = et
        ei = batch[et].edge_index
        for j, value in enumerate(store.edge_mask.detach().cpu().tolist()):
            rows.append((et, label(src_t, batch[src_t].n_id[ei[0, j]]), label(dst_t, batch[dst_t].n_id[ei[1, j]]), float(value)))
    return sorted(rows, key=lambda r: -r[3])[:k]


def pyg_fidelity(out):
    """PyG's fidelity+/- and characterization score for one hetero explanation, or None if PyG cannot compute them."""
    try:
        from torch_geometric.explain.metric import characterization_score, fidelity

        plus, minus = fidelity(out["explainer"], out["explanation"])
        return {"fidelity_plus": float(plus), "fidelity_minus": float(minus),
                "characterization": float(characterization_score(plus, minus))}
    except Exception as exc:                              # unsupported for this model or graph: say so, do not hide it
        warnings.warn(f"PyG fidelity unavailable: {exc!r}")
        return None


def explain_user(prepared, model, heads, user_node, seed, device, k=5):
    """Most influential sampled edges behind one user's account-risk score (hetero mode; users have no features)."""
    batch = next(iter(make_loader(prepared, "user", np.array([user_node]), 1, False))).to(device)
    wrapper = AccountLogitWrapper(model, heads, {nt: batch[nt].num_nodes for nt in batch.node_types}).to(device).eval()
    torch.manual_seed(seed)
    explainer = _hetero_explainer(wrapper, prepared.cfg["explain"]["epochs"])
    expl = explainer(batch.x_dict, _nonempty(batch.edge_index_dict), index=torch.tensor([0], device=device))
    return top_edges({"explanation": expl, "batch": batch}, k, prepared.entity_index)


def explain_report(prepared, checkpoint, out_dir, device=None):
    cfg = prepared.cfg
    e = cfg["explain"]
    hetero = e["mode"] == "hetero"
    device = resolve_device(device or cfg["train"]["device"])
    model, heads, meta = load_checkpoint(checkpoint, prepared, device)
    names = prepared.transformer.feature_names
    te = np.flatnonzero(prepared.graph["transaction"].test_mask.numpy())
    p = predict_scores(prepared, model, heads, "fraud", te, device)
    y = prepared.df["isFraud"].to_numpy()[te]
    order = np.argsort(-p, kind="stable")
    half = e["n_explained"] // 2
    picks = ([int(te[i]) for i in order if y[i] == 1][:half]
             + [int(te[i]) for i in order if y[i] == 0][: e["n_explained"] - half])
    sample = np.random.default_rng(0).choice(te, size=min(e["fidelity_sample"], len(te)), replace=False)
    stability, rows, mean_imp = [], [], np.zeros(len(names))
    for row in picks:
        outs = [explain_transaction(prepared, model, heads, row, e["mode"], s, device) for s in range(e["stability_seeds"])]
        runs = [o["importance"] for o in outs]
        stability.append(topk_overlap(runs, e["top_k"]))
        mean_imp += np.mean(runs, axis=0)
        edges = top_edges(outs[0], 5, prepared.entity_index) if hetero else []
        rows.append((row, top_features(np.mean(runs, axis=0), names, e["top_k"]), edges))
    fid_top, fid_rand, pyg = [], [], []
    for row in sample:
        occ = explain_transaction(prepared, model, heads, int(row), "occlusion", 0, device)
        f = deletion_fidelity(occ["wrapper"], occ["x_dict"], occ["edge_index_dict"], occ["importance"], e["top_k"])
        fid_top.append(f["drop_top_k"])
        fid_rand.append(f["drop_random_k"])
        if hetero:
            r = pyg_fidelity(explain_transaction(prepared, model, heads, int(row), "hetero", 0, device))
            if r is not None:
                pyg.append(r)
    user_rows = []
    if hetero:
        index = prepared.entity_index["user"]
        for u in np.flatnonzero(prepared.graph["user"].test_mask.numpy())[:3]:
            user_rows.append((index[int(u)], explain_user(prepared, model, heads, int(u), 0, device)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top = np.argsort(-mean_imp)[: e["top_k"]]
    plt.figure(figsize=(7, 4))
    plt.barh([names[i] for i in top][::-1], mean_imp[top][::-1] / max(len(picks), 1))
    plt.xlabel("mean importance"), plt.title("Top features across the explained transactions")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=130)
    plt.close()
    mean_pyg = ({k: float(np.mean([r[k] for r in pyg])) for k in ("fidelity_plus", "fidelity_minus", "characterization")}
                if pyg else None)
    lines = ["# Explanations", "",
             f"Mode: `{e['mode']}`. Features are anonymised, so names are column names, not business meaning. "
             "Explanations describe the model, not causes.", "",
             f"- Stability (mean top-{e['top_k']} overlap across {e['stability_seeds']} runs): {np.mean(stability):.2f}",
             f"- Deletion fidelity (occlusion, {len(sample)} transactions): dropping the top-{e['top_k']} features changes the "
             f"logit by {np.mean(fid_top):.3f} on average, against {np.mean(fid_rand):.3f} for random features"]
    if mean_pyg:
        lines.append(f"- PyG fidelity+ {mean_pyg['fidelity_plus']:.3f}, fidelity- {mean_pyg['fidelity_minus']:.3f}, "
                     f"characterization {mean_pyg['characterization']:.3f} ({len(pyg)} transactions)")
    elif hetero:
        lines.append("- PyG fidelity metrics were not available for this model (see the warnings of the run)")
    lines.append("")
    for row, feats, edges in rows:
        lines.append(f"## Transaction row {row} (TransactionID {int(prepared.df['TransactionID'].iloc[row])})")
        lines += [f"- `{n}`: {v:.3f}" for n, v in feats]
        lines += [f"- edge `{et[1]}`: {s} -> {d} ({v:.3f})" for et, s, d, v in edges] + [""]
    for key, edges in user_rows:
        lines.append(f"## User {key} (account-risk head)")
        lines += [f"- edge `{et[1]}`: {s} -> {d} ({v:.3f})" for et, s, d, v in edges] + [""]
    (out_dir / "explanations.md").write_text("\n".join(lines), encoding="utf-8")
    return {"stability": float(np.mean(stability)), "fidelity_top": float(np.mean(fid_top)),
            "fidelity_random": float(np.mean(fid_rand)), "n_fidelity": int(len(sample)),
            "pyg_fidelity_plus": mean_pyg and mean_pyg["fidelity_plus"],
            "pyg_fidelity_minus": mean_pyg and mean_pyg["fidelity_minus"],
            "pyg_characterization": mean_pyg and mean_pyg["characterization"]}
