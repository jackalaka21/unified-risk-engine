"""Save and load everything the scoring service needs. The pickle is for artifacts you built yourself only."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from models.gnn_backbone import HeteroGNN
from models.multi_task_heads import TaskHeads
from serving.graph_index import ENTITIES, GraphIndex
from serving.scorer import OnlineScorer, collate

WEIGHTS, BUNDLE, META, SAMPLES = "weights.pt", "bundle.pkl", "meta.json", "sample_requests.json"


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_bundle(out_dir, model_cfg, model, heads, transformer, index, raw_kinds, hist_columns, meta, samples=None,
                entities=ENTITIES):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, out / WEIGHTS)
    with open(out / BUNDLE, "wb") as f:
        pickle.dump({"model_cfg": model_cfg, "transformer": transformer, "index": index, "raw_kinds": raw_kinds,
                     "hist_columns": list(hist_columns), "entities": list(entities)}, f)
    # an integrity check (not authentication): it catches a corrupted or swapped file before the pickle is read
    meta = {**meta, "entity_types": list(entities), "sha256": {WEIGHTS: _sha256(out / WEIGHTS), BUNDLE: _sha256(out / BUNDLE)}}
    (out / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if samples is not None:
        (out / SAMPLES).write_text(json.dumps(samples), encoding="utf-8")
    return {"weights": out / WEIGHTS, "bundle": out / BUNDLE, "meta": out / META, "samples": out / SAMPLES}


def _warm(model, heads, n_features):
    """Lazy PyG layers need one forward pass over every edge type before weights can be loaded."""
    index = GraphIndex(n_features, (2, 2))
    index.add(0, 1.0, ("u", "e", "d"), np.zeros(n_features, np.float32))
    sg = index.subgraph(("u", "e", "d"), 1, np.zeros(n_features, np.float32))
    with torch.no_grad():
        model(*collate([sg])[:3])


def load_bundle(out_dir, device="cpu", late="reject") -> OnlineScorer:
    out = Path(out_dir)
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    for name, expected in meta.get("sha256", {}).items():
        if _sha256(out / name) != expected:
            raise ValueError(f"checksum mismatch for {name}: the file differs from what was exported; refusing to load it")
    with open(out / BUNDLE, "rb") as f:
        b = pickle.load(f)
    cfg = b["model_cfg"]
    model, heads = HeteroGNN(**cfg), TaskHeads(cfg["hidden"], cfg["dropout"])
    _warm(model, heads, b["index"].n_features)
    weights = torch.load(out / WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(weights["model"])
    heads.load_state_dict(weights["heads"])
    return OnlineScorer(model, heads, b["transformer"], b["index"], b["raw_kinds"], device, meta["version"], late, meta,
                        entities=tuple(b["entities"]))
