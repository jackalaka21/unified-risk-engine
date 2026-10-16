"""Online scorers. A request becomes features and a sampled neighbourhood exactly as in training, then the model runs."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from pipeline.data_loading import finalize
from serving.graph_index import EDGES, ENTITIES, OutOfOrderError, keys_from_frame

REQUIRED = ("TransactionDT", "TransactionAmt")
DERIVED = ("user", "email", "device")
LABEL = "isFraud"
ALL_EDGE_TYPES = [et for pair in EDGES.values() for et in pair]


def raw_kinds_of(df: pd.DataFrame) -> dict:
    kinds = {}
    for c in df.columns:
        if c in DERIVED:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            kinds[c] = "float"
        elif pd.api.types.is_integer_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            kinds[c] = "int"
        else:
            kinds[c] = "object"
    return kinds


def _py(v):
    if isinstance(v, (list, tuple, dict)):
        return v
    if v is None or pd.isna(v):
        return None
    return v.item() if isinstance(v, np.generic) else v


def payload_of(df: pd.DataFrame, i: int) -> dict:
    """A request body for row `i`: raw fields only (no label, no derived keys), JSON-safe."""
    row = df.iloc[i]
    return {c: _py(row[c]) for c in df.columns if c not in DERIVED and c != LABEL}


def build_frame(payloads, kinds) -> pd.DataFrame:
    for p in payloads:
        missing = [c for c in REQUIRED if p.get(c) is None]
        if missing:
            raise ValueError(f"missing required field(s): {missing}")
    frame = pd.DataFrame({c: [p.get(c) for p in payloads] for c in kinds})
    for c, kind in kinds.items():
        if kind == "object":
            frame[c] = frame[c].astype(object)
        else:
            s = pd.to_numeric(frame[c], errors="coerce")
            frame[c] = s.astype("int64") if kind == "int" and not s.isna().any() else s.astype("float64")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("TransactionDT and TransactionAmt must be numbers")
    if LABEL in frame:
        frame[LABEL] = 0                                   # unknown at scoring time and never used by the features
    frame["_pos"] = np.arange(len(frame))
    return finalize(frame)


def collate(subs, device="cpu"):
    """Merge per-request subgraphs into one disjoint batch (what the training-time loader produces)."""
    offsets = {nt: 0 for nt in ("transaction",) + ENTITIES}
    xs, seeds = [], []
    src = {et: [] for et in ALL_EDGE_TYPES}
    dst = {et: [] for et in ALL_EDGE_TYPES}
    for s in subs:
        seeds.append(offsets["transaction"])
        xs.append(s.x)
        for et, pairs in s.edges.items():
            for a, b in pairs:
                src[et].append(a + offsets[et[0]])
                dst[et].append(b + offsets[et[2]])
        for nt, n in s.num_nodes.items():
            offsets[nt] += n
    edge_index = {et: torch.tensor([src[et], dst[et]], dtype=torch.long, device=device) for et in ALL_EDGE_TYPES}
    x_dict = {"transaction": torch.from_numpy(np.concatenate(xs)).to(device)}
    return x_dict, edge_index, dict(offsets), seeds


class _BaseScorer:
    def __init__(self, transformer, index, raw_kinds, version, late, info):
        if late not in ("reject", "clamp"):
            raise ValueError("late must be 'reject' or 'clamp'")
        self.transformer, self.index, self.raw_kinds = transformer, index, raw_kinds
        self.version, self.late, self.info = version, late, info or {}

    def _frame(self, payloads):
        return build_frame(payloads, self.raw_kinds)

    def _times(self, frame):
        times = frame["TransactionDT"].to_numpy(np.int64)
        if times.min() < self.index.last_time:
            if self.late == "reject":
                raise OutOfOrderError(f"request time {int(times.min())} is older than the newest indexed time "
                                      f"{int(self.index.last_time)}")
            times = np.maximum(times, int(self.index.last_time))   # late events are treated as arriving now
        return times

    def score(self, payload) -> float:
        return self.score_batch([payload])[0]


def _lap(timings, name, t0):
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0


class OnlineScorer(_BaseScorer):
    def __init__(self, model, heads, transformer, index, raw_kinds, device="cpu", version="dev", late="reject", info=None):
        super().__init__(transformer, index, raw_kinds, version, late, info)
        self.model, self.heads, self.device = model.to(device).eval(), heads.to(device).eval(), device

    def score_batch(self, payloads, timings=None):
        if not payloads:
            return []
        tm = timings if timings is not None else {}
        t0 = time.perf_counter()
        frame = self._frame(payloads)
        x = self.transformer.transform(frame)
        _lap(tm, "feature_transform", t0)
        times, amounts = self._times(frame), frame["TransactionAmt"].to_numpy(float)
        subs = []
        for i in range(len(frame)):
            keys = keys_from_frame(frame, i)
            t0 = time.perf_counter()
            subs.append(self.index.subgraph(keys, int(times[i]), x[i]))
            _lap(tm, "neighbour_lookup", t0)
            t0 = time.perf_counter()
            self.index.add(int(times[i]), float(amounts[i]), keys, x[i])
            _lap(tm, "index_update", t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            x_dict, edge_index, num_nodes, seeds = collate(subs, self.device)
            h = self.model(x_dict, edge_index, num_nodes)
            prob = torch.sigmoid(self.heads.fraud_logits(h["transaction"][torch.tensor(seeds, device=self.device)]))
            prob = prob.cpu().numpy()                      # also synchronises a GPU, so the lap is honest
        _lap(tm, "model_forward", t0)
        out = np.empty(len(frame))
        out[frame["_pos"].to_numpy()] = prob
        return out.tolist()


class TabularOnlineScorer(_BaseScorer):
    """The tabular baseline under the same serving protocol; `hist_columns=None` is the naive model."""

    def __init__(self, model, transformer, index, raw_kinds, hist_columns=None, version="dev", late="reject", info=None):
        super().__init__(transformer, index, raw_kinds, version, late, info)
        self.model, self.hist_columns = model, hist_columns

    def score_batch(self, payloads, timings=None):
        if not payloads:
            return []
        tm = timings if timings is not None else {}
        t0 = time.perf_counter()
        frame = self._frame(payloads)
        x = self.transformer.transform(frame)
        _lap(tm, "feature_transform", t0)
        times, amounts = self._times(frame), frame["TransactionAmt"].to_numpy(float)
        rows = []
        for i in range(len(frame)):
            keys = keys_from_frame(frame, i)
            t0 = time.perf_counter()
            if self.hist_columns is not None:
                h = self.index.history(keys, int(times[i]))
                rows.append(np.hstack([x[i], np.array([h[c] for c in self.hist_columns], dtype=np.float32)]))
            else:
                rows.append(x[i])
            _lap(tm, "neighbour_lookup", t0)
            t0 = time.perf_counter()
            self.index.add(int(times[i]), float(amounts[i]), keys, x[i])
            _lap(tm, "index_update", t0)
        t0 = time.perf_counter()
        prob = self.model.predict_proba(np.stack(rows))[:, 1]
        _lap(tm, "model_forward", t0)
        out = np.empty(len(frame))
        out[frame["_pos"].to_numpy()] = prob
        return out.tolist()
