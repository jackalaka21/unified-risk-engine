"""Append-only, time-ordered neighbour index that rebuilds the training-time sampled neighbourhood online.

It reproduces PyG temporal neighbour sampling with the "last" strategy on a two-hop graph: the seed transaction
connects to its user, email domain and device; each of those connects to its `fanouts[1]` most recent
transactions at or before the seed time. The new transaction is always the most recent of its own entities.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd

ENTITIES = ("user", "email_domain", "device")
DAY = 86400
# entity -> (edge into the transaction, edge out of the transaction); must equal pipeline.graph_builder.EDGE_TYPES
EDGES = {
    "user": (("user", "owns", "transaction"), ("transaction", "rev_owns", "user")),
    "email_domain": (("email_domain", "rev_uses_email", "transaction"), ("transaction", "uses_email", "email_domain")),
    "device": (("device", "rev_uses_device", "transaction"), ("transaction", "uses_device", "device")),
}
PREFIX = {"user": "user", "email_domain": "email", "device": "device"}   # history_features column prefixes
BASE_NAMES = ["n_prior", "n_1d", "n_7d", "amt_mean", "amt_std", "secs_since_prev"]


class OutOfOrderError(ValueError):
    """A request is older than a transaction already in the index."""


def _missing(key) -> bool:
    return key is None or (not isinstance(key, str) and bool(pd.isna(key)))


def _clean(keys):
    return tuple(None if _missing(k) else k for k in keys)


def keys_from_frame(frame, i):
    return _clean((frame["user"].iat[i], frame["email"].iat[i], frame["device"].iat[i]))


class _Members:
    """Everything one entity has done, in time order, with prefix sums so history queries are O(log n)."""

    __slots__ = ("rows", "times", "cs", "cs2", "dev_prefix", "mail_prefix")

    def __init__(self):
        self.rows, self.times = [], []
        self.cs, self.cs2 = [0.0], [0.0]
        self.dev_prefix, self.mail_prefix = [0], [0]       # distinct devices / emails over the first j members (users)


_EMPTY = _Members()                                        # shared, never mutated: an entity seen for the first time


@dataclass
class Subgraph:
    x: np.ndarray
    tx_rows: list
    entity_keys: dict
    edges: dict

    @property
    def num_nodes(self) -> dict:
        return {"transaction": len(self.tx_rows), **{e: len(self.entity_keys[e]) for e in ENTITIES}}


class GraphIndex:
    def __init__(self, n_features: int, fanouts=(15, 10)):
        if len(fanouts) != 2:
            raise ValueError("the index reproduces a two-hop sampler: train.num_neighbors needs exactly two entries")
        self.n_features, self.k2 = n_features, int(fanouts[1])
        self.members = {e: {} for e in ENTITIES}
        self._seen = {}                                    # user key -> (device keys, email keys)
        self.times, self.amounts, self.row_keys = [], [], []
        self._x = np.zeros((1024, n_features), np.float32)
        self.last_time = -np.inf

    def __len__(self) -> int:
        return len(self.times)

    def x_of(self, row: int) -> np.ndarray:
        return self._x[row]

    def add(self, time, amount, keys, x) -> int:
        if time < self.last_time:
            raise OutOfOrderError(f"transaction time {time} is older than the newest indexed time {self.last_time}")
        keys = _clean(keys)
        row = len(self.times)
        if row >= len(self._x):
            self._x = np.concatenate([self._x, np.zeros_like(self._x)])
        self._x[row] = x
        for e, key in zip(ENTITIES, keys):
            if key is None:
                continue
            m = self.members[e].setdefault(key, _Members())
            m.rows.append(row)
            m.times.append(time)
            m.cs.append(m.cs[-1] + amount)
            m.cs2.append(m.cs2[-1] + amount * amount)
            if e == "user":
                devices, emails = self._seen.setdefault(key, (set(), set()))
                if keys[2] is not None:
                    devices.add(keys[2])
                if keys[1] is not None:
                    emails.add(keys[1])
                m.dev_prefix.append(len(devices))
                m.mail_prefix.append(len(emails))
        self.times.append(time)
        self.amounts.append(amount)
        self.row_keys.append(keys)
        self.last_time = time
        return row

    def lookup(self, entity: str, key, t, k: int) -> list:
        m = self.members[entity].get(key)
        if m is None:
            return []
        hi = bisect_right(m.times, t)
        return m.rows[max(0, hi - k):hi]

    def subgraph(self, keys, t, x_new) -> Subgraph:
        keys = _clean(keys)
        tx_rows, tx_x, local = [-1], [np.asarray(x_new, np.float32)], {}
        entity_keys = {e: [] for e in ENTITIES}
        edges: dict = {}
        for e, key in zip(ENTITIES, keys):
            if key is None:
                continue
            into, out = EDGES[e]
            entity_keys[e].append(key)
            edges.setdefault(into, []).append((0, 0))
            for row in self.lookup(e, key, t, self.k2 - 1):
                if row not in local:
                    local[row] = len(tx_rows)
                    tx_rows.append(row)
                    tx_x.append(self._x[row])
                edges.setdefault(out, []).append((local[row], 0))
            edges.setdefault(out, []).append((0, 0))        # the new transaction is the most recent neighbour
        return Subgraph(np.stack(tx_x), tx_rows, entity_keys, edges)

    def history(self, keys, t) -> dict:
        """The tabular baseline's history features for a transaction arriving at time `t` (strictly earlier rows only)."""
        out = {}
        for e, key in zip(ENTITIES, _clean(keys)):
            pre = PREFIX[e]
            names = BASE_NAMES + (["distinct_devices", "distinct_emails"] if e == "user" else [])
            if key is None:
                out.update({f"{pre}_{n}": np.nan for n in names})
                continue
            m = self.members[e].get(key, _EMPTY)
            prior = bisect_left(m.times, t)
            mean = m.cs[prior] / prior if prior > 0 else np.nan
            var = (m.cs2[prior] - prior * mean ** 2) / (prior - 1) if prior > 1 else np.nan
            out[f"{pre}_n_prior"] = float(prior)
            out[f"{pre}_n_1d"] = float(prior - bisect_left(m.times, t - DAY))
            out[f"{pre}_n_7d"] = float(prior - bisect_left(m.times, t - 7 * DAY))
            out[f"{pre}_amt_mean"] = mean
            out[f"{pre}_amt_std"] = float(np.sqrt(max(var, 0.0))) if prior > 1 else np.nan
            out[f"{pre}_secs_since_prev"] = float(t - m.times[prior - 1]) if prior > 0 else np.nan
            if e == "user":
                out[f"{pre}_distinct_devices"] = float(m.dev_prefix[prior])
                out[f"{pre}_distinct_emails"] = float(m.mail_prefix[prior])
        return out

    @classmethod
    def from_prepared(cls, prepared, upto_row=None) -> "GraphIndex":
        df = prepared.df
        n = len(df) if upto_row is None else int(upto_row)
        index = cls(prepared.x.shape[1], prepared.cfg["train"]["num_neighbors"])
        t, a = df["TransactionDT"].to_numpy(np.int64), df["TransactionAmt"].to_numpy(float)
        keys = list(zip(df["user"], df["email"], df["device"]))
        for i in range(n):
            index.add(int(t[i]), float(a[i]), keys[i], prepared.x[i])
        return index
