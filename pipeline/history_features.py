"""Label-free, strictly-causal history features for the tabular baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400


def _per_entity(codes: np.ndarray, t: np.ndarray, a: np.ndarray, distinct: dict | None = None) -> dict:
    n = len(t)
    out = {k: np.full(n, np.nan) for k in ("n_prior", "n_1d", "n_7d", "amt_mean", "amt_std", "secs_since_prev")}
    for name in (distinct or {}):
        out[f"distinct_{name}"] = np.full(n, np.nan)
    valid = np.flatnonzero(codes >= 0)
    if len(valid) == 0:
        return out
    order = valid[np.lexsort((t[valid], codes[valid]))]  # by entity, then time
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    for grp in np.split(order, cuts):
        times, amts = t[grp], a[grp]
        prior = np.searchsorted(times, times, side="left")  # count of strictly earlier rows
        cs = np.concatenate([[0.0], np.cumsum(amts)])
        cs2 = np.concatenate([[0.0], np.cumsum(amts ** 2)])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(prior > 0, cs[prior] / np.maximum(prior, 1), np.nan)
            var = np.where(prior > 1, (cs2[prior] - prior * mean ** 2) / np.maximum(prior - 1, 1), np.nan)
        out["n_prior"][grp] = prior
        out["n_1d"][grp] = prior - np.searchsorted(times, times - DAY, side="left")
        out["n_7d"][grp] = prior - np.searchsorted(times, times - 7 * DAY, side="left")
        out["amt_mean"][grp] = mean
        out["amt_std"][grp] = np.sqrt(np.maximum(var, 0.0))
        out["secs_since_prev"][grp] = np.where(prior > 0, times - times[np.maximum(prior - 1, 0)], np.nan)
        for name, other in (distinct or {}).items():
            e = other[grp]
            first = (e >= 0) & ~pd.Series(e).duplicated().to_numpy()
            out[f"distinct_{name}"][grp] = np.concatenate([[0], np.cumsum(first)])[prior]
    return out


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["TransactionDT"].to_numpy(np.int64)
    a = df["TransactionAmt"].to_numpy(float)
    user = pd.factorize(df["user"])[0]
    email = pd.factorize(df["email"])[0]
    device = pd.factorize(df["device"])[0]
    blocks = {
        "user": _per_entity(user, t, a, distinct={"devices": device, "emails": email}),
        "email": _per_entity(email, t, a),
        "device": _per_entity(device, t, a),
    }
    cols = {f"{ent}_{name}": arr for ent, block in blocks.items() for name, arr in block.items()}
    return pd.DataFrame(cols, index=df.index)
