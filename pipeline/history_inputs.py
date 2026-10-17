"""History features as GNN inputs: log-scaled, standardised on the train period, NaN-safe.

The GNN normally sees only per-transaction features, while the history-aware trees also see hand-built history. Feeding the
drift-robust history columns to the GNN tests whether the graph adds anything once both models see the same information.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch

from pipeline.history_features import stable_history_columns


class HistoryInputs:
    """log1p of the (non-negative) history columns, standardised on the fit rows; missing -> 0 plus an indicator column."""

    def fit(self, hist: pd.DataFrame, fit_rows) -> "HistoryInputs":
        self.columns = list(hist.columns)
        z = self._log(hist)
        rows = np.asarray(fit_rows, dtype=bool)
        with np.errstate(all="ignore"):
            self.mean = np.nan_to_num(np.nanmean(z[rows], axis=0))
            std = np.nan_to_num(np.nanstd(z[rows], axis=0))
        self.std = np.where(std > 0, std, 1.0)
        self.na_columns = [c for c, has in zip(self.columns, np.isnan(z[rows]).any(axis=0)) if has]
        self.names = ([f"hist_{c}" for c in self.columns] + [f"hist_na_{c}" for c in self.na_columns])
        return self

    def _log(self, hist: pd.DataFrame) -> np.ndarray:
        return np.log1p(np.clip(hist[self.columns].to_numpy(dtype=float), 0.0, None))

    def transform(self, hist: pd.DataFrame) -> np.ndarray:
        z = self._log(hist)
        scaled = np.nan_to_num((z - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        index = [self.columns.index(c) for c in self.na_columns]
        missing = np.isnan(z[:, index]).astype(float) if index else np.zeros((len(z), 0))
        return np.hstack([scaled, missing]).astype(np.float32)


def with_history_inputs(prepared):
    """A copy of `prepared` whose graph's transaction features also carry the history inputs. The original is untouched."""
    train = (prepared.df["TransactionDT"] <= prepared.periods.t_train).to_numpy()
    hist = prepared.hist[stable_history_columns(list(prepared.hist.columns))]
    inputs = HistoryInputs().fit(hist, train)
    x = np.hstack([np.asarray(prepared.x, dtype=np.float32), inputs.transform(hist)])
    graph = prepared.graph.clone()
    graph["transaction"].x = torch.as_tensor(x)
    return dataclasses.replace(prepared, graph=graph, x=x, history_inputs=inputs)
