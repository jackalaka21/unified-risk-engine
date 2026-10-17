"""One call that turns raw data into everything the training, baseline and serving code needs."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.data_loading import finalize, load_raw
from pipeline.feature_engineering import fit_on_train
from pipeline.graph_builder import build_graph
from pipeline.graph_features import graph_features
from pipeline.history_features import history_features
from pipeline.splits import account_table, check_positives, compute_periods, split_users


@dataclass
class Prepared:
    cfg: dict
    df: pd.DataFrame
    periods: object
    user_split: pd.Series
    account: pd.DataFrame
    transformer: object
    x: object
    hist: pd.DataFrame
    graph: object
    entity_index: dict
    gfeat: pd.DataFrame = None        # graph-style aggregates for the tabular models (delayed labels, shared entities)


def prepare(cfg: dict, df: pd.DataFrame | None = None) -> Prepared:
    df = load_raw(cfg) if df is None else finalize(df)
    s, a = cfg["splits"], cfg["account"]
    periods = compute_periods(df["TransactionDT"], s["train_frac"], s["val_end_frac"], s["embargo_days"])
    user_split = split_users(df["user"], s["user_fractions"], cfg["seed"])
    account = account_table(df, periods, user_split, a["min_transactions"], a["fraud_rate_threshold"])
    check_positives(account, a["min_positives"])
    transformer = fit_on_train(df, periods, cfg)
    x = transformer.transform(df)
    hist = history_features(df)
    graph, entity_index = build_graph(df, x, periods, user_split, account)
    train_rate = float(df.loc[df["TransactionDT"] <= periods.t_train, "isFraud"].mean())
    gfeat = graph_features(df, periods.embargo, prior_rate=train_rate)          # labels only after the same embargo as the splits
    return Prepared(cfg, df, periods, user_split, account, transformer, x, hist, graph, entity_index, gfeat)
