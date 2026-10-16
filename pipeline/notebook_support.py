"""Helpers shared by the notebooks: where results live, how to load them, and the plots."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import load_config, resolve_path

HOW_TO = {
    "results_fraud.csv": "python -m scripts.evaluate", "results_account.csv": "python -m scripts.evaluate",
    "results_weekly.csv": "python -m scripts.evaluate", "calibration.csv": "python -m scripts.evaluate",
    "drift_weekly.csv": "python -m scripts.drift_study", "drift_summary.csv": "python -m scripts.drift_study",
}


def fixture_mode() -> bool:
    return os.environ.get("URE_FIXTURE") == "1" or os.environ.get("URE_QUICK") == "1"   # URE_QUICK: the spec's name for the tiny smoke run


def out_dir() -> Path:
    return resolve_path(os.environ.get("URE_OUT_DIR", "plots"))


def reports_dir() -> Path:
    return resolve_path(os.environ.get("URE_REPORTS_DIR", "reports"))


def get_prepared():
    from pipeline.prepare import prepare

    if fixture_mode():
        from tests.fixtures import fixture_overrides, make_fixture

        return prepare(load_config(overrides=fixture_overrides()), df=make_fixture()[0])
    return prepare(load_config())


def load_results(name: str) -> pd.DataFrame:
    path = out_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with `{HOW_TO.get(name, 'the matching script')}`.")
    return pd.read_csv(path)


def seed_summary(df: pd.DataFrame, metrics: list[str], by: str = "model") -> pd.DataFrame:
    g = df.groupby(by)[metrics]
    return g.mean().add_suffix("_mean").join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max")).reset_index()


def plot_pr_auc(fraud: pd.DataFrame):
    s = seed_summary(fraud, ["pr_auc"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(s))
    ax.barh(y, s["pr_auc_mean"], xerr=[s["pr_auc_mean"] - s["pr_auc_min"], s["pr_auc_max"] - s["pr_auc_mean"]],
            color="#4c78a8")
    ax.axvline(fraud["prevalence"].mean(), color="grey", linestyle="--", label="prevalence (chance)")
    ax.set_yticks(y, s["model"])
    ax.set_xlabel("test PR-AUC (bar = mean over seeds, whisker = min to max)")
    ax.legend()
    fig.tight_layout()
    return fig


def _by_model_week(weekly, models):
    models = models or list(weekly["model"].unique())
    return weekly[weekly["model"].isin(models)].groupby(["model", "week"], as_index=False)["pr_auc"].mean(), models


def plot_weekly(weekly: pd.DataFrame, models=None):
    data, models = _by_model_week(weekly, models)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for m in models:
        d = data[data["model"] == m]
        ax.plot(d["week"], d["pr_auc"], marker="o", label=m)
    ax.set_xlabel("week of the test period"), ax.set_ylabel("PR-AUC (gaps = no positives that week)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(cal: pd.DataFrame, models=None):
    models = models or list(cal["model"].unique())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    for m in models:
        d = cal[cal["model"] == m].groupby("mean_pred", as_index=False)["frac_pos"].mean()
        ax.plot(d["mean_pred"], d["frac_pos"], marker="o", label=m)
    ax.set_xlabel("mean predicted probability"), ax.set_ylabel("observed fraud rate")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drift(drift: pd.DataFrame):
    kinds = list(drift["kind"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 3.5), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for schedule, g in drift[drift["kind"] == kind].groupby("schedule"):
            ax.plot(g["week"], g["pr_auc"], marker="o", label=schedule)
        ax.set_title(kind), ax.set_xlabel("week since validation start"), ax.set_ylabel("PR-AUC")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_latency(df: pd.DataFrame, budget_ms: float = 100.0):
    d = df[df["mode"].isin(["inprocess", "http"])].copy()
    d["label"] = (d["model"] + " " + d["device"] + " " + d["mode"] + " b" + d["batch_size"].astype(str)
                  + " c" + d["concurrency"].astype(str))
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.barh(y, d["p99_ms"], color="#e45756", label="p99")
    ax.barh(y, d["p95_ms"], color="#f58518", label="p95")
    ax.barh(y, d["p50_ms"], color="#4c78a8", label="p50")
    ax.axvline(budget_ms, color="grey", linestyle="--", label=f"illustrative budget {budget_ms:g} ms")
    ax.set_yticks(y, d["label"])
    ax.set_xlabel("milliseconds per call (log scale)"), ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_breakdown(df: pd.DataFrame):
    comps = ["feature_transform_ms", "neighbour_lookup_ms", "model_forward_ms", "index_update_ms"]
    d = df[(df["mode"] == "inprocess") & (df["batch_size"] == 1)].dropna(subset=comps)
    d = d.assign(label=d["model"] + " " + d["device"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    left = np.zeros(len(d))
    for c in comps:
        ax.barh(d["label"], d[c], left=left, label=c.replace("_ms", ""))
        left += d[c].to_numpy()
    ax.set_xlabel("milliseconds per request (batch size 1)")
    ax.legend()
    fig.tight_layout()
    return fig
