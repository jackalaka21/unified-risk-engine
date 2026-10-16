import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.prepare import prepare
from pipeline.splits import DAY, Periods
from scripts.drift_study import combine_by_cutoff, drift_study, retrain_cutoffs, summarize_drift
from tests.fixtures import fixture_overrides, make_fixture


def test_retrain_cutoffs_step_from_val_start_until_the_end():
    p = Periods(t_train=0.0, val_start=10 * DAY, val_end=50 * DAY, test_start=60 * DAY, embargo=10 * DAY)
    assert retrain_cutoffs(p, max_time=100 * DAY, every_days=28) == [38 * DAY, 66 * DAY, 94 * DAY]
    assert retrain_cutoffs(p, max_time=30 * DAY, every_days=28) == []


def test_combine_by_cutoff_switches_models_at_each_cutoff():
    times = np.array([1, 5, 11, 15, 25])
    base, m1, m2 = np.full(5, 0.0), np.full(5, 1.0), np.full(5, 2.0)
    assert combine_by_cutoff(times, [10, 20], [base, m1, m2]).tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]
    assert combine_by_cutoff(times, [], [base]).tolist() == [0.0] * 5


def test_summarize_drift_reports_a_negative_slope_for_a_decaying_curve():
    table = pd.DataFrame({"kind": "k", "schedule": "static", "week": [0, 1, 2, 3],
                          "pr_auc": [0.8, 0.6, 0.4, np.nan]})
    s = summarize_drift(table).iloc[0]
    assert s["slope_per_week"] == pytest.approx(-0.2) and s["mean_pr_auc"] == pytest.approx(0.6)


def test_a_planted_concept_flip_shows_decay_that_retraining_repairs():
    tx, _ = make_fixture(n_users=800, n_tx=20000, days=240, seed=1)
    flip = tx["TransactionDT"].quantile(0.75)                       # inside the validation period
    late = (tx["TransactionDT"] > flip).to_numpy()
    low_c1 = (tx["C1"] <= tx.loc[late, "C1"].quantile(0.3)).to_numpy()
    rng = np.random.default_rng(0)
    y = tx["isFraud"].to_numpy().copy()
    y[late] = (low_c1[late] & (rng.random(int(late.sum())) < 0.4)).astype(int)   # after the flip LOW C1 means fraud
    tx["isFraud"] = y
    overrides = fixture_overrides()
    overrides["drift"] = {"retrain_every_days": 14, "seed": 0}
    prep = prepare(load_config(overrides=overrides), df=tx)
    table = drift_study(prep, seed=0, kinds=("history_tabular",), log=lambda *_: None)
    flip_week = (flip - prep.periods.val_start) / (7 * DAY)

    def mean_pr(schedule, keep):
        g = table[(table["schedule"] == schedule) & keep(table["week"])]
        return g["pr_auc"].mean()

    before = mean_pr("static", lambda w: w < flip_week - 0.5)
    late_static = mean_pr("static", lambda w: w >= flip_week + 3)
    late_retrain = mean_pr("retrain", lambda w: w >= flip_week + 3)
    assert late_static < before and late_retrain > late_static
    # If this fails, strengthen the planted flip (more rows, a cleaner rule) before touching drift_study:
    # the test exists to prove the study can see decay and recovery.


def test_history_tabular_drift_study_on_the_fixture(raw_frames):
    prep = prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])
    table = drift_study(prep, seed=0, kinds=("history_tabular",), log=lambda *_: None)
    assert {"static", "retrain"} == set(table["schedule"]) and set(table["kind"]) == {"history_tabular"}
    for _, g in table.groupby("schedule"):
        assert g["week"].is_monotonic_increasing
    assert table["n"].sum() > 0
