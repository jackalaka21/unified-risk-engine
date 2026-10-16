import numpy as np
import pytest

from pipeline.metrics import (alert_budget, bootstrap_metric, calibration, expected_cost, pr_auc,
                              recall_at_fpr, roc_auc, weekly_table)

Y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
PERFECT = np.array([0.9, 0.8, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0])


def test_ranking_metrics_on_perfect_scores():
    assert pr_auc(Y, PERFECT) == pytest.approx(1.0) and roc_auc(Y, PERFECT) == pytest.approx(1.0)


def test_degenerate_inputs():
    assert np.isnan(pr_auc(np.zeros(5), np.random.rand(5))) and np.isnan(roc_auc(np.ones(5), np.random.rand(5)))
    assert pr_auc(Y, np.full(10, 0.5)) == pytest.approx(Y.mean())  # constant scores: precision = prevalence
    assert np.isnan(recall_at_fpr(Y, PERFECT, np.zeros(10), PERFECT)["recall"])  # test set with no positives


def test_recall_at_fpr_uses_the_validation_threshold():
    y_val = np.array([0] * 100 + [1] * 10)
    s_val = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.6, 1.0, 10)])
    r = recall_at_fpr(y_val, s_val, y_val, s_val, fpr=0.01)
    assert r["recall"] == 1.0 and r["realised_fpr"] <= 0.02
    assert 0.49 <= r["threshold"] <= 0.5


def test_alert_budget_counts_exactly():
    amount = np.array([100, 10, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    r = alert_budget(Y, PERFECT, amount, 0.2)
    assert r["k"] == 2 and r["recall"] == 1.0 and r["precision"] == 1.0 and r["amount_recall"] == 1.0
    r5 = alert_budget(Y, np.array([0.1, 0.9, 0.8, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0]), amount, 0.2)
    assert r5["recall"] == 0.5 and r5["amount_recall"] == pytest.approx(10 / 110)


def test_expected_cost_beats_flagging_nothing_with_a_good_model():
    amount = np.full(10, 50.0)
    c = expected_cost(Y, PERFECT, amount, Y, PERFECT, amount, review_cost=5.0)
    assert c["cost"] < c["cost_flag_none"] and c["cost_flag_none"] == 100.0 and c["cost_flag_all"] == 40.0


def test_calibration_of_a_calibrated_and_a_miscalibrated_score():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    good, table = calibration(y, p)
    bad, _ = calibration(y, np.clip(p * 0.3, 0, 1))
    assert good["ece"] < 0.03 < bad["ece"] and good["brier"] < bad["brier"]
    assert list(table.columns) == ["bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"]


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int)
    s = y * 0.4 + rng.random(500) * 0.6
    a = bootstrap_metric(pr_auc, y, s, n_boot=200, seed=0)
    assert a == bootstrap_metric(pr_auc, y, s, n_boot=200, seed=0)
    assert a[0] <= pr_auc(y, s) <= a[1]


def test_weekly_table_marks_weeks_without_positives():
    day = 86400
    times = np.concatenate([np.arange(0, 7 * day, day), np.arange(7 * day, 14 * day, day)])
    y = np.concatenate([[1, 0, 0, 0, 1, 0, 0], np.zeros(7)])
    s = np.random.default_rng(0).random(14)
    t = weekly_table(times, y, s, start=0)
    assert t["week"].tolist() == [0, 1] and t["n"].tolist() == [7, 7]
    assert not np.isnan(t.loc[0, "pr_auc"]) and np.isnan(t.loc[1, "pr_auc"]) and t.loc[1, "n_pos"] == 0


def test_alert_budget_with_tied_scores_does_not_depend_on_row_order():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    amount = np.arange(1, 11, dtype=float)
    s = np.full(10, 0.5)
    a = alert_budget(y, s, amount, 0.2)
    b = alert_budget(y[::-1], s, amount[::-1], 0.2)            # same data, reversed row order
    assert a["recall"] == pytest.approx(0.2) and a["recall"] == pytest.approx(b["recall"])
    assert a["precision"] == pytest.approx(0.2) and a["amount_recall"] == pytest.approx(b["amount_recall"])


def test_expected_cost_can_choose_to_flag_everything():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    amount = np.full(6, 1000.0)
    c = expected_cost(y, s, amount, y, s, amount, review_cost=0.001)      # reviewing is nearly free: flag all
    assert c["cost"] == pytest.approx(c["cost_flag_all"])


def test_bootstrap_can_report_the_share_of_resamples_it_kept():
    y = np.array([1] + [0] * 19)
    s = np.random.default_rng(0).random(20)
    lo, hi, kept = bootstrap_metric(pr_auc, y, s, n_boot=300, seed=0, return_kept=True)
    assert 0 < kept < 1                                                  # resamples without a positive are dropped
    assert bootstrap_metric(pr_auc, y, s, n_boot=300, seed=0) == (lo, hi)
