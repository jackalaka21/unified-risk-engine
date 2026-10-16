import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pipeline import notebook_support as ns


def test_paths_follow_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("URE_OUT_DIR", str(tmp_path / "o"))
    monkeypatch.setenv("URE_REPORTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("URE_FIXTURE", "1")
    assert ns.out_dir() == tmp_path / "o" and ns.reports_dir() == tmp_path / "r" and ns.fixture_mode()
    monkeypatch.delenv("URE_FIXTURE")
    assert not ns.fixture_mode()
    monkeypatch.setenv("URE_QUICK", "1")
    assert ns.fixture_mode()


def test_load_results_names_the_script_that_creates_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv("URE_OUT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="scripts.evaluate"):
        ns.load_results("results_fraud.csv")
    with pytest.raises(FileNotFoundError, match="scripts.drift_study"):
        ns.load_results("drift_weekly.csv")


def test_seed_summary_reports_a_range_not_a_confidence_interval():
    df = pd.DataFrame({"model": ["a", "a", "a", "b"], "seed": [0, 1, 2, 0], "pr_auc": [0.4, 0.5, 0.6, 0.3]})
    s = ns.seed_summary(df, ["pr_auc"]).set_index("model")
    assert s.loc["a", "pr_auc_mean"] == pytest.approx(0.5)
    assert s.loc["a", "pr_auc_min"] == 0.4 and s.loc["a", "pr_auc_max"] == 0.6
    assert s.loc["b", "pr_auc_min"] == s.loc["b", "pr_auc_max"] == 0.3


def test_plots_draw_something_and_survive_nan_weeks():
    fraud = pd.DataFrame({"model": ["a", "a", "b"], "seed": [0, 1, 0], "pr_auc": [0.4, 0.5, 0.3],
                          "pr_auc_lo": [0.3, 0.4, 0.2], "pr_auc_hi": [0.5, 0.6, 0.4], "prevalence": [0.03] * 3})
    weekly = pd.DataFrame({"model": "a", "seed": 0, "week": [0, 1, 2], "pr_auc": [0.4, np.nan, 0.3]})
    cal = pd.DataFrame({"model": "a", "seed": 0, "bin_lo": [0, 0.5], "bin_hi": [0.5, 1], "mean_pred": [0.1, 0.7],
                        "frac_pos": [0.05, 0.6], "count": [90, 10]})
    drift = pd.DataFrame({"kind": "k", "schedule": ["static"] * 2 + ["retrain"] * 2, "week": [0, 1, 0, 1],
                          "pr_auc": [0.5, 0.4, 0.5, 0.5]})
    for fig in (ns.plot_pr_auc(fraud), ns.plot_weekly(weekly), ns.plot_calibration(cal), ns.plot_drift(drift)):
        assert len(fig.axes[0].lines) + len(fig.axes[0].patches) + len(fig.axes[0].collections) > 0
        plt.close(fig)
