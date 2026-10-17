import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.evaluate import evaluate_account, evaluate_all, evaluate_fraud
from tests.fixtures import fixture_overrides


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_fraud_row_has_every_metric_family(prep):
    va = np.flatnonzero(prep.graph["transaction"].val_mask.numpy())
    te = np.flatnonzero(prep.graph["transaction"].test_mask.numpy())
    y = prep.df["isFraud"].to_numpy()
    rng = np.random.default_rng(0)
    row, weekly, cal = evaluate_fraud("demo", 0, np.clip(y[va] * 0.5 + rng.random(len(va)) * 0.5, 0, 1),
                                      np.clip(y[te] * 0.5 + rng.random(len(te)) * 0.5, 0, 1), prep)
    for key in ("pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "recall_at_fpr", "realised_fpr", "recall@0.01",
                "precision@0.05", "amount_recall@0.005", "cost", "cost_flag_none", "brier", "ece", "config_hash"):
        assert key in row, key
    assert row["pr_auc_lo"] <= row["pr_auc"] <= row["pr_auc_hi"] and row["head"] == "fraud"
    assert {"week", "pr_auc"} <= set(weekly.columns) and len(cal) > 0


def test_account_row(prep):
    users = prep.account
    n_val = int(((users["split"] == "val") & users["eligible"]).sum())
    n_test = int(((users["split"] == "test") & users["eligible"]).sum())
    row = evaluate_account("demo", 0, np.random.rand(n_val), np.random.rand(n_test), prep)
    assert row["head"] == "account" and row["n_test"] == n_test and 0 <= row["prevalence"] <= 1


def test_results_carry_one_config_hash(prep, tmp_path):
    paths = evaluate_all(prep, tmp_path, variants=("multitask", "no_graph"), seeds=[0], device="cpu", log=lambda *_: None)
    fraud = pd.read_csv(paths["fraud"])
    account = pd.read_csv(paths["account"])
    assert fraud["config_hash"].nunique() == 1 and account["config_hash"].nunique() == 1
    assert fraud["config_hash"].iloc[0] == account["config_hash"].iloc[0]
    assert {"naive_tabular", "history_tabular", "multitask", "no_graph"} <= set(fraud["model"])
    assert {"hgb_user_aggregates", "mean_fraud_score", "multitask"} <= set(account["model"])
    assert fraud["pr_auc"].notna().all()
    assert (tmp_path / "results_weekly.csv").exists() and (tmp_path / "calibration.csv").exists()


def test_every_table_carries_the_config_hash_the_fixture_flag_and_the_bootstrap_share(prep, tmp_path):
    paths = evaluate_all(prep, tmp_path, variants=("fraud_only",), seeds=[0], device="cpu", log=lambda *_: None, fixture=True)
    for key in ("fraud", "account", "weekly", "calibration"):
        df = pd.read_csv(paths[key])
        assert df["config_hash"].nunique() == 1 and df["fixture"].all(), key
    assert pd.read_csv(paths["fraud"])["pr_auc_boot_kept"].between(0, 1).all()
    real = evaluate_all(prep, tmp_path / "real", variants=("fraud_only",), seeds=[0], device="cpu", log=lambda *_: None)
    assert not pd.read_csv(real["fraud"])["fixture"].any()


def test_history_input_variants_are_trained_on_the_wider_inputs_and_reported_under_their_own_name(prep, tmp_path):
    paths = evaluate_all(prep, tmp_path, variants=("fraud_only", "multitask_history"), seeds=[0], device="cpu",
                         log=lambda *_: None)
    fraud, account = pd.read_csv(paths["fraud"]), pd.read_csv(paths["account"])
    assert {"fraud_only", "multitask_history"} <= set(fraud["model"]) and fraud["pr_auc"].notna().all()
    assert "multitask_history" in set(account["model"])              # the history multitask model has an account head too
    assert "fraud_only_history" not in set(fraud["model"])


def test_stacked_models_are_evaluated_when_asked_and_not_otherwise(prep, tmp_path):
    with_stack = evaluate_all(prep, tmp_path / "a", variants=("fraud_only",), seeds=[0], device="cpu", log=lambda *_: None,
                              stacking=True)
    models = set(pd.read_csv(with_stack["fraud"])["model"])
    assert {"stacked_tabular", "stacked_labels_tabular", "graph_shared_tabular", "graph_labels_tabular"} <= models
    without = evaluate_all(prep, tmp_path / "b", variants=("fraud_only",), seeds=[0], device="cpu", log=lambda *_: None)
    assert "stacked_tabular" not in set(pd.read_csv(without["fraud"])["model"])
