import json

import pandas as pd
import pytest

from scripts.make_model_card import build_model_card, check_single_config, results_markdown, splice_readme


def write_tables(tmp_path, hash_a="aaaa", hash_b="aaaa"):
    out, reports = tmp_path / "plots", tmp_path / "reports"
    out.mkdir(), reports.mkdir()
    pd.DataFrame({"model": ["history_tabular", "history_tabular", "multitask", "multitask"], "seed": [0, 1, 0, 1],
                  "config_hash": [hash_a, hash_a, hash_b, hash_b], "n_test": 1000, "prevalence": 0.03,
                  "pr_auc": [0.50, 0.52, 0.5678, 0.5722], "pr_auc_lo": 0.4, "pr_auc_hi": 0.7, "roc_auc": 0.9,
                  "recall_at_fpr": 0.4, "recall@0.01": 0.31, "amount_recall@0.01": 0.35, "cost": 900.0,
                  "cost_flag_none": 1500.0, "cost_flag_all": 4000.0, "brier": 0.02, "ece": 0.01}).to_csv(out / "results_fraud.csv", index=False)
    pd.DataFrame({"model": ["hgb_user_aggregates", "multitask"], "seed": 0, "config_hash": hash_a, "n_test": 200,
                  "prevalence": 0.1, "pr_auc": [0.3, 0.4], "roc_auc": 0.8}).to_csv(out / "results_account.csv", index=False)
    pd.DataFrame({"kind": "history_tabular", "schedule": ["static", "retrain"], "mean_pr_auc": [0.4, 0.45],
                  "slope_per_week": [-0.01, 0.0], "config_hash": hash_a}).to_csv(out / "drift_summary.csv", index=False)
    (reports / "explain_metrics.json").write_text(json.dumps(
        {"stability": 0.71, "fidelity_top": 1.2, "fidelity_random": 0.3, "config_hash": hash_a, "mode": "occlusion"}))
    return out, reports


def test_model_card_numbers_come_from_the_csvs(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports)
    assert "0.570" in card  # mean PR-AUC of the multitask rows: (0.5678 + 0.5722) / 2
    assert "0.510" in card and "aaaa" in card and "0.71" in card
    for heading in ("## Intended use", "## Data", "## Leakage controls", "## Results", "## Drift", "## Explanations",
                    "## Limitations"):
        assert heading in card


def test_model_card_refuses_mixed_config_hashes(tmp_path):
    out, reports = write_tables(tmp_path, hash_a="aaaa", hash_b="bbbb")
    with pytest.raises(ValueError, match="mixed config hashes"):
        build_model_card(out, reports)


def test_fixture_cards_are_marked_as_not_findings(tmp_path):
    out, reports = write_tables(tmp_path)
    assert "FIXTURE RESULTS" in build_model_card(out, reports, fixture=True)
    assert "FIXTURE RESULTS" not in build_model_card(out, reports, fixture=False)


def test_missing_optional_tables_are_reported_not_hidden(tmp_path):
    out, reports = write_tables(tmp_path)
    (out / "drift_summary.csv").unlink()
    assert "not run" in build_model_card(out, reports)


def test_splice_readme_replaces_only_the_marked_block():
    text = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n"
    out = splice_readme(text, "NEW")
    assert "NEW" in out and "old" not in out and out.startswith("intro") and out.endswith("outro\n")
    with pytest.raises(ValueError, match="RESULTS"):
        splice_readme("no markers", "NEW")


def test_results_markdown_has_one_row_per_model(tmp_path):
    out, _ = write_tables(tmp_path)
    md = results_markdown(pd.read_csv(out / "results_fraud.csv"), pd.read_csv(out / "results_account.csv"))
    assert md.count("| multitask |") == 2 and md.count("| history_tabular |") == 1  # fraud table + account table
