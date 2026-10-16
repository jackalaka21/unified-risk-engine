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
                  "recall_at_fpr": 0.4, "realised_fpr": 0.012, "recall@0.005": 0.2, "recall@0.01": 0.31, "recall@0.05": 0.55,
                  "amount_recall@0.005": 0.22, "amount_recall@0.01": 0.35, "amount_recall@0.05": 0.6, "cost": 900.0,
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


from scripts.make_model_card import latency_markdown, splice_block


def write_latency(out, config_hash="aaaa", parity=True):
    pd.DataFrame({"model": ["gnn", "history_tabular"], "device": "cpu", "mode": "inprocess", "batch_size": 1,
                  "concurrency": 1, "p50_ms": [3.2, 1.1], "p95_ms": [4.5, 1.9], "p99_ms": [6.789, 2.4],
                  "throughput_rps": [250.0, 700.0], "p99_within_budget": True}).to_csv(out / "latency.csv", index=False)
    (out / "latency_meta.json").write_text(json.dumps(
        {"budget_ms": 100.0, "config_hash": config_hash, "fixture": False, "hardware": {"gpu": "test"},
         "parity": {"checked": parity, "max_abs_diff": 3e-6, "n_rows": 200, "reason": ""}, "index_pickle_mb": 12.3}))


def test_model_card_reports_latency_numbers_from_the_csv(tmp_path):
    out, reports = write_tables(tmp_path)
    write_latency(out)
    card = build_model_card(out, reports)
    assert "## Serving and latency" in card and "6.79" in card      # p99 of the gnn row, two decimals
    assert "illustrative" in card and "parity" in card.lower()


def test_latency_from_another_configuration_is_refused(tmp_path):
    out, reports = write_tables(tmp_path)
    write_latency(out, config_hash="zzzz")
    with pytest.raises(ValueError, match="mixed config hashes"):
        build_model_card(out, reports)


def test_missing_latency_is_reported_not_hidden(tmp_path):
    out, reports = write_tables(tmp_path)
    assert "latency benchmark was not run" in build_model_card(out, reports)


def test_an_unchecked_parity_is_stated_plainly(tmp_path):
    out, reports = write_tables(tmp_path)
    write_latency(out, parity=False)
    assert "not checked" in build_model_card(out, reports)


def test_splice_block_replaces_only_the_named_block():
    text = "a\n<!-- X:START -->\nold\n<!-- X:END -->\nb\n<!-- Y:START -->\nkeep\n<!-- Y:END -->\n"
    out = splice_block(text, "X", "NEW")
    assert "NEW" in out and "old" not in out and "keep" in out
    with pytest.raises(ValueError, match="Z"):
        splice_block(text, "Z", "NEW")


def test_fixture_latency_cannot_enter_a_real_card(tmp_path):
    out, reports = write_tables(tmp_path)
    write_latency(out)
    meta = json.loads((out / "latency_meta.json").read_text())
    meta["fixture"] = True
    (out / "latency_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fixture"):
        build_model_card(out, reports, fixture=False)
    assert "## Serving and latency" in build_model_card(out, reports, fixture=True)


def test_sampling_rule_table_numbers_reach_the_card(tmp_path):
    out, reports = write_tables(tmp_path)
    pd.DataFrame({"rule": ["uniform", "last"], "pr_auc": [0.4321, 0.4123], "variant": "multitask", "seed": 0,
                  "temporal": True, "config_hash": "aaaa"}).to_csv(out / "results_sampling_rule.csv", index=False)
    card = build_model_card(out, reports)
    assert "0.432" in card and "0.412" in card and "neighbour rule" in card


def test_card_shows_bootstrap_intervals_every_budget_and_the_flag_nothing_cost(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports)
    assert "0.400-0.700" in card                                   # mean bootstrap bounds of the PR-AUC
    for column in ("realised_fpr", "recall@0.005", "recall@0.01", "recall@0.05", "amount_recall@0.05", "cost_flag_none"):
        assert column in card, column


def test_card_does_not_claim_the_graph_model_gets_history_features(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports)
    assert "same history features as the graph model" not in card
    assert "message passing" in card
