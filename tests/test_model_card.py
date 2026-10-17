import json

import pandas as pd
import pytest

from scripts.make_model_card import build_model_card, results_markdown, splice_readme


def write_tables(tmp_path, hash_a="aaaa", hash_b="aaaa"):
    out, reports = tmp_path / "plots", tmp_path / "reports"
    out.mkdir(), reports.mkdir()
    pd.DataFrame({"model": ["history_tabular", "history_tabular", "multitask", "multitask"], "seed": [0, 1, 0, 1],
                  "config_hash": [hash_a, hash_a, hash_b, hash_b], "n_test": 1000, "prevalence": 0.03,
                  "pr_auc": [0.50, 0.52, 0.5678, 0.5722], "pr_auc_lo": 0.4, "pr_auc_hi": 0.7, "pr_auc_boot_kept": 0.97, "roc_auc": 0.9,
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


from scripts.make_model_card import splice_block


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


def test_weekly_and_calibration_tables_are_part_of_the_config_hash_check(tmp_path):
    out, reports = write_tables(tmp_path)
    pd.DataFrame({"model": "m", "seed": 0, "week": [0], "pr_auc": [0.5], "config_hash": "zzzz"}).to_csv(out / "results_weekly.csv", index=False)
    with pytest.raises(ValueError, match="mixed config hashes"):
        build_model_card(out, reports)
    pd.DataFrame({"model": "m", "seed": 0, "week": [0], "pr_auc": [0.5]}).to_csv(out / "results_weekly.csv", index=False)
    with pytest.raises(ValueError, match="config_hash"):
        build_model_card(out, reports)


def test_tables_flagged_as_fixture_cannot_enter_a_real_card(tmp_path):
    out, reports = write_tables(tmp_path)
    fraud = pd.read_csv(out / "results_fraud.csv").assign(fixture=True)
    fraud.to_csv(out / "results_fraud.csv", index=False)
    with pytest.raises(ValueError, match="fixture"):
        build_model_card(out, reports, fixture=False)
    assert "FIXTURE RESULTS" in build_model_card(out, reports, fixture=True)


def test_explanation_metrics_flagged_as_fixture_cannot_enter_a_real_card(tmp_path):
    out, reports = write_tables(tmp_path)
    meta = json.loads((reports / "explain_metrics.json").read_text())
    meta["fixture"] = True
    (reports / "explain_metrics.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fixture"):
        build_model_card(out, reports, fixture=False)


def test_the_card_types_no_dataset_numbers_and_states_every_known_limit(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports)
    assert "590k" not in card and "six months" not in card
    assert "label-based history" in card.lower() and "real data" in card.lower()
    assert "bootstrap resamples" in card.lower() and "0.97" in card


def _frames(pr):
    """pr: model -> per-seed PR-AUCs; the other columns are filler."""
    rows = [{"model": m, "seed": s, "pr_auc": v, "ece": 0.01 + 0.1 * (m == "multitask"), "config_hash": "aaaa"}
            for m, vs in pr.items() for s, v in enumerate(vs)]
    return pd.DataFrame(rows)


def test_findings_are_computed_from_the_tables_with_directions_and_sizes():
    from scripts.make_model_card import findings_markdown

    fraud = _frames({"naive_tabular": [0.47, 0.46, 0.48], "history_tabular": [0.45, 0.45, 0.45],
                     "history_stable_tabular": [0.49, 0.48, 0.50], "no_graph": [0.33, 0.34, 0.33],
                     "fraud_only": [0.35, 0.36, 0.34], "multitask": [0.32, 0.33, 0.34]})
    md = findings_markdown(fraud, pd.DataFrame({"model": ["multitask", "mean_fraud_score"], "seed": 0, "pr_auc": [0.25, 0.45]}))
    assert "Highest mean test PR-AUC: history_stable_tabular (0.490)" in md
    assert "best graph model is fraud_only (0.350)" in md
    assert "fraud_only versus no_graph" in md and "+0.017" in md                 # graph structure
    assert "multitask versus fraud_only" in md and "-0.020" in md                # the account head
    assert "history_tabular versus naive_tabular" in md and "-0.020" in md
    assert "history_stable_tabular versus naive_tabular" in md and "+0.020" in md
    assert "mean_fraud_score" in md                                              # the trivial account baseline


def test_differences_inside_the_seed_range_are_not_presented_as_effects():
    from scripts.make_model_card import findings_markdown

    fraud = _frames({"no_graph": [0.30, 0.40], "fraud_only": [0.34, 0.36]})
    line = [l for l in findings_markdown(fraud, None).splitlines() if "fraud_only versus no_graph" in l][0]
    assert "within the seed range" in line


def test_findings_tolerate_missing_models_and_state_no_causes():
    from scripts.make_model_card import findings_markdown

    md = findings_markdown(_frames({"multitask": [0.4, 0.41]}), None)
    assert "multitask" in md and "because" not in md.lower() and "caused" not in md.lower()
    assert findings_markdown(_frames({"naive_tabular": [0.5]}), None)               # a single seed has no range: still fine


def test_the_card_has_a_findings_section_and_the_cumulative_count_note_is_a_hypothesis(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports)
    assert "## Findings" in card
    assert "hypothesis" in card.lower() and "cumulative" in card.lower()


def test_card_text_is_clean(tmp_path):
    out, reports = write_tables(tmp_path)
    write_latency(out)
    pd.read_csv(out / "drift_summary.csv").assign(fixture=False).to_csv(out / "drift_summary.csv", index=False)
    card = build_model_card(out, reports)
    assert "features, Users" not in card and "features. Users" in card
    drift = card.split("## Drift")[1].split("## Explanations")[0]
    assert "fixture" not in drift and "config_hash" not in drift
    serving = card.split("## Serving and latency")[1].split("## Limitations")[0]
    assert "'cpu' rows run on the CPU" in serving and "| 3.20 |" in serving      # two-decimal latencies, not 3.200


def test_findings_cover_history_inputs_stacking_and_graph_aggregates():
    from scripts.make_model_card import findings_markdown

    fraud = _frames({"history_stable_tabular": [0.46, 0.47, 0.46], "fraud_only": [0.35, 0.35, 0.35], "multitask": [0.34, 0.34, 0.34],
                     "fraud_only_history": [0.44, 0.45, 0.44], "multitask_history": [0.43, 0.44, 0.43],
                     "stacked_tabular": [0.50, 0.50, 0.50], "graph_shared_tabular": [0.45, 0.45, 0.45],
                     "graph_labels_tabular": [0.52, 0.52, 0.52], "stacked_labels_tabular": [0.55, 0.55, 0.55]})
    md = findings_markdown(fraud, None)
    for text in ("fraud_only_history versus fraud_only", "multitask_history versus multitask", "stacked_tabular versus history_stable_tabular",
                 "graph_shared_tabular versus history_stable_tabular", "graph_labels_tabular versus history_stable_tabular",
                 "stacked_labels_tabular versus graph_labels_tabular"):
        assert text in md, text
    assert "Highest mean test PR-AUC: stacked_labels_tabular (0.550)" in md
    assert "best graph model is fraud_only_history" in md                       # history-input GNNs count as graph models


def test_limitations_disclose_delayed_labels_and_the_validation_selection(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports).lower()
    assert "delayed" in card and "chosen on validation" in card and "online scorer" in card


def test_latency_table_reports_the_median_over_repeats_and_the_run_to_run_range(tmp_path):
    from scripts.make_model_card import latency_markdown

    rows = [{"model": "gnn", "device": "cpu", "mode": "inprocess", "batch_size": 1, "concurrency": 1, "repeat": r,
             "p50_ms": v, "p95_ms": v + 5, "p99_ms": v + 10, "throughput_rps": 1000 / v, "p99_within_budget": True}
            for r, v in enumerate((20.0, 30.0, 40.0))]
    meta = {"budget_ms": 100.0, "hardware": {"gpu": "g"}, "parity": {"checked": True, "max_abs_diff": 0.0, "n_rows": 5, "reason": ""}}
    md = latency_markdown(pd.DataFrame(rows), meta)
    assert "| 30.00 |" in md and "20.00-40.00" in md                       # median p50 and its run-to-run range
    assert "median of 3 runs" in md


def test_missing_threshold_metrics_are_shown_as_not_available(tmp_path):
    out, reports = write_tables(tmp_path)
    fraud = pd.read_csv(out / "results_fraud.csv")
    fraud.loc[fraud["model"] == "multitask", ["recall_at_fpr", "cost"]] = float("nan")
    fraud.to_csv(out / "results_fraud.csv", index=False)
    assert "n/a" in build_model_card(out, reports) and "nan" not in build_model_card(out, reports).lower().split("## results")[1].split("## findings")[0]


def test_findings_cover_the_account_identity_windowed_rates_tuning_bagging_and_refit():
    from scripts.make_model_card import findings_markdown

    fraud = _frames({"graph_labels_tabular": [0.52, 0.52, 0.52], "account_tabular": [0.60, 0.60, 0.60], "final_tabular": [0.62, 0.62, 0.62],
                     "final_tuned_tabular": [0.64, 0.64, 0.64], "final_bag_tabular": [0.65, 0.65, 0.65],
                     "final_refit_tabular": [0.68, 0.68, 0.68], "fraud_only": [0.35, 0.35, 0.35]})
    md = findings_markdown(fraud, None)
    for text in ("account_tabular versus graph_labels_tabular", "final_tabular versus account_tabular",
                 "final_tuned_tabular versus final_tabular", "final_bag_tabular versus final_tuned_tabular",
                 "final_refit_tabular versus final_bag_tabular"):
        assert text in md, text
    assert "Highest mean test PR-AUC: final_refit_tabular (0.680)" in md
    assert "best tabular model is final_refit_tabular" in md


def test_limitations_state_what_the_gnn_did_not_get(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports).lower()
    assert "not tuned" in card and "refit" in card and "d1" in card


def test_limitations_warn_that_label_memory_features_may_exploit_how_the_dataset_was_labelled(tmp_path):
    out, reports = write_tables(tmp_path)
    card = build_model_card(out, reports)
    limitations = card.split("## Limitations")[1].lower()
    assert "propagated" in limitations and "hypothesis" in limitations and "labelled" in limitations
    assert "account" in limitations and "earlier fraud labels" in limitations
