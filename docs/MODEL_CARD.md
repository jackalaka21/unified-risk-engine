# Model card: unified risk engine

These results come from a real run on the IEEE-CIS Kaggle training data, not generated data.

Configuration hash: `2098e521aa7d`. Seeds: [0, 1, 2]. Test transactions: 66606; fraud prevalence 0.036.

## Intended use

A research prototype that ranks transactions by fraud risk and users by an account-risk proxy, to study whether a temporal graph model adds value over strong tabular baselines. Not for production decisions, credit, or any decision about a person.

## Data

IEEE-CIS Fraud Detection (Kaggle; Vesta Corporation): card-not-present e-commerce transactions with anonymised features. Users are an approximation built from card and address fields, so "user" is a proxy for a real account holder. Test-set size and prevalence are given above.

## Leakage controls

- Chronological split: train, validation, test, separated by an embargo so late labels cannot cross a boundary.
- Users are split into disjoint train, validation and test groups; the account label uses history up to the end of training only.
- Feature statistics are fitted on the train period only; history features are strictly causal.
- Neighbour sampling only reaches transactions that arrived no later than the seed (equal timestamps are ordered by transaction ID).
- The history-aware baseline gets hand-built history features (counts, amounts and recency per user, email domain and device, from strictly earlier transactions); the graph model gets the same per-transaction features and has to learn history through message passing.

## Results

Transaction fraud, test period:

| model | pr_auc (mean, min-max over seeds) | bootstrap 95% CI (mean of per-seed bounds) | recall_at_fpr | realised_fpr | amount_recall@0.005 | precision@0.005 | recall@0.005 | amount_recall@0.01 | precision@0.01 | recall@0.01 | amount_recall@0.05 | precision@0.05 | recall@0.05 | cost | cost_flag_none | brier | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| account_tabular | 0.514 (0.479-0.571) | 0.493-0.537 | 0.587 | 0.023 | 0.079 | 0.766 | 0.106 | 0.164 | 0.808 | 0.223 | 0.505 | 0.451 | 0.623 | 98249.853 | 370373.533 | 0.026 | 0.011 |
| final_bag_tabular | 0.692 (0.691-0.692) | 0.674-0.709 | 0.634 | 0.013 | 0.100 | 0.980 | 0.136 | 0.205 | 0.941 | 0.260 | 0.616 | 0.511 | 0.706 | 94305.076 | 370373.533 | 0.018 | 0.002 |
| final_refit_tabular | 0.713 (0.712-0.714) | 0.696-0.730 | n/a | n/a | 0.092 | 0.972 | 0.134 | 0.209 | 0.941 | 0.260 | 0.623 | 0.523 | 0.723 | n/a | 370373.533 | 0.017 | 0.001 |
| final_tabular | 0.548 (0.397-0.635) | 0.529-0.569 | 0.599 | 0.019 | 0.076 | 0.709 | 0.098 | 0.154 | 0.736 | 0.204 | 0.554 | 0.476 | 0.658 | 96875.411 | 370373.533 | 0.023 | 0.007 |
| final_tuned_tabular | 0.681 (0.679-0.684) | 0.663-0.698 | 0.629 | 0.013 | 0.103 | 0.975 | 0.135 | 0.208 | 0.938 | 0.260 | 0.605 | 0.506 | 0.700 | 100369.512 | 370373.533 | 0.018 | 0.002 |
| fraud_only | 0.347 (0.341-0.356) | 0.327-0.368 | 0.303 | 0.012 | 0.043 | 0.778 | 0.108 | 0.096 | 0.683 | 0.189 | 0.355 | 0.312 | 0.432 | 156333.143 | 370373.533 | 0.068 | 0.189 |
| fraud_only_history | 0.342 (0.325-0.351) | 0.322-0.362 | 0.319 | 0.013 | 0.036 | 0.728 | 0.101 | 0.083 | 0.664 | 0.184 | 0.349 | 0.313 | 0.432 | 159939.603 | 370373.533 | 0.071 | 0.194 |
| graph_labels_tabular | 0.520 (0.516-0.524) | 0.499-0.541 | 0.460 | 0.012 | 0.078 | 0.872 | 0.121 | 0.148 | 0.837 | 0.231 | 0.516 | 0.421 | 0.582 | 117395.283 | 370373.533 | 0.023 | 0.004 |
| graph_shared_tabular | 0.464 (0.457-0.469) | 0.443-0.486 | 0.423 | 0.013 | 0.074 | 0.861 | 0.119 | 0.134 | 0.834 | 0.231 | 0.466 | 0.384 | 0.531 | 135126.542 | 370373.533 | 0.025 | 0.005 |
| history_stable_tabular | 0.465 (0.456-0.476) | 0.444-0.487 | 0.434 | 0.014 | 0.075 | 0.858 | 0.119 | 0.131 | 0.831 | 0.230 | 0.455 | 0.384 | 0.531 | 139487.131 | 370373.533 | 0.025 | 0.005 |
| history_tabular | 0.446 (0.438-0.452) | 0.425-0.469 | 0.415 | 0.014 | 0.073 | 0.841 | 0.116 | 0.128 | 0.823 | 0.228 | 0.441 | 0.377 | 0.521 | 136641.191 | 370373.533 | 0.025 | 0.006 |
| multitask | 0.337 (0.324-0.351) | 0.317-0.357 | 0.306 | 0.013 | 0.037 | 0.798 | 0.110 | 0.078 | 0.710 | 0.196 | 0.322 | 0.288 | 0.399 | 161078.404 | 370373.533 | 0.070 | 0.195 |
| multitask_history | 0.325 (0.309-0.341) | 0.305-0.345 | 0.307 | 0.017 | 0.038 | 0.810 | 0.112 | 0.082 | 0.703 | 0.194 | 0.295 | 0.278 | 0.385 | 158796.251 | 370373.533 | 0.076 | 0.206 |
| naive_tabular | 0.462 (0.449-0.469) | 0.440-0.484 | 0.425 | 0.014 | 0.071 | 0.838 | 0.116 | 0.125 | 0.829 | 0.229 | 0.472 | 0.383 | 0.530 | 134285.125 | 370373.533 | 0.025 | 0.006 |
| no_graph | 0.335 (0.328-0.344) | 0.315-0.356 | 0.302 | 0.012 | 0.045 | 0.738 | 0.102 | 0.096 | 0.659 | 0.182 | 0.357 | 0.312 | 0.431 | 172599.068 | 370373.533 | 0.063 | 0.172 |
| stacked_labels_tabular | 0.490 (0.451-0.512) | 0.469-0.512 | 0.461 | 0.014 | 0.080 | 0.812 | 0.112 | 0.145 | 0.789 | 0.218 | 0.521 | 0.413 | 0.572 | 125776.279 | 370373.533 | 0.025 | 0.006 |
| stacked_tabular | 0.404 (0.382-0.423) | 0.384-0.426 | 0.409 | 0.028 | 0.078 | 0.830 | 0.115 | 0.134 | 0.795 | 0.220 | 0.349 | 0.322 | 0.446 | 140199.752 | 370373.533 | 0.029 | 0.014 |

Account risk (proxy label), test users:

| model | pr_auc_mean | pr_auc_min | pr_auc_max |
|---|---|---|---|
| account_only | 0.216 | 0.207 | 0.225 |
| hgb_user_aggregates | 0.133 | 0.116 | 0.151 |
| mean_fraud_score | 0.452 | 0.439 | 0.471 |
| multitask | 0.246 | 0.220 | 0.264 |
| multitask_history | 0.270 | 0.256 | 0.278 |

## Findings

- Highest mean test PR-AUC: final_refit_tabular (0.713).
- The best graph model is fraud_only (0.347) and the best tabular model is final_refit_tabular (0.713).
- Best graph model against best tabular model: fraud_only versus final_refit_tabular, -0.366 PR-AUC.
- Graph structure: fraud_only versus no_graph, +0.012 PR-AUC (within the seed range, so not an effect this run can claim).
- The second (account) head: multitask versus fraud_only, -0.011 PR-AUC (within the seed range, so not an effect this run can claim).
- History features as specified: history_tabular versus naive_tabular, -0.015 PR-AUC (within the seed range, so not an effect this run can claim).
- History features without cumulative counts: history_stable_tabular versus naive_tabular, +0.004 PR-AUC (within the seed range, so not an effect this run can claim).
- Giving the GNN the history inputs (fraud only): fraud_only_history versus fraud_only, -0.006 PR-AUC (within the seed range, so not an effect this run can claim).
- Giving the GNN the history inputs (multitask): multitask_history versus multitask, -0.012 PR-AUC (within the seed range, so not an effect this run can claim).
- Users sharing a device or email domain, as tree features: graph_shared_tabular versus history_stable_tabular, -0.001 PR-AUC (within the seed range, so not an effect this run can claim).
- Delayed fraud rates of the entities, as tree features: graph_labels_tabular versus history_stable_tabular, +0.055 PR-AUC.
- A finer account identity from D1, as tree features: account_tabular versus graph_labels_tabular, -0.006 PR-AUC (within the seed range, so not an effect this run can claim).
- 30-day windowed fraud rates, as tree features: final_tabular versus account_tabular, +0.034 PR-AUC (within the seed range, so not an effect this run can claim).
- Tuned tree hyper-parameters: final_tuned_tabular versus final_tabular, +0.132 PR-AUC (within the seed range, so not an effect this run can claim).
- Bagging five tuned trees: final_bag_tabular versus final_tuned_tabular, +0.011 PR-AUC.
- Refitting the bag on train + validation: final_refit_tabular versus final_bag_tabular, +0.021 PR-AUC.
- The GNN score as a tree feature: stacked_tabular versus history_stable_tabular, -0.061 PR-AUC.
- The GNN score added to the delayed-label trees: stacked_labels_tabular versus graph_labels_tabular, -0.030 PR-AUC (within the seed range, so not an effect this run can claim).
- Calibration (ECE, lower is better) runs from 0.001 (final_refit_tabular) to 0.206 (multitask_history).
- Account risk: highest mean PR-AUC is mean_fraud_score (0.452); multitask_history 0.270, multitask 0.246, account_only 0.216, hgb_user_aggregates 0.133.
- Drift, gnn_fraud_only: mean weekly PR-AUC 0.358 when trained once and 0.352 with periodic retraining; trend per week +0.0016 (static) and -0.0010 (retrained).
- Drift, history_stable_tabular: mean weekly PR-AUC 0.468 when trained once and 0.471 with periodic retraining; trend per week +0.0014 (static) and +0.0036 (retrained).
- Drift, history_tabular: mean weekly PR-AUC 0.448 when trained once and 0.468 with periodic retraining; trend per week +0.0006 (static) and +0.0068 (retrained).
- Whether dropping the cumulative counts explains any gap between history_tabular and history_stable_tabular is a hypothesis: such counts only grow, so their distribution can differ between the train and test periods, but this pipeline does not test that directly. Likewise, any reason given for a graph model doing better or worse than a tabular one is a hypothesis unless an ablation above tested it.

Bootstrap resamples without both classes are dropped; at least 1.00 of the resamples were kept for every row.

Cost uses an assumed review cost per false alert; compare it with `cost_flag_none`, the cost of flagging nothing. Differences smaller than the bootstrap interval or the range over seeds are not evidence of a better model.

## Drift

| kind | schedule | mean_pr_auc | slope_per_week |
|---|---|---|---|
| gnn_fraud_only | retrain | 0.352 | -0.001 |
| gnn_fraud_only | static | 0.358 | 0.002 |
| history_stable_tabular | retrain | 0.471 | 0.004 |
| history_stable_tabular | static | 0.468 | 0.001 |
| history_tabular | retrain | 0.468 | 0.007 |
| history_tabular | static | 0.448 | 0.001 |

## Explanations

Mode `hetero`. Stability (top-k overlap across runs): 0.42. Deleting the top-k features of the reported (hetero) importances moves the logit by 0.24 on average, against 0.06 for random features. Features are anonymised; explanations describe the model and are not causal.

## Serving and latency

| model | device | mode | batch_size | concurrency | p50_ms | p50 range | p95_ms | p99_ms | throughput_rps |
|---|---|---|---|---|---|---|---|---|---|
| gnn | cpu | inprocess | 1 | 1 | 19.04 | 19.00-19.19 | 21.13 | 22.86 | 53.40 |
| gnn | cpu | inprocess | 8 | 1 | 25.80 | 25.76-26.16 | 28.77 | 30.57 | 313.89 |
| gnn | cpu | inprocess | 32 | 1 | 44.22 | 43.80-45.25 | 51.66 | 60.81 | 715.58 |
| gnn | cpu | inprocess | 128 | 1 | 135.95 | 134.02-143.27 | 151.73 | 153.80 | 935.98 |
| gnn | cuda | inprocess | 1 | 1 | 25.88 | 25.45-26.06 | 33.45 | 37.63 | 37.59 |
| gnn | cuda | inprocess | 8 | 1 | 32.73 | 32.23-32.80 | 39.82 | 42.72 | 240.65 |
| gnn | cuda | inprocess | 32 | 1 | 46.97 | 46.77-47.28 | 55.39 | 65.24 | 668.04 |
| gnn | cuda | inprocess | 128 | 1 | 119.21 | 119.00-119.30 | 133.01 | 136.11 | 1073.71 |
| gnn | cpu | http | 1 | 1 | 21.00 | 20.94-21.28 | 24.15 | 25.55 | 46.93 |
| gnn | cpu | http | 32 | 1 | 55.34 | 55.16-57.70 | 61.86 | 63.83 | 566.93 |
| gnn | cpu | http | 1 | 8 | 231.37 | 231.17-231.52 | 242.86 | 257.78 | 34.41 |
| history_tabular | cpu | inprocess | 1 | 1 | 14.87 | 14.83-15.08 | 16.79 | 18.38 | 69.75 |
| history_tabular | cpu | inprocess | 8 | 1 | 20.11 | 20.08-20.33 | 22.64 | 24.55 | 406.29 |
| history_tabular | cpu | inprocess | 32 | 1 | 37.60 | 37.15-38.54 | 42.32 | 45.69 | 877.85 |
| history_tabular | cpu | inprocess | 128 | 1 | 116.59 | 112.82-118.24 | 128.98 | 134.23 | 1099.18 |
| history_tabular | cpu | http | 1 | 1 | 18.17 | 16.81-20.53 | 24.70 | 31.66 | 53.20 |
| history_tabular | cpu | http | 1 | 8 | 331.47 | 254.59-333.41 | 489.41 | 983.21 | 21.23 |

Parity check passed: online and offline scores agree (max absolute difference 0.0e+00 over 200 transactions). Latencies are per call, in milliseconds, the median of 3 runs with the run-to-run range of the p50 beside it; they depend on what else the machine is doing. The background CPU load just before each measurement averaged 11% (peak 74%); a busy machine slows the model more than the trees. Measured on one laptop (GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU; the 'cpu' rows run on the CPU). The 100 ms budget is illustrative, not a requirement. One machine replaying historical data is not a production claim.

Test PR-AUC of the served model under each neighbour rule (the headline results use `uniform`, serving uses `last`; `uniform` samples at random, so its number is a single draw):

| rule | pr_auc |
|---|---|
| uniform | 0.326 |
| last | 0.328 |

## Limitations

- One dataset, one split in time; no claim about other portfolios or later periods.
- The account label is a proxy derived from fraud history, not a ground-truth account-takeover label.
- Anonymised features limit what an explanation can say.
- Models trained with focal loss are typically poorly calibrated (see the ECE and Brier columns), so their scores rank transactions but are not probabilities; thresholds are chosen on the validation period.
- Three seeds give a range, not a confidence interval.
- Label-based history features: `graph_labels_tabular` and `stacked_labels_tabular` use the delayed fraud labels of earlier transactions (any user, only those older than the embargo) as features; a deployed system with feedback would have them, but the GNN variants do not. `graph_labels_tabular` was chosen on validation among six variants of the extra features.
- The strongest features remember an account's earlier fraud labels (the delayed rates and the D1-based account identity). My recollection, not checked here, is that IEEE-CIS fraud labels were propagated across a customer's transactions when the data was labelled; if so, these features partly exploit how the dataset was built, and a real system may see weaker label clustering. This is a hypothesis, not a finding.
- The GNN variants were not tuned, not bagged and not refit on validation, and have no D1-based account identity or windowed label features, so the best tree models had more done to them; `final_refit_tabular` has no validation-threshold metrics (recall at FPR, cost) because its validation scores are in-sample. The account identity relies on the D1 column, which must be present in every request.
- The online scorer serves only models on per-transaction inputs; the history-input GNN, the stacked and the graph-aggregate models are evaluated offline only.
- Runs on generated fixture data exist only for the test suite and never count as findings; a card built from them carries a banner at the top. Every number in this card is measured on real data.
- The review cost is an assumption.
- Any explanation of *why* a model wins is a hypothesis unless an ablation tested it.
