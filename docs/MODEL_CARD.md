# Model card: unified risk engine

Configuration hash: `0a4fb49dd420`. Seeds: [0, 1, 2]. Test transactions: 66606; fraud prevalence 0.036.

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
| fraud_only | 0.351 (0.348-0.355) | 0.331-0.371 | 0.308 | 0.012 | 0.043 | 0.782 | 0.108 | 0.095 | 0.699 | 0.193 | 0.357 | 0.311 | 0.430 | 149587.220 | 370373.533 | 0.070 | 0.194 |
| fraud_only_history | 0.338 (0.330-0.347) | 0.318-0.358 | 0.323 | 0.013 | 0.035 | 0.742 | 0.103 | 0.078 | 0.664 | 0.184 | 0.312 | 0.307 | 0.424 | 172197.761 | 370373.533 | 0.072 | 0.194 |
| graph_labels_tabular | 0.520 (0.516-0.524) | 0.499-0.541 | 0.460 | 0.012 | 0.078 | 0.872 | 0.121 | 0.148 | 0.837 | 0.231 | 0.516 | 0.421 | 0.582 | 117395.283 | 370373.533 | 0.023 | 0.004 |
| graph_shared_tabular | 0.464 (0.457-0.469) | 0.443-0.486 | 0.423 | 0.013 | 0.074 | 0.861 | 0.119 | 0.134 | 0.834 | 0.231 | 0.466 | 0.384 | 0.531 | 135126.542 | 370373.533 | 0.025 | 0.005 |
| history_stable_tabular | 0.465 (0.456-0.476) | 0.444-0.487 | 0.434 | 0.014 | 0.075 | 0.858 | 0.119 | 0.131 | 0.831 | 0.230 | 0.455 | 0.384 | 0.531 | 139487.131 | 370373.533 | 0.025 | 0.005 |
| history_tabular | 0.446 (0.438-0.452) | 0.425-0.469 | 0.415 | 0.014 | 0.073 | 0.841 | 0.116 | 0.128 | 0.823 | 0.228 | 0.441 | 0.377 | 0.521 | 136641.191 | 370373.533 | 0.025 | 0.006 |
| multitask | 0.343 (0.332-0.349) | 0.322-0.364 | 0.306 | 0.012 | 0.037 | 0.794 | 0.110 | 0.080 | 0.712 | 0.197 | 0.331 | 0.293 | 0.405 | 153915.310 | 370373.533 | 0.074 | 0.205 |
| multitask_history | 0.318 (0.307-0.326) | 0.299-0.339 | 0.304 | 0.017 | 0.040 | 0.817 | 0.113 | 0.072 | 0.696 | 0.192 | 0.282 | 0.274 | 0.379 | 156105.542 | 370373.533 | 0.074 | 0.202 |
| naive_tabular | 0.462 (0.449-0.469) | 0.440-0.484 | 0.425 | 0.014 | 0.071 | 0.838 | 0.116 | 0.125 | 0.829 | 0.229 | 0.472 | 0.383 | 0.530 | 134285.125 | 370373.533 | 0.025 | 0.006 |
| no_graph | 0.335 (0.328-0.344) | 0.315-0.356 | 0.302 | 0.012 | 0.045 | 0.738 | 0.102 | 0.096 | 0.659 | 0.182 | 0.357 | 0.312 | 0.431 | 172599.068 | 370373.533 | 0.063 | 0.172 |
| stacked_labels_tabular | 0.512 (0.507-0.515) | 0.491-0.534 | 0.469 | 0.013 | 0.079 | 0.860 | 0.119 | 0.144 | 0.822 | 0.227 | 0.509 | 0.420 | 0.581 | 123434.121 | 370373.533 | 0.024 | 0.005 |
| stacked_tabular | 0.398 (0.343-0.457) | 0.378-0.419 | 0.418 | 0.026 | 0.073 | 0.788 | 0.109 | 0.119 | 0.712 | 0.197 | 0.370 | 0.335 | 0.463 | 139791.442 | 370373.533 | 0.030 | 0.014 |

Account risk (proxy label), test users:

| model | pr_auc_mean | pr_auc_min | pr_auc_max |
|---|---|---|---|
| account_only | 0.210 | 0.203 | 0.214 |
| hgb_user_aggregates | 0.133 | 0.116 | 0.151 |
| mean_fraud_score | 0.452 | 0.439 | 0.471 |
| multitask | 0.249 | 0.234 | 0.264 |
| multitask_history | 0.274 | 0.259 | 0.282 |

## Findings

- Highest mean test PR-AUC: graph_labels_tabular (0.520).
- The best graph model is fraud_only (0.351) and the best tabular model is graph_labels_tabular (0.520).
- Best graph model against best tabular model: fraud_only versus graph_labels_tabular, -0.169 PR-AUC.
- Graph structure: fraud_only versus no_graph, +0.016 PR-AUC.
- The second (account) head: multitask versus fraud_only, -0.009 PR-AUC (within the seed range, so not an effect this run can claim).
- History features as specified: history_tabular versus naive_tabular, -0.015 PR-AUC (within the seed range, so not an effect this run can claim).
- History features without cumulative counts: history_stable_tabular versus naive_tabular, +0.004 PR-AUC (within the seed range, so not an effect this run can claim).
- Giving the GNN the history inputs (fraud only): fraud_only_history versus fraud_only, -0.014 PR-AUC (within the seed range, so not an effect this run can claim).
- Giving the GNN the history inputs (multitask): multitask_history versus multitask, -0.024 PR-AUC.
- Users sharing a device or email domain, as tree features: graph_shared_tabular versus history_stable_tabular, -0.001 PR-AUC (within the seed range, so not an effect this run can claim).
- Delayed fraud rates of the entities, as tree features: graph_labels_tabular versus history_stable_tabular, +0.055 PR-AUC.
- The GNN score as a tree feature: stacked_tabular versus history_stable_tabular, -0.067 PR-AUC (within the seed range, so not an effect this run can claim).
- The GNN score added to the delayed-label trees: stacked_labels_tabular versus graph_labels_tabular, -0.008 PR-AUC (within the seed range, so not an effect this run can claim).
- Calibration (ECE, lower is better) runs from 0.004 (graph_labels_tabular) to 0.205 (multitask).
- Account risk: highest mean PR-AUC is mean_fraud_score (0.452); multitask_history 0.274, multitask 0.249, account_only 0.210, hgb_user_aggregates 0.133.
- Drift, gnn_fraud_only: mean weekly PR-AUC 0.352 when trained once and 0.351 with periodic retraining; trend per week +0.0008 (static) and +0.0003 (retrained).
- Drift, history_stable_tabular: mean weekly PR-AUC 0.468 when trained once and 0.471 with periodic retraining; trend per week +0.0014 (static) and +0.0036 (retrained).
- Drift, history_tabular: mean weekly PR-AUC 0.448 when trained once and 0.468 with periodic retraining; trend per week +0.0006 (static) and +0.0068 (retrained).
- Whether dropping the cumulative counts explains any gap between history_tabular and history_stable_tabular is a hypothesis: such counts only grow, so their distribution can differ between the train and test periods, but this pipeline does not test that directly. Likewise, any reason given for a graph model doing better or worse than a tabular one is a hypothesis unless an ablation above tested it.

Bootstrap resamples without both classes are dropped; at least 1.00 of the resamples were kept for every row.

Cost uses an assumed review cost per false alert; compare it with `cost_flag_none`, the cost of flagging nothing. Differences smaller than the bootstrap interval or the range over seeds are not evidence of a better model.

## Drift

| kind | schedule | mean_pr_auc | slope_per_week |
|---|---|---|---|
| gnn_fraud_only | retrain | 0.351 | 0.000 |
| gnn_fraud_only | static | 0.352 | 0.001 |
| history_stable_tabular | retrain | 0.471 | 0.004 |
| history_stable_tabular | static | 0.468 | 0.001 |
| history_tabular | retrain | 0.468 | 0.007 |
| history_tabular | static | 0.448 | 0.001 |

## Explanations

Mode `hetero`. Stability (top-k overlap across runs): 0.40. Deleting the top-k features of the reported (hetero) importances moves the logit by 0.34 on average, against 0.09 for random features. Features are anonymised; explanations describe the model and are not causal.

## Serving and latency

| model | device | mode | batch_size | concurrency | p50_ms | p50 range | p95_ms | p99_ms | throughput_rps |
|---|---|---|---|---|---|---|---|---|---|
| gnn | cpu | inprocess | 1 | 1 | 18.99 | 18.98-18.99 | 21.23 | 22.84 | 53.59 |
| gnn | cpu | inprocess | 8 | 1 | 25.45 | 25.42-25.88 | 28.80 | 31.78 | 315.98 |
| gnn | cpu | inprocess | 32 | 1 | 46.26 | 44.99-46.40 | 50.65 | 62.69 | 700.65 |
| gnn | cpu | inprocess | 128 | 1 | 133.84 | 133.70-140.34 | 153.82 | 158.28 | 943.42 |
| gnn | cuda | inprocess | 1 | 1 | 26.00 | 25.85-26.07 | 33.57 | 37.74 | 37.57 |
| gnn | cuda | inprocess | 8 | 1 | 32.78 | 32.50-33.25 | 40.24 | 44.64 | 238.36 |
| gnn | cuda | inprocess | 32 | 1 | 47.13 | 46.66-47.91 | 54.54 | 68.84 | 664.64 |
| gnn | cuda | inprocess | 128 | 1 | 114.85 | 114.51-116.29 | 134.94 | 140.71 | 1078.23 |
| gnn | cpu | http | 1 | 1 | 20.74 | 20.57-21.81 | 24.94 | 34.00 | 47.07 |
| gnn | cpu | http | 32 | 1 | 66.89 | 60.81-76.08 | 74.65 | 81.55 | 470.13 |
| gnn | cpu | http | 1 | 8 | 267.60 | 244.05-271.67 | 351.76 | 485.29 | 28.34 |
| history_tabular | cpu | inprocess | 1 | 1 | 17.30 | 16.58-17.53 | 23.58 | 29.00 | 55.25 |
| history_tabular | cpu | inprocess | 8 | 1 | 22.17 | 22.15-22.37 | 28.54 | 32.53 | 355.23 |
| history_tabular | cpu | inprocess | 32 | 1 | 39.75 | 39.06-40.78 | 49.87 | 62.67 | 789.00 |
| history_tabular | cpu | inprocess | 128 | 1 | 126.52 | 125.82-129.01 | 148.69 | 153.48 | 983.24 |
| history_tabular | cpu | http | 1 | 1 | 19.75 | 18.61-19.98 | 27.36 | 35.38 | 48.50 |
| history_tabular | cpu | http | 1 | 8 | 221.42 | 216.02-222.55 | 263.33 | 378.13 | 34.97 |

Parity check passed: online and offline scores agree (max absolute difference 0.0e+00 over 200 transactions). Latencies are per call, in milliseconds, the median of 3 runs with the run-to-run range of the p50 beside it; they depend on what else the machine is doing (runs taken while other work was running were clearly slower). Measured on one idle laptop (GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU; the 'cpu' rows run on the CPU). The 100 ms budget is illustrative, not a requirement. One machine replaying historical data is not a production claim.

Test PR-AUC of the served model under each neighbour rule (the headline results use `uniform`, serving uses `last`; `uniform` samples at random, so its number is a single draw):

| rule | pr_auc |
|---|---|
| uniform | 0.344 |
| last | 0.343 |

## Limitations

- One dataset, one split in time; no claim about other portfolios or later periods.
- The account label is a proxy derived from fraud history, not a ground-truth account-takeover label.
- Anonymised features limit what an explanation can say.
- Models trained with focal loss are typically poorly calibrated (see the ECE and Brier columns), so their scores rank transactions but are not probabilities; thresholds are chosen on the validation period.
- Three seeds give a range, not a confidence interval.
- Label-based history features: `graph_labels_tabular` and `stacked_labels_tabular` use the delayed fraud labels of earlier transactions (any user, only those older than the embargo) as features; a deployed system with feedback would have them, but the GNN variants do not. `graph_labels_tabular` was chosen on validation among six variants of the extra features.
- The online scorer serves only models on per-transaction inputs; the history-input GNN, the stacked and the graph-aggregate models are evaluated offline only.
- Nothing here is measured on real data unless the Kaggle files were supplied: check the banner at the top of this card.
- The review cost is an assumption.
- Any explanation of *why* a model wins is a hypothesis unless an ablation tested it.
