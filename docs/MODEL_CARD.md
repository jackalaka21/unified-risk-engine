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
| fraud_only | 0.349 (0.345-0.358) | 0.330-0.369 | 0.307 | 0.012 | 0.039 | 0.768 | 0.106 | 0.089 | 0.679 | 0.188 | 0.366 | 0.312 | 0.432 | 150521.539 | 370373.533 | 0.071 | 0.197 |
| history_stable_tabular | 0.465 (0.456-0.476) | 0.444-0.487 | 0.434 | 0.014 | 0.075 | 0.858 | 0.119 | 0.131 | 0.831 | 0.230 | 0.455 | 0.384 | 0.531 | 139487.131 | 370373.533 | 0.025 | 0.005 |
| history_tabular | 0.446 (0.438-0.452) | 0.425-0.469 | 0.415 | 0.014 | 0.073 | 0.841 | 0.116 | 0.128 | 0.823 | 0.228 | 0.441 | 0.377 | 0.521 | 136641.191 | 370373.533 | 0.025 | 0.006 |
| multitask | 0.341 (0.334-0.350) | 0.321-0.362 | 0.308 | 0.012 | 0.037 | 0.797 | 0.110 | 0.084 | 0.701 | 0.194 | 0.352 | 0.295 | 0.408 | 154641.457 | 370373.533 | 0.072 | 0.200 |
| naive_tabular | 0.462 (0.449-0.469) | 0.440-0.484 | 0.425 | 0.014 | 0.071 | 0.838 | 0.116 | 0.125 | 0.829 | 0.229 | 0.472 | 0.383 | 0.530 | 134285.125 | 370373.533 | 0.025 | 0.006 |
| no_graph | 0.335 (0.328-0.344) | 0.315-0.356 | 0.302 | 0.012 | 0.045 | 0.738 | 0.102 | 0.096 | 0.659 | 0.182 | 0.357 | 0.312 | 0.431 | 172599.068 | 370373.533 | 0.063 | 0.172 |

Account risk (proxy label), test users:

| model | pr_auc_mean | pr_auc_min | pr_auc_max |
|---|---|---|---|
| account_only | 0.215 | 0.210 | 0.224 |
| hgb_user_aggregates | 0.133 | 0.116 | 0.151 |
| mean_fraud_score | 0.452 | 0.439 | 0.471 |
| multitask | 0.247 | 0.226 | 0.263 |

## Findings

- Highest mean test PR-AUC: history_stable_tabular (0.465).
- The best graph model is fraud_only (0.349) and the best tabular model is history_stable_tabular (0.465).
- Best graph model against best tabular model: fraud_only versus history_stable_tabular, -0.116 PR-AUC.
- Graph structure: fraud_only versus no_graph, +0.014 PR-AUC (within the seed range, so not an effect this run can claim).
- The second (account) head: multitask versus fraud_only, -0.008 PR-AUC (within the seed range, so not an effect this run can claim).
- History features as specified: history_tabular versus naive_tabular, -0.015 PR-AUC (within the seed range, so not an effect this run can claim).
- History features without cumulative counts: history_stable_tabular versus naive_tabular, +0.004 PR-AUC (within the seed range, so not an effect this run can claim).
- Calibration (ECE, lower is better) runs from 0.005 (history_stable_tabular) to 0.200 (multitask).
- Account risk: highest mean PR-AUC is mean_fraud_score (0.452); multitask 0.247, account_only 0.215, hgb_user_aggregates 0.133.
- Drift, gnn_fraud_only: mean weekly PR-AUC 0.353 when trained once and 0.346 with periodic retraining; trend per week -0.0004 (static) and -0.0031 (retrained).
- Drift, history_stable_tabular: mean weekly PR-AUC 0.468 when trained once and 0.471 with periodic retraining; trend per week +0.0014 (static) and +0.0036 (retrained).
- Drift, history_tabular: mean weekly PR-AUC 0.448 when trained once and 0.468 with periodic retraining; trend per week +0.0006 (static) and +0.0068 (retrained).
- Whether dropping the cumulative counts explains any gap between history_tabular and history_stable_tabular is a hypothesis: such counts only grow, so their distribution can differ between the train and test periods, but this pipeline does not test that directly. Likewise, any reason given for a graph model doing better or worse than a tabular one is a hypothesis unless an ablation above tested it.

Bootstrap resamples without both classes are dropped; at least 1.00 of the resamples were kept for every row.

Cost uses an assumed review cost per false alert; compare it with `cost_flag_none`, the cost of flagging nothing. Differences smaller than the bootstrap interval or the range over seeds are not evidence of a better model.

## Drift

| kind | schedule | mean_pr_auc | slope_per_week |
|---|---|---|---|
| gnn_fraud_only | retrain | 0.346 | -0.003 |
| gnn_fraud_only | static | 0.353 | -0.000 |
| history_stable_tabular | retrain | 0.471 | 0.004 |
| history_stable_tabular | static | 0.468 | 0.001 |
| history_tabular | retrain | 0.468 | 0.007 |
| history_tabular | static | 0.448 | 0.001 |

## Explanations

Mode `hetero`. Stability (top-k overlap across runs): 0.42. Deleting the top-k features of the reported (hetero) importances moves the logit by 0.31 on average, against 0.09 for random features. Features are anonymised; explanations describe the model and are not causal.

## Serving and latency

| model | device | mode | batch_size | concurrency | p50_ms | p95_ms | p99_ms | throughput_rps |
|---|---|---|---|---|---|---|---|---|
| gnn | cpu | inprocess | 1 | 1 | 18.88 | 22.58 | 26.44 | 53.01 |
| gnn | cpu | inprocess | 8 | 1 | 25.51 | 28.86 | 30.27 | 317.02 |
| gnn | cpu | inprocess | 32 | 1 | 46.35 | 50.27 | 69.96 | 700.93 |
| gnn | cpu | inprocess | 128 | 1 | 140.13 | 158.32 | 161.91 | 907.76 |
| gnn | cuda | inprocess | 1 | 1 | 25.78 | 33.83 | 39.66 | 37.65 |
| gnn | cuda | inprocess | 8 | 1 | 32.80 | 41.95 | 49.63 | 237.83 |
| gnn | cuda | inprocess | 32 | 1 | 47.32 | 54.91 | 61.37 | 663.43 |
| gnn | cuda | inprocess | 128 | 1 | 119.54 | 127.59 | 128.86 | 1076.84 |
| gnn | cpu | http | 1 | 1 | 21.28 | 29.05 | 37.28 | 45.52 |
| gnn | cpu | http | 32 | 1 | 55.83 | 64.15 | 67.40 | 561.06 |
| gnn | cpu | http | 1 | 8 | 230.44 | 274.05 | 384.48 | 33.46 |
| history_tabular | cpu | inprocess | 1 | 1 | 14.91 | 19.22 | 21.75 | 67.35 |
| history_tabular | cpu | inprocess | 8 | 1 | 20.22 | 25.91 | 27.64 | 393.79 |
| history_tabular | cpu | inprocess | 32 | 1 | 36.53 | 41.82 | 585.34 | 553.96 |
| history_tabular | cpu | inprocess | 128 | 1 | 116.37 | 131.77 | 142.18 | 1095.93 |
| history_tabular | cpu | http | 1 | 1 | 16.65 | 20.60 | 23.73 | 59.37 |
| history_tabular | cpu | http | 1 | 8 | 194.04 | 206.58 | 263.33 | 40.70 |

Parity check passed: online and offline scores agree (max absolute difference 0.0e+00 over 200 transactions). Latencies are per call, in milliseconds, measured on one laptop (GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU; the 'cpu' rows run on the CPU). The 100 ms budget is illustrative, not a requirement. One machine replaying historical data is not a production claim.

Test PR-AUC of the served model under each neighbour rule (the headline results use `uniform`, serving uses `last`; `uniform` samples at random, so its number is a single draw):

| rule | pr_auc |
|---|---|
| uniform | 0.346 |
| last | 0.350 |

## Limitations

- One dataset, one split in time; no claim about other portfolios or later periods.
- The account label is a proxy derived from fraud history, not a ground-truth account-takeover label.
- Anonymised features limit what an explanation can say.
- Models trained with focal loss are typically poorly calibrated (see the ECE and Brier columns), so their scores rank transactions but are not probabilities; thresholds are chosen on the validation period.
- Three seeds give a range, not a confidence interval.
- Label-based history features (a user's past confirmed frauds) are excluded from the baseline, so a production system could do better.
- Nothing here is measured on real data unless the Kaggle files were supplied: check the banner at the top of this card.
- The review cost is an assumption.
- Any explanation of *why* a model wins is a hypothesis unless an ablation tested it.
