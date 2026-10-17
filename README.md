# Unified Risk Engine

![tests](https://github.com/jackalaka21/unified-risk-engine/actions/workflows/tests.yml/badge.svg)

A temporal heterogeneous graph neural network for transaction fraud and an account-risk proxy on the IEEE-CIS fraud data, evaluated against history-aware gradient-boosted trees, with ablations, a drift and retraining study, finance-shaped metrics, explanations with fidelity checks, and (phase 2) a scoring service with a latency benchmark.

## The question

Under strict temporal evaluation, does a heterogeneous GNN add predictive value for transaction fraud over *history-aware* tree models (which get hand-built history features, while the GNN must learn history through message passing), and does a second account-level head help or hurt? Either answer is a valid result; this repository reports whichever is true.

Honest scope: IEEE-CIS has no real credit-default labels. The account-level target is a **proxy** derived from fraud history, so the two tasks are coupled by construction. This is joint fraud and account-risk modelling on a shared graph, not validated credit scoring.

## Design in brief

- **Graph:** transaction, user (`card1` + `addr1`), email-domain and device nodes with reverse edges; every node has a `time` (a transaction's `TransactionDT`, an entity's first-seen time).
- **Leakage control:** chronological train/validation/test periods separated by a label-delay **embargo**; users split into disjoint groups; account labels use history up to the end of training only; feature statistics fitted on the train period only; history features use strictly earlier transactions and never labels; neighbour sampling only reaches transactions at or before the seed's time (graph time is arrival order, so transactions that share a second are ordered by ID and a seed never sees a same-second transaction that arrives after it).
- **Models:** 2-layer GATv2 backbone, fraud and account heads, Focal Loss, alternating transaction and user mini-batches. Variants: `multitask`, `fraud_only`, `account_only`, `no_graph`.
- **Metrics:** PR-AUC first; alert-budget recall and precision, amount-weighted recall, expected cost (an assumed review cost), Brier and ECE, recall at 1% FPR, per-week curves; bootstrap intervals over test examples; three seeds give a range, not a confidence interval.

## Results

<!-- RESULTS:START -->
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
<!-- RESULTS:END -->

### Findings

<!-- FINDINGS:START -->
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
<!-- FINDINGS:END -->

## How to run

```bash
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install torch-geometric==2.8.1
.venv/Scripts/python -m pip install "pyg-lib==0.9.0+pt214cu130" -f https://data.pyg.org/whl/torch-2.14.0+cu130.html
.venv/Scripts/python -m pip install -r requirements.txt
```

The data is not included. Download `train_transaction.csv` (and optionally `train_identity.csv`) from the Kaggle competition `ieee-fraud-detection` (a Kaggle account and acceptance of the rules are required) into `data/`, or point `DATA_DIR` at the folder. `python -m scripts.check_data` verifies it.

```bash
python -m pytest -m "not gpu and not slow"      # fixture-based test suite
python -m scripts.run_all --fixture              # whole pipeline on generated data (results are NOT findings)
python -m scripts.run_all                        # real run, needs the data; writes models/multitask_seed0.pt
python -m scripts.sampling_rule                  # test PR-AUC of that checkpoint under the uniform and last neighbour rules
python -m scripts.export_artifacts               # scoring bundle from the same checkpoint
python -m scripts.benchmark_latency              # latency of the served model (default checkpoint: the one above)
python -m scripts.make_model_card --readme README.md
```

Notebooks in `notebooks/` read the result files; they are generated by `python -m scripts.build_notebooks`. CI runs the CPU test suite on GitHub Actions; if the CPU `pyg-lib` wheel were unavailable it would cover only the non-temporal paths.

## Limitations

- The account label is a proxy; `card1` + `addr1` approximates a customer.
- IEEE-CIS features are anonymised, so explanations name columns, not business meaning, and no fairness analysis is possible.
- One dataset, three seeds, no hyperparameter search; the drift study uses one seed by default.
- The cost model uses an assumed review cost; label-based history features are excluded from the baseline.
- Anything explaining *why* a result occurred is a hypothesis unless an ablation tested it.
- Nothing is measured on real data until the Kaggle files are supplied; fixture results never count as findings.

## Layout

`pipeline/` data, splits, features, graph; `models/` backbone, heads, loss; `scripts/` training, baselines, evaluation, drift, explanations, model card; `notebooks/`; `tests/`; `docs/` (spike report, model card, spec and plan under `docs/superpowers/`).

## Scoring service and latency

A FastAPI service scores one transaction at a time. It keeps a time-sorted in-memory neighbour index, builds the same sampled neighbourhood as training (the deterministic `"last"` rule: the most recent neighbours at or before the request time, plus the new transaction itself), runs the model, then appends the transaction to the index.

```bash
python -m scripts.export_artifacts            # bundle from models/multitask_seed0.pt + history before the test period
uvicorn serving.app:app --port 8000           # reads the bundle from artifacts/ (or $URE_ARTIFACTS)
curl localhost:8000/health
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{"TransactionDT": 15000000, "TransactionAmt": 49.0, "card1": 1234}'
curl -X POST localhost:8000/score_batch -H "Content-Type: application/json" -d '{"transactions": [{"TransactionDT": 15000001, "TransactionAmt": 12.5}]}'
```

- `POST /score` and `/score_batch` validate the body (422 when malformed); a request older than the newest indexed transaction gets 409; 503 until a model is loaded.
- Scoring is serialised by one lock because the index is mutable, so extra clients add queueing, not throughput.
- The headline results use the `"uniform"` neighbour rule (training); serving uses `"last"`. The model card reports the test PR-AUC under both.
- A parity test replays transactions through the online scorer and requires the offline `"last"`-rule scores to match; the benchmark refuses to report if it does not. Because graph time is arrival order, the comparison includes transactions that share a timestamp with others.
- `docker build -t ure-scorer .` builds a CPU image from `artifacts/`; CI builds it and smoke-tests it.
- Latency numbers come from one laptop replaying historical data, not a production environment; the 100 ms p99 budget is illustrative.

<!-- LATENCY:START -->
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
<!-- LATENCY:END -->
