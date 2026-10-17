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
<!-- RESULTS:END -->

### Findings

<!-- FINDINGS:START -->
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
<!-- LATENCY:END -->
