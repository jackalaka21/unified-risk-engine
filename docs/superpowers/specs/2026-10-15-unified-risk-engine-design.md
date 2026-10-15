# Temporal Graph Risk Engine for Transaction Fraud and Account-Level Risk: Design Spec

Date: 2026-10-16 (revised after the user's review request; supersedes the 2026-10-15 draft)
Repo: `unified-risk-engine` (separate from `deep-feedback-control`). Audience: AI engineering roles, so the emphasis is on trustworthy evaluation, reproducibility, tests and CI, not on a novel method.
Decisions made with the user: real data is IEEE-CIS with a *derived* account-level label; Jupyter notebooks are required; leakage is controlled with **temporal neighbour sampling (Option B)**, with a masked transductive graph (Option A) as the fallback; the project stays in fintech (fraud and risk scoring); engineering add-ons chosen: **CI, a drift and retraining study, a model card with explanation-fidelity checks** (a serving API is out of scope); the repo is private on GitHub.
Working assumption (the user did not answer): the pipeline is built and tested on a generated IEEE-CIS-shaped fixture while the user downloads the real data; the real-data run happens when the files land in `data/`.

## 1. Central question and success criteria

**Question:** under strict temporal evaluation, does a heterogeneous graph neural network add predictive value for transaction fraud over *history-aware* gradient-boosted trees, and does a second, account-level risk head help or hurt? Either answer is a valid result; the project reports whichever is true.

Success:
- The pipeline builds the graph, trains, evaluates and explains end to end on the fixture (tested) and on the real data once supplied.
- No label from a held-out user, from a later period, or from inside the label-delay embargo reaches training or the model's inputs, enforced by tests.
- Comparisons are fair and attributable: history-aware baseline, no-graph ablation, single-task ablations, 3 seeds, metrics defined with prevalence.
- Results are reported in finance-shaped terms (alert budget, amount-weighted recall, expected cost, calibration) as well as PR-AUC.
- A drift and retraining study quantifies how performance decays and whether periodic retraining helps.
- CI passes on GitHub; a model card documents use, data, results and limits; the README reports what was measured, including negative results, with every number checked against the output files.

Non-goals: real credit-default labels (IEEE-CIS has none), fairness analysis (no protected attributes), hyperparameter search, a serving API, other datasets, SHAP, automated Kaggle download or credential handling.

## 2. Data

- Required: `train_transaction.csv` (`TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`, `ProductCD`, `card1`-`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `C*`, `D*`, `M*`, `V*`). Optional: `train_identity.csv` (`DeviceType`, `DeviceInfo`, `id_*`), joined on `TransactionID`. Files live in `data/` (git-ignored, never committed, licensed by Kaggle); `DATA_DIR` overrides the location.
- Missing files give an error naming the expected path; a schema check fails loudly on absent columns.
- Expected shape (from memory, to be verified): about 590k transactions over about 6 months, about 3.5% fraud.
- A **fixture generator** (`tests/fixtures.py`) writes IEEE-CIS-shaped frames (same names and dtypes, planted fraud signal tied to amount, email domain, shared devices and repeated users) so every test runs without the real data.

## 3. Graph schema

| Node type | Key | Features |
|---|---|---|
| `transaction` | `TransactionID` | engineered features (section 5) |
| `user` | `card1` + `addr1` (missing -> sentinel) | none hand-built |
| `email_domain` | `P_emaildomain` | none hand-built |
| `device` | `DeviceInfo` (only if the identity file is present) | none hand-built |

Edges: `user -> transaction`, `transaction -> email_domain`, `transaction -> device`, plus reverse edges. Non-transaction nodes get a single learned vector per node type as input, so information reaches them by message passing. Every node has a `time`: a transaction's is `TransactionDT`; an entity node's is the time of its first transaction.

## 4. Leakage control (Option B), splits and the label-delay embargo

**Periods** by `TransactionDT` quantiles: train = first 70% (`T_train` = its last timestamp); validation = from `T_train + E` to the 85% quantile; test = from the end of validation `+ E` onward. `E` is the **embargo** (default 7 days): transactions inside an embargo window are neither trained on nor evaluated on, because real fraud labels arrive late and would not yet be known. (They remain visible as graph neighbours by their timestamps, features only.)

**Users** are split once at random with a fixed seed, 70/15/15 into train/validation/test users.

**Temporal sampling.** `NeighborLoader` with `time_attr="time"`: for a seed at time `t`, only neighbours with `time <= t` are visible, so no future transaction or entity can influence a prediction. Equal-timestamp visibility depends on the PyG release (strictly earlier in 2.2, earlier-or-equal later); the spike verifies the installed rule and the README states it. Only features, never labels, are ever visible.

**Fraud head:** seeds are transactions at their own times. Train rows = train-period transactions of **train users**. Validation and test rows = all transactions in the validation and test periods (any user), excluding embargo windows.

**Account-risk head:** seeds are users, all at time `T_train`, so a prediction sees only transactions up to `T_train`. Population = users with at least `min_transactions` (default 3) transactions up to `T_train`. **Label (a proxy, not credit data):** positive if the fraud rate over those transactions is at least `fraud_rate_threshold` (default 0.2). Train/validation/test users come from the user split. Thresholds are chosen by prevalence only (never model performance), recorded, and each split needs at least 30 positives or the pipeline raises. The label is a deterministic function of fraud labels, so the two heads are coupled by construction; section 7's ablations test whether that coupling helps.

**Why train rows exclude held-out users:** a held-out user's account label is derived from their fraud labels, so those transactions must never contribute a training loss.

**Enforced by tests:** (1) no fraud-train transaction belongs to a validation or test user; (2) every fraud-train transaction has `time <= T_train`; (3) account labels are unchanged if all fraud labels after `T_train` are shuffled; (4) no validation or test row falls inside an embargo window; (5) on the fixture, no sampled neighbour has `time` greater than its seed's; (6) features and encoders are fitted on the train period only.

**Fallback (Option A):** if the spike shows temporal hetero sampling is unusable, fall back to one transductive graph with masks, tell the user, and add a limitation (future transactions' features visible through neighbours).

## 5. Features (`pipeline/feature_engineering.py`)

- **Transaction features (GNN and baselines):** log-amount; hour of day and day of week; frequency encodings of categoricals (`ProductCD`, `card*`, `addr*`, email domains, `M*`) fitted on the train period; NaN indicators; `C*` and `D*`; a subset of `V*` (missing rate at most 50% in the train period, then the `max_v_columns` (default 100) of highest train-period variance); numeric columns standardised with train-period statistics. Fit statistics come from the train period only.
- **History-aware features (baseline only):** for each transaction, computed from strictly earlier transactions only (expanding windows, never labels): counts, mean and standard deviation of amount for the same user, email domain and device over all prior history and the last 1 and 7 days; seconds since the user's previous transaction; number of distinct devices and email domains the user has used so far. Label-based history ("this user's past confirmed frauds") is excluded from v1 because delayed labels need embargo-aware handling; it is listed as a limitation.

## 6. Model and training

- `models/gnn_backbone.py`: 2-layer `HeteroConv` of GATv2 per relation (hidden 64, 4 heads, dropout 0.2, residual connections), SAGEConv option.
- `models/multi_task_heads.py`: transaction head (fraud logit) and account head (risk logit), each a small MLP.
- `models/losses.py`: Focal Loss (`alpha` default 0.75 weighting positives, `gamma` default 2, configurable).
- Mini-batch training with `NeighborLoader` (`num_neighbors` `[15, 10]`). One seed type per loader, so each step takes **one transaction batch (2048) and one user batch (1024)** from two loaders: `loss = w_fraud * L_fraud + w_default * L_default` (defaults 1.0 and 0.5). AdamW (lr 1e-3, weight decay 1e-5), up to 20 epochs, early stopping (patience 3) on the validation score (the mean of the two heads' PR-AUCs, or the active head's when single-task), best checkpoint kept. GPU (RTX 3070 Ti, 8 GB). Seeds `[0, 1, 2]`.
- **Variants (all in the same codebase, set by config):** `multitask` (default); `fraud_only` and `account_only` (the other head's loss weight is 0 and its batches are skipped); `no_graph` (all edges removed, so the transaction head sees only transaction features; fraud head only, since users then have no neighbours).

## 7. Evaluation, baselines and ablations

- **Baselines (3 seeds each):** (a) `HistGradientBoostingClassifier` on the per-transaction features ("naive tabular"); (b) the same model on per-transaction **plus history-aware features** ("history-aware tabular"), the fair comparison for whether message passing adds anything over feature engineering; (c) for the account head, HGB on user aggregates computed up to `T_train`, and a **trivial baseline**: the mean per user of out-of-fold fraud scores from (b) (user-grouped 5-fold on the train period, so no in-sample label leakage).
- **Attribution table:** naive tabular, history-aware tabular, GNN `no_graph`, GNN `fraud_only`, GNN `multitask`, for the fraud head; `account_only` against `multitask` and the baselines for the account head.
- **Ranking metrics:** PR-AUC (primary), ROC-AUC; prevalence per split always shown.
- **Finance-shaped metrics (fraud head, test period):**
  - *Alert budget:* flag the top `b` fraction of transactions for `b` in {0.5%, 1%, 5%}: recall, precision, and **amount-weighted recall** (flagged fraud amount over total fraud amount).
  - *Expected cost:* threshold chosen on validation to minimise missed-fraud amount plus `review_cost` (default 5 currency units, an explicit assumption) per false alarm; report test cost against flag-none and flag-all.
  - *Calibration:* Brier score, expected calibration error and a reliability curve on raw scores.
  - Recall at 1% FPR with the threshold chosen on validation; the realised test FPR is reported too.
- **Reporting:** every seed's metrics, the mean, and a bootstrap 95% CI over test examples for each seed's PR-AUC. No cross-seed confidence claim from 3 seeds. Per-week metrics across the test period.

## 7b. Drift and retraining study (`scripts/drift_study.py`)

Fraud patterns drift. Using the periods after the train period (validation and test together, minus embargoes; this study tunes nothing):
- **Decay:** weekly PR-AUC of the static model (trained once on the train period) against weeks since training, for the GNN and the history-aware tabular model.
- **Retraining:** retrain every 4 weeks on all labels up to that point minus the embargo, and compare weekly PR-AUC against the static model.
- One seed by default (three if time allows); each weekly point carries a bootstrap CI. The README states the conclusion only as far as the evidence reaches.

## 8. Explainability and fidelity (`scripts/explain.py`)

PyG `Explainer` with `GNNExplainer` for selected transactions (highest-scoring true and false positives) and users: top feature importances and influential edges on the sampled neighbourhood, as a plot and a short report in `reports/`. Checks: **stability** (3 explainer seeds, overlap of the top-10 features) and **fidelity** (PyG's fidelity+/- and characterization score on a sample of 200 nodes). IEEE-CIS features are anonymised (`V123`), so explanations name columns, not business meaning; the README says so. Explanations describe the model, not causes.

## 9. Delivery, CI and the model card

```
unified-risk-engine/
├── .github/workflows/tests.yml        # CI
├── configs/config.yaml
├── pipeline/   data_loading.py  feature_engineering.py  history_features.py  splits.py  graph_builder.py
├── models/     gnn_backbone.py  multi_task_heads.py  losses.py
├── scripts/    train.py  baseline.py  evaluate.py  drift_study.py  explain.py  make_model_card.py
├── notebooks/  01_data_and_graph.ipynb  02_train_and_compare.ipynb  03_drift_and_retraining.ipynb  04_explainability.ipynb
├── tests/      fixtures.py  test_*.py
├── docs/       MODEL_CARD.md  superpowers/{specs,plans}/
├── data/ models/ reports/ plots/      (data and models git-ignored; final plots and CSVs committed)
├── requirements.txt  README.md
```

- Notebooks call the same functions as the scripts. `URE_QUICK=1` runs a tiny smoke version. Notebook 01: data checks, prevalence and threshold choice, split, embargo and leakage checks, graph statistics.
- **CI:** GitHub Actions on Ubuntu, CPU-only, Python 3.14, installing the CPU builds of torch and `pyg-lib` for the pinned versions; runs the fixture test suite (GPU tests skipped) on every push and pull request; a status badge in the README. The project is not called done until CI is green on GitHub (pushing needs the user's approval).
- **Model card:** `docs/MODEL_CARD.md` is generated by `make_model_card.py` from the output CSVs (no hand-copied numbers): intended use, data and licence, splits and embargo, headline metrics with CIs, baselines and ablations, drift behaviour, explanation fidelity, known failure modes and limitations.
- Environment: **its own new virtual environment**, separate from the sibling project's (whose published results used torch 2.11). Python 3.14. Target stack, availability checked 2026-10-15 by dry runs (nothing installed): `torch==2.14.1` for CUDA 13.0 (`https://download.pytorch.org/whl/cu130`; the driver reports CUDA 13.1); `torch-geometric==2.8.1`; `pyg-lib==0.9.0+pt214cu130` from `https://data.pyg.org/whl/torch-2.14.0+cu130.html` (built against torch 2.14.0; the spike confirms it loads and samples correctly under 2.14.1); the latest releases at install time of pandas, scikit-learn, matplotlib, pyyaml, pytest and the jupyter packages, then pinned exactly. **Fallback** if the spike finds a problem: `torch==2.11.0` (CUDA 12.8) with `pyg-lib==0.9.0+pt211cu128` from `https://data.pyg.org/whl/torch-2.11.0+cu128.html`. No `pyg-lib` wheels exist for the CUDA 12.8 builds of torch 2.12 to 2.14.

## 10. Tests

On the fixture: schema and column checks; feature transformer and history features fitted or computed from earlier data only (a test perturbs future rows and asserts unchanged features); graph node and edge counts; the six leakage tests of section 4; focal loss reduces to cross-entropy at gamma 0 and weights positives by alpha; forward shapes for both heads and each variant; short training smoke runs on CPU and GPU; the baselines; metrics on synthetic scores (PR-AUC, recall at fixed FPR, alert-budget and amount-weighted recall, expected cost, Brier and ECE, bootstrap); the drift study on a synthetic drifting stream; explainer smoke test, stability and fidelity metrics; clear errors for missing data, too few positives and an unusable embargo.

## 11. Risks and fallbacks

| Risk | Plan |
|---|---|
| Temporal sampling unsupported or buggy for hetero graphs in PyG 2.8.1 / pyg-lib 0.9 | **First plan task is a spike** on the fixture checking it and the leakage property; otherwise Option A, and tell the user |
| The PyG docs do not say whether `time` can be set per node type on a heterogeneous graph | The spike tests it; if needed give every node type a `time` (entity nodes use first-seen time, as designed) |
| Temporal sampling switches on `disjoint` mode (one subgraph per seed), which can multiply memory and run time | The spike measures memory and speed on a realistically sized fixture before batch sizes and neighbour counts are fixed |
| Equal-timestamp visibility changed between PyG releases | The spike checks the installed rule; the README states it |
| `pyg-lib` built for torch 2.14.0 may not load under 2.14.1; the newest torch may break another dependency | The spike imports and exercises it first; fall back to the torch 2.11 stack |
| `GNNExplainer` or the fidelity metrics do not support the hetero model as written | Spike early; fall back to a homogeneous projection of the neighbourhood or feature-mask-only, and say so |
| Run count (variants x seeds, baselines, drift retrains) exceeds the time budget | Measure one run in the spike, then fix the schedule; drift study defaults to one seed |
| The account proxy is too rare or too easy | Thresholds by prevalence only; at least 30 positives per split; state the coupling plainly; the trivial baseline shows whether the head adds anything |
| CI cannot install the CPU `pyg-lib`/torch combination on the runner | Check in the spike on a branch; adjust pins or Python version for CI only, and document it |
| The real data differs from the fixture | Schema check with clear errors; real-data results are a separate, reported step |

## 12. Honest limitations (go in the README and the model card)

- The account-level label is a proxy derived from fraud history, so the two tasks are coupled by construction. This is fraud and account-risk scoring, not validated credit scoring.
- `card1` + `addr1` approximates a customer identity.
- IEEE-CIS features are anonymised; explanations name columns, not meaning; no fairness analysis is possible.
- One dataset; 3 seeds; no hyperparameter search; the drift study uses one seed by default; results may not transfer.
- The cost model uses an assumed review cost.
- Label-based history features are excluded from the baseline, so a production system could do better.
- Anything explaining *why* a result occurred is a hypothesis unless an experiment tested it.
- Nothing is measured on real data until the files are supplied; only fixture-based tests are verified before that.

## 13. Verification plan

- Spike first, in the new environment: install the target stack; check `pyg-lib` loads under torch 2.14.1; temporal sampling on a heterogeneous graph (per-node-type `time`, the equal-timestamp rule, no neighbour later than its seed, `disjoint` memory and speed); GNNExplainer and fidelity metrics on a hetero model; a CPU-only CI dry run. Then `pytest` green on the fixture.
- Real-data run when the files arrive: record size and prevalence, train all variants and baselines, run the drift study.
- README and model-card numbers are generated or checked programmatically against the output CSVs; test from a fresh clone; CI green on GitHub; a fresh reviewer reviews the whole branch; nothing is pushed without asking.

## 14. Provenance and references

Option B is **not copied from a single paper**; it combines established pieces.
- **Verified this session (read via search):** PyG `NeighborLoader` temporal sampling (`time_attr`, `input_time`, `temporal_strategy` `"uniform"` or `"last"`, automatic `disjoint` mode) and its version-dependent semantics, from the PyG documentation; *Relational Deep Learning: Graph Representation Learning on Relational Databases* (Fey, Hu, Huang, Leskovec and others, arXiv 2312.04615), the closest published precedent for time-consistent sampling on temporal heterogeneous graphs (that PyG's feature descends from it is not confirmed); and, as motivation for the history-aware baseline and the no-graph ablation, GADBench (Tang et al., NeurIPS 2023: tree ensembles with simple neighbourhood aggregation can outperform recent GNNs) and Weber et al. 2019 (Random Forest beat GCN on the Elliptic dataset). Only the abstracts of the last two were read.
- **Cited from memory, not looked up this session:** GAT (Velickovic et al., ICLR 2018), GATv2 (Brody, Alon and Yahav, ICLR 2022), Focal Loss (Lin et al., ICCV 2017), GNNExplainer (Ying et al., NeurIPS 2019), the PyG library (Fey and Lenssen, 2019), and the IEEE-CIS Fraud Detection Kaggle competition (data from Vesta). Check them before citing formally.
- **This project's own design choices, with no published basis claimed:** the user-level split rule keeping held-out users' transactions out of fraud training, the fraud-history account proxy, the alternating-batch multi-task loop, and the embargo and cost-model defaults.
