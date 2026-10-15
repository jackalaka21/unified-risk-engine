# Unified Credit & Fraud Adaptive Risk Engine: Design Spec

Date: 2026-10-15
Repo: `unified-risk-engine` (separate from `deep-feedback-control`).
Decisions already made with the user: real data is IEEE-CIS with a *derived* default label; Jupyter notebooks are required; leakage is controlled with **temporal neighbour sampling (Option B)**, with a transductive masked graph (Option A) as the fallback; the repo is private on GitHub.
Amended 2026-10-15 after the user asked about provenance and software versions: Project 2 uses its own, newer environment (section 9), and the spike scope and risks were extended (sections 11 and 13). Provenance of the design is in section 14.
Working assumption (the user did not answer): the pipeline is built and tested against a generated IEEE-CIS-shaped fixture while the user downloads the real data; the real-data run happens when the files land in `data/`.

## 1. Goal and success criteria

Train a heterogeneous graph attention network on IEEE-CIS with a shared backbone and two heads: **fraud** per transaction (short-term) and a **default proxy** per user (long-term), and measure honestly what the graph adds over a tabular baseline.

Success:
- The pipeline builds the graph, trains, evaluates and explains end to end on the fixture (tested), and on the real data once supplied.
- No label from a held-out user or a later time period reaches training or the model's inputs; this is enforced by tests, not only by design.
- Results are reported for 3 seeds against a gradient-boosting baseline on the same features, with the metric definitions and prevalence stated.
- GNNExplainer produces readable per-prediction explanations, with a stability check.
- The README reports what was measured, including negative results, and every number is checked against the output files.

Non-goals: real credit-default labels (IEEE-CIS has none), hyperparameter search, deployment or serving, other datasets, SHAP, automated Kaggle download or credential handling, a no-graph neural ablation.

## 2. Data

- Required: `train_transaction.csv` (`TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`, `ProductCD`, `card1`-`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `C*`, `D*`, `M*`, `V*`). Optional: `train_identity.csv` (`DeviceType`, `DeviceInfo`, `id_*`), joined on `TransactionID`. Files live in `data/` (git-ignored; never committed; licensed by Kaggle). `DATA_DIR` overrides the location.
- Missing required files produce an error naming the expected path. A column-schema check fails loudly if required columns are absent.
- Expected shape (from memory, to be verified when the file arrives): about 590k transactions over about 6 months, about 3.5% fraud.
- A **fixture generator** (`tests/fixtures.py`) writes IEEE-CIS-shaped frames (same column names and dtypes, planted fraud signal tied to amount, email domain and shared devices) so every test runs without the real data.

## 3. Graph schema

| Node type | Key | Features |
|---|---|---|
| `transaction` | `TransactionID` | engineered features (section 4) |
| `user` | `card1` + `addr1` (missing -> a sentinel) | none hand-built |
| `email_domain` | `P_emaildomain` | none hand-built |
| `device` | `DeviceInfo` (only when the identity file is present) | none hand-built |

Edges: `user -> transaction`, `transaction -> email_domain`, `transaction -> device`, plus reverse edges. Non-transaction nodes receive a single **learned vector per node type** as input, so all information reaches them by message passing. (Hand-built user aggregates over the whole period would leak future information.)

Every node has a `time`: a transaction's is `TransactionDT`; an entity node's is the time of its first transaction (first-seen time).

## 4. Leakage control (Option B) and splits

**Time periods** by `TransactionDT` quantiles over transactions: train = first 70%, validation = next 15%, test = last 15%. `T_train` is the last timestamp of the train period.

**Users** are split once, at random with a fixed seed, 70/15/15 into train/validation/test users.

**Temporal sampling.** Training and inference use `NeighborLoader` with `time_attr="time"`: for a seed at time `t`, only neighbours with `time <= t` are visible. No future transaction or entity can influence a prediction. Whether a neighbour with an *equal* timestamp is visible depends on the PyG version (the docs say strictly earlier in 2.2 and earlier-or-equal in later releases), so the spike verifies the behaviour of the installed version. Either way only features, never labels, are visible, and the README states which rule applies.

**Fraud head**
- Seeds are transactions, each with its own time.
- Train rows: transactions in the train period that belong to **train users**. Validation and test rows: all transactions in the validation and test periods respectively (any user).

**Default head**
- Seeds are users, all at time `T_train`, so a user's prediction sees only transactions up to `T_train`.
- Population: users with at least `min_transactions` (default 3) transactions up to `T_train`.
- **Label (a proxy, not credit data):** positive if the fraud rate over those transactions is at least `fraud_rate_threshold` (default 0.2). Train, validation and test users come from the user split. The thresholds are chosen by looking at prevalence only (never model performance), recorded, and each split must contain at least 30 positives, otherwise the pipeline raises and asks for different thresholds.

**Why train rows exclude held-out users:** a held-out user's default label is derived from their fraud labels, so those transactions must never contribute a training loss.

**Enforced by tests:** (1) no fraud-train transaction belongs to a validation or test user; (2) every fraud-train transaction has `time <= T_train`; (3) default labels are unchanged if all fraud labels after `T_train` are shuffled; (4) on the fixture, no sampled neighbour has `time` greater than its seed's; (5) features and encoders are fitted on the train period only.

**Fallback (Option A):** if the early spike shows temporal hetero sampling is unusable, fall back to one transductive graph with masks, tell the user, and add a limitation (future transactions' features are visible through neighbours).

## 5. Features (`pipeline/feature_engineering.py`)

- Transaction: log-amount; hour of day and day of week from `TransactionDT`; frequency encodings of categorical columns (`ProductCD`, `card*`, `addr*`, email domains, `M*`) fitted on the train period; NaN indicators; `C*` and `D*` as numeric; a subset of `V*` (missing rate at most 50% in the train period, then the `max_v_columns` (default 100) with highest train-period variance); numeric columns standardised with train-period statistics.
- Fit statistics come from train-period transactions only; the same transformer is applied to later periods.

## 6. Model and training

- `models/gnn_backbone.py`: 2-layer `HeteroConv` of GATv2 per relation (hidden 64, 4 heads, dropout 0.2, residual connections), with a SAGEConv option.
- `models/multi_task_heads.py`: transaction head (fraud logit) and user head (default logit), each a small MLP on the backbone embeddings.
- `models/losses.py`: Focal Loss (configurable `alpha`, default 0.75 weighting positives, and `gamma`, default 2).
- Mini-batch training with `NeighborLoader` (`num_neighbors` `[15, 10]`). A loader has a single seed type, so each step takes **one transaction batch (2048) and one user batch (1024)** from two loaders, with `loss = w_fraud * L_fraud + w_default * L_default` (defaults 1.0 and 0.5). AdamW (lr 1e-3, weight decay 1e-5), up to 20 epochs, early stopping (patience 3) on the mean of the two validation PR-AUCs, best checkpoint kept.
- Runs on the GPU (RTX 3070 Ti, 8 GB). Seeds `[0, 1, 2]`.

## 7. Evaluation and baseline

- Metrics per head: **PR-AUC** (primary), ROC-AUC, and **recall at 1% FPR** with the threshold chosen on the validation set and applied to the test set (both the recall and the realised test FPR are reported). Prevalence per split is always shown.
- **Baseline:** scikit-learn `HistGradientBoostingClassifier` on the same transaction features (fraud) and on user aggregates computed up to `T_train` (default), with the same splits, 3 seeds.
- Reporting: each seed's metrics, the mean, and a bootstrap 95% CI over test examples for each seed's PR-AUC. No cross-seed confidence claim is made from 3 seeds.

## 8. Explainability (`scripts/explain.py`)

PyG `Explainer` with `GNNExplainer` for selected transactions (for example the highest-scoring true and false positives) and users: top feature importances and influential edges on the sampled neighbourhood, written as a plot and a short readable report in `reports/`. A **stability check** re-runs the explainer with 3 seeds and reports the overlap of the top-10 features. IEEE-CIS features are anonymised (`V123`), so explanations name columns, not business meaning; the README says so. Explanations describe the model's behaviour, not causes.

## 9. Delivery

```
unified-risk-engine/
├── configs/config.yaml
├── pipeline/   data_loading.py  feature_engineering.py  splits.py  graph_builder.py
├── models/     gnn_backbone.py  multi_task_heads.py  losses.py
├── scripts/    train.py  baseline.py  evaluate.py  explain.py
├── notebooks/  01_data_and_graph.ipynb  02_train_and_evaluate.ipynb  03_explainability.ipynb
├── tests/      fixtures.py  test_*.py
├── data/ models/ reports/ plots/        (data and models git-ignored; final plots and CSVs committed)
├── docs/superpowers/{specs,plans}/
├── requirements.txt  README.md
```

- Notebooks drive the same functions as the scripts (no duplicated logic). Notebook 01: data checks, label prevalence and threshold choice, split and leakage checks, graph statistics. Notebook 02: training, evaluation and the baseline comparison. Notebook 03: explanations. `DFC_QUICK`-style switch (`URE_QUICK=1`) runs a tiny smoke version.
- Environment: **its own new virtual environment**, separate from the sibling project's (whose published results were produced with torch 2.11 and whose requirements pin it). Python 3.14. Target stack, checked for availability on 2026-10-15 with dry runs (nothing installed yet):
  - `torch==2.14.1` built for CUDA 13.0 (`https://download.pytorch.org/whl/cu130`; the driver reports CUDA 13.1).
  - `torch-geometric==2.8.1` (the latest release).
  - `pyg-lib==0.9.0+pt214cu130` from `https://data.pyg.org/whl/torch-2.14.0+cu130.html`. That wheel is built against torch 2.14.0; the spike confirms it loads and samples correctly under 2.14.1.
  - The latest releases at install time of pandas, scikit-learn, matplotlib, pyyaml, pytest and the jupyter packages, then pinned exactly in `requirements.txt`.
  - **Fallback if the spike finds a problem with the newer stack:** `torch==2.11.0` (CUDA 12.8) with `pyg-lib==0.9.0+pt211cu128` from `https://data.pyg.org/whl/torch-2.11.0+cu128.html`. No `pyg-lib` wheels exist for the CUDA 12.8 builds of torch 2.12 to 2.14.
  - `pyg-lib` provides the neighbour sampling that `NeighborLoader` needs.

## 10. Tests

On the fixture: schema and column checks; feature transformer fitted on train only; graph node and edge counts; the five leakage tests in section 4; focal loss reduces to cross-entropy at gamma 0 and weights positives by alpha; forward shapes for both heads; a short training smoke run on CPU and GPU; baseline runs; metrics (PR-AUC, recall at fixed FPR, bootstrap) on synthetic scores; explainer smoke test and stability metric; error messages for missing data files and too-few-positives.

## 11. Risks and fallbacks

| Risk | Plan |
|---|---|
| Temporal sampling is unsupported or buggy for hetero graphs in PyG 2.8.1 / pyg-lib 0.9 | The **first plan task is a spike** on the fixture that checks it and the leakage property; otherwise fall back to Option A and tell the user |
| The PyG docs do not say whether `time` can be set per node type on a heterogeneous graph | The spike tests it; if needed, give every node type a `time` (entity nodes use first-seen time, as designed) |
| Temporal sampling switches on `disjoint` mode automatically (one subgraph per seed), which can multiply memory and run time | The spike measures memory and speed on a realistically sized fixture before the batch sizes and neighbour counts are fixed; reduce them if needed |
| Equal-timestamp visibility changed between PyG releases | The spike checks the installed version's rule; the README states it |
| `pyg-lib` built for torch 2.14.0 may not load under torch 2.14.1, or the newest torch may break another dependency | The spike imports and exercises it first; fall back to the pinned torch 2.11 stack (section 9) |
| `GNNExplainer` does not support the hetero model as written | Spike early; fall back to explaining a homogeneous projection of the neighbourhood, or feature-mask-only, and say so |
| 8 GB GPU memory or run time too high | Smaller batch, fewer neighbours, smaller hidden size; measure in the spike before committing to the schedule |
| The default proxy is too rare or too easy | Choose thresholds by prevalence only; at least 30 positives per split; state the coupling with fraud plainly |
| The real data differs from the fixture (columns, sizes) | Schema check with clear errors; real-data verification is a separate, reported step |

## 12. Honest limitations (go in the README)

- The default label is a proxy derived from fraud history, so the two tasks are coupled by construction. This is joint risk modelling on a shared graph, not validated credit scoring.
- `card1` + `addr1` is an approximation of a customer identity.
- IEEE-CIS features are anonymised; explanations name columns, not meaning.
- One dataset, 3 seeds; no hyperparameter search; results may not transfer.
- Anything in the README explaining *why* a result occurred is a hypothesis unless an experiment tested it.
- Not measured until the real data is supplied: everything on the real dataset. Only fixture-based tests are verified before that.

## 13. Verification plan

- Spike first, in the new environment: install the target stack; check `pyg-lib` loads under torch 2.14.1; temporal sampling on a heterogeneous graph (per-node-type `time`, the equal-timestamp rule, no neighbour later than its seed, the effect of `disjoint` mode on memory and speed); GNNExplainer on a hetero model. Then `pytest` green on the fixture.
- Real-data run when the files are supplied: record dataset size and prevalence, train 3 seeds and the baseline, and compare.
- README numbers are checked programmatically against the output CSVs; a fresh clone is tested; a fresh reviewer reviews the whole branch; nothing is pushed without asking.

## 14. Provenance and references

Option B is **not copied from a single paper**; it combines established pieces. Status of each reference:

- **Verified this session (read via search):** the PyG `NeighborLoader` temporal-sampling parameters (`time_attr`, `input_time`, `temporal_strategy` in `"uniform"` or `"last"`, automatic `disjoint` mode) and their version-dependent semantics, from the PyG documentation; and *Relational Deep Learning: Graph Representation Learning on Relational Databases* (Fey, Hu, Huang, Leskovec and others, arXiv 2312.04615), which describes time-consistent sampling on temporal heterogeneous graphs and is the closest published precedent. That PyG's feature descends directly from that paper is **not confirmed**.
- **Cited from memory, not looked up this session:** GAT (Velickovic et al., ICLR 2018), GATv2 (Brody, Alon and Yahav, ICLR 2022), Focal Loss (Lin et al., ICCV 2017), GNNExplainer (Ying et al., NeurIPS 2019), the PyG library (Fey and Lenssen, 2019), and the IEEE-CIS Fraud Detection Kaggle competition (data provided by Vesta). Check them before citing in anything formal.
- **This project's own design choices, with no published basis claimed:** the user-level split rule that keeps held-out users' transactions out of fraud training, the fraud-history default proxy, and the alternating-batch multi-task loop.
