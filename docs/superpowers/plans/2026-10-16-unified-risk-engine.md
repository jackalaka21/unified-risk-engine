# Unified Risk Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe temporal heterogeneous-GNN pipeline for transaction fraud and account-level risk on IEEE-CIS, evaluate it fairly against history-aware tree models with ablations, a drift study and finance-shaped metrics, and then add a scoring service with a latency benchmark.

**Architecture:** `pipeline/` turns raw CSVs into a time-stamped heterogeneous graph with leakage-safe splits, an embargo and fitted-on-train features; `models/` holds the GATv2 backbone, two heads and focal loss; `scripts/` trains variants with alternating transaction/user mini-batches via PyG temporal neighbour sampling, runs baselines, evaluation, a drift study and explanations; `serving/` (phase 2) reproduces the sampled neighbourhood online from a time-sorted index. A generated IEEE-CIS-shaped fixture lets everything be tested before the real data arrives.

**Tech Stack:** Python 3.14 in a new venv, PyTorch 2.14.1 (CUDA 13.0), PyG 2.8.1 + pyg-lib, pandas, scikit-learn, matplotlib, PyYAML, pytest, Jupyter, FastAPI (phase 2), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-15-unified-risk-engine-design.md` (executors read it, especially sections 4, 7, 9b and 11).

## Global Constraints

- Repo: `unified-risk-engine`, branch `main`; work on a feature branch. Never commit data, models, or credentials. Do NOT push without asking the user. Never touch Kaggle credentials.
- `<venv>` = a NEW virtual environment for this project (Python 3.14), not the sibling project's. In bash use `PY="<venv>/Scripts/python.exe"`, set again in every call.
- Target stack (spec section 9): `torch==2.14.1` from `https://download.pytorch.org/whl/cu130`; `torch-geometric==2.8.1`; `pyg-lib==0.9.0+pt214cu130` from `https://data.pyg.org/whl/torch-2.14.0+cu130.html`. Fallback only if the spike fails: `torch==2.11.0` (CUDA 12.8) with `pyg-lib==0.9.0+pt211cu128` from `https://data.pyg.org/whl/torch-2.11.0+cu128.html`.
- Splits: train = first 70% of transactions by `TransactionDT`; validation ends at the 85% quantile; embargo `E` = 7 days between train and validation and between validation and test; users split 70/15/15 with a fixed seed.
- Account label (a proxy): `min_transactions` = 3, `fraud_rate_threshold` = 0.2, at least 30 positives per split or raise; thresholds chosen by prevalence only.
- Features: `max_v_columns` = 100, `V*` kept only if missing rate at most 50% in the train period; all fit statistics from the train period only.
- Model: 2 layers, GATv2, hidden 64, 4 heads, dropout 0.2, residual connections; Focal Loss `alpha` 0.75, `gamma` 2; `num_neighbors` `[15, 10]`; one transaction batch of 2048 and one user batch of 1024 per step; loss weights 1.0 (fraud) and 0.5 (account); AdamW lr 1e-3, weight decay 1e-5; up to 20 epochs, patience 3; seeds `[0, 1, 2]`.
- Evaluation: PR-AUC primary; alert budgets 0.5%, 1%, 5%; `review_cost` 5.0 (an explicit assumption); recall at 1% FPR with the threshold chosen on validation; bootstrap 95% CIs over test examples; no cross-seed CI claim from 3 seeds.
- Scores passed to metrics are probabilities in [0, 1].
- CI never asserts on timing values. GPU tests are marked `gpu`, long tests `slow`; CI runs `-m "not gpu and not slow"`.
- Any "why" in the README or model card is a hypothesis unless an experiment tested it. README and model-card numbers are generated or checked programmatically against output CSVs.
- End every commit message with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (an extra `-m` argument). Use the repo-local git identity already configured.
- Fixture-based results are NOT real results: never report fixture numbers as findings.

## Review Focus

Failure modes the spec implies that a first implementation would likely mishandle, most likely first. Each has a test in the task that owns the code.

1. **Missing entity keys.** Rows with no email, no device, or no `addr1` must produce no edge, NaN history features, and no crash. (Task 6: `test_missing_entities_give_nan_not_crashes`; Task 7: `test_rows_without_entities_have_no_edges`)
2. **Users with no usable history.** A user whose transactions all fall after `T_train`, or who has fewer than `min_transactions` before it, must be excluded from the account population and never labelled. (Task 4: `test_users_without_train_period_history_are_ineligible`)
3. **Degenerate metric inputs.** A week or split with no positives, constant scores, or tied scores must give NaN or a documented value, never a crash or a silently wrong number. (Task 10: `test_degenerate_inputs`, `test_weekly_table_marks_weeks_without_positives`)
4. **CPU-only machines.** `device: auto` must fall back to CPU, and `device: cuda` without CUDA must fail with a message naming the fix; CI has no GPU. (Task 3: `test_device_auto_falls_back_to_cpu`, `test_device_cuda_unavailable_message`)
5. **Results mixed across configurations.** Output tables record a config hash; the model-card generator must refuse a table that mixes hashes. (Task 12: `test_results_carry_one_config_hash`; Task 16: `test_model_card_refuses_mixed_config_hashes`)

---

## File Structure

```
unified-risk-engine/
├── .github/workflows/tests.yml            (Task 16; container job in Task 21)
├── Dockerfile  .dockerignore              (Task 20)
├── configs/config.yaml                    (Task 3)
├── conftest.py  pytest.ini                (Task 1)
├── requirements.txt  requirements-ci.txt  requirements-serving.txt   (Tasks 1, 16, 20)
├── pipeline/
│   ├── config.py  data_loading.py         (Task 3)
│   ├── splits.py                          (Task 4)
│   ├── feature_engineering.py             (Task 5)
│   ├── history_features.py                (Task 6)
│   ├── graph_builder.py  prepare.py       (Task 7)
│   ├── metrics.py                         (Task 10)
│   └── notebook_support.py                (Tasks 15, 21)
├── models/  gnn_backbone.py  multi_task_heads.py  losses.py   (Task 8; also holds git-ignored checkpoints)
├── scripts/
│   ├── spike_pyg.py                       (Task 2)
│   ├── train.py                           (Task 9)
│   ├── baseline.py                        (Task 11)
│   ├── evaluate.py                        (Task 12)
│   ├── drift_study.py                     (Task 13)
│   ├── explain.py                         (Task 14, 16)
│   ├── build_notebooks.py                 (Tasks 15, 21)
│   ├── make_model_card.py  make_ci_requirements.py   (Tasks 16, 20, 22)
│   ├── run_all.py                         (Task 17)
│   ├── check_data.py                      (Task 18)
│   ├── spike_serving.py                   (Task 19)
│   ├── export_artifacts.py  smoke_service.py   (Task 20)
│   ├── benchmark_latency.py               (Task 21)
│   └── sampling_rule.py                   (Task 22)
├── serving/  graph_index.py  scorer.py    (Task 19);  bundle.py  app.py   (Task 20)
├── notebooks/  01..05                     (Tasks 15, 21)
├── tests/  fixtures.py  conftest.py  serving_helpers.py  test_*.py
├── docs/  spike-report.md  serving-spike.md  MODEL_CARD.md  superpowers/{specs,plans}/
└── data/ models/*.pt outputs/ plots/ reports/ artifacts/   (data, checkpoints, outputs, artifacts git-ignored)
```

---

## Phase 0: environment and spike

### Task 1: New environment, pins and test scaffold

**Files:**
- Create: `conftest.py`, `pytest.ini`, `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: a working interpreter `<venv>/Scripts/python.exe` with CUDA PyTorch, PyG and pyg-lib importable; pytest markers `gpu` and `slow`.

- [ ] **Step 1: Create the venv and install the target stack**

```bash
py -3.14 -m venv "<venv>"
PY="<venv>/Scripts/python.exe"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install torch-geometric==2.8.1
"$PY" -m pip install "pyg-lib==0.9.0+pt214cu130" -f https://data.pyg.org/whl/torch-2.14.0+cu130.html
"$PY" -m pip install pandas scikit-learn matplotlib pyyaml pytest ipykernel nbformat nbconvert jupyterlab
```
Expected: all install. If the `torch` or `pyg-lib` install fails, record the error in the ledger and apply the fallback stack from Global Constraints (spec section 9).

- [ ] **Step 2: Verify the stack loads and sees the GPU**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" -c "import torch, torch_geometric, pyg_lib; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', '| pyg', torch_geometric.__version__, '| pyg_lib', pyg_lib.__version__)"
```
Expected: `torch 2.14.1+cu130 | cuda True NVIDIA GeForce RTX 3070 Ti Laptop GPU | pyg 2.8.1 | pyg_lib 0.9.0+pt214cu130`. If `pyg_lib` fails to import under torch 2.14.1 (it was built for 2.14.0), that is the first spike finding: apply the fallback stack and ledger it.

- [ ] **Step 3: Register the kernel and write the pinned `requirements.txt`**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" -m ipykernel install --user --name unified_risk_engine --display-name "Python (unified_risk_engine)"
"$PY" - <<'EOF'
import subprocess, sys
names = ["torch", "torch-geometric", "pyg-lib", "pandas", "scikit-learn", "matplotlib", "pyyaml", "pytest",
         "ipykernel", "nbformat", "nbconvert", "jupyterlab", "numpy", "scipy"]
frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
want = {n.lower().replace("-", "_") for n in names}
lines = [l for l in frozen if l.split("==")[0].lower().replace("-", "_") in want]
header = ["# torch and pyg-lib come from special indexes (see README):",
          "#   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130",
          "#   pip install pyg-lib==0.9.0+pt214cu130 -f https://data.pyg.org/whl/torch-2.14.0+cu130.html"]
open("requirements.txt", "w", encoding="utf-8").write("\n".join(header + sorted(lines)) + "\n")
print(open("requirements.txt").read())
EOF
```
Expected: a `requirements.txt` with exact pins for every listed package.

- [ ] **Step 4: Write `conftest.py`, `pytest.ini`, and extend `.gitignore`**

`conftest.py`:
```python
import matplotlib

matplotlib.use("Agg")
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
markers =
    gpu: needs a CUDA GPU (skipped in CI)
    slow: long-running (skipped in CI)
filterwarnings =
    ignore::DeprecationWarning
```

Append to `.gitignore`:
```gitignore

# generated outputs
outputs/
artifacts/
reports/
```

- [ ] **Step 5: Commit**

```bash
git checkout -b feature/risk-engine-core
git add conftest.py pytest.ini requirements.txt .gitignore
git commit -m "chore: new environment, pinned requirements and pytest scaffold" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Spike: verify the PyG capabilities the design depends on

**Files:**
- Create: `scripts/spike_pyg.py`, `docs/spike-report.md`

**Interfaces:**
- Produces: `docs/spike-report.md` stating, with evidence, which of these hold: (1) temporal hetero sampling with `time` on every node type, no neighbour later than its seed; (2) the equal-timestamp rule; (3) whether entity nodes without `time` break sampling; (4) `"last"` strategy determinism; (5) memory and speed per batch size; (6) hetero `Explainer`/`GNNExplainer` and fidelity metrics; (7) CPU-only Linux wheels for CI. It sets the decisions the later tasks read: `train.temporal` (true or false), batch sizes, `explain.mode` (`hetero` or `occlusion`).

- [ ] **Step 1: Write the spike script** (`scripts/spike_pyg.py`)

```python
"""Spike: verify the PyG features the design depends on. Prints PASS/FAIL/INFO lines and writes nothing."""
from __future__ import annotations

import time
import traceback

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv, HeteroConv

I64MAX = np.iinfo(np.int64).max


def synthetic_graph(n_tx=5000, n_users=300, n_emails=40, entity_time=True, seed=0):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.choice(n_tx * 50, size=n_tx, replace=False)).astype(np.int64)
    user = rng.integers(0, n_users, n_tx)
    email = rng.integers(0, n_emails, n_tx)
    d = HeteroData()
    d["transaction"].x = torch.randn(n_tx, 8)
    d["transaction"].y = torch.as_tensor((rng.random(n_tx) < 0.05).astype(np.int64))
    d["transaction"].time = torch.as_tensor(times)
    for name, codes, n in (("user", user, n_users), ("email_domain", email, n_emails)):
        first = np.full(n, I64MAX, dtype=np.int64)
        np.minimum.at(first, codes, times)
        d[name].num_nodes = n
        if entity_time:
            d[name].time = torch.as_tensor(first)
    tx = torch.arange(n_tx)
    u, e = torch.as_tensor(user), torch.as_tensor(email)
    d["user", "owns", "transaction"].edge_index = torch.stack([u, tx])
    d["transaction", "rev_owns", "user"].edge_index = torch.stack([tx, u])
    d["transaction", "uses_email", "email_domain"].edge_index = torch.stack([tx, e])
    d["email_domain", "rev_uses_email", "transaction"].edge_index = torch.stack([e, tx])
    return d


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def check_temporal(data, strategy="uniform"):
    idx = torch.arange(data["transaction"].num_nodes)[-512:]
    loader = NeighborLoader(
        data, num_neighbors=[15, 10], input_nodes=("transaction", idx),
        input_time=data["transaction"].time[idx], time_attr="time",
        temporal_strategy=strategy, batch_size=128,
    )
    batch = next(iter(loader))
    bs = batch["transaction"].batch_size
    seed_time = batch["transaction"].time[:bs]
    has_vec = all(hasattr(batch[nt], "batch") for nt in batch.node_types)
    violations = 0
    if has_vec:
        for nt in batch.node_types:
            if hasattr(batch[nt], "time"):
                violations += int((batch[nt].time > seed_time[batch[nt].batch]).sum())
    counts = {nt: batch[nt].num_nodes for nt in batch.node_types}
    edges = {str(et): int(batch[et].edge_index.shape[1]) for et in batch.edge_types}
    return {"batch_size": bs, "has_batch_vector": has_vec, "violations": violations,
            "node_counts": counts, "edge_counts": edges, "n_id_tx": batch["transaction"].n_id.clone()}


def check_equal_timestamp():
    d = HeteroData()
    d["transaction"].x = torch.zeros(3, 2)
    d["transaction"].time = torch.tensor([5, 5, 9])
    d["user"].num_nodes = 1
    d["user"].time = torch.tensor([5])
    tx = torch.arange(3)
    u = torch.zeros(3, dtype=torch.long)
    d["user", "owns", "transaction"].edge_index = torch.stack([u, tx])
    d["transaction", "rev_owns", "user"].edge_index = torch.stack([tx, u])
    loader = NeighborLoader(d, num_neighbors=[5, 5], input_nodes=("transaction", torch.tensor([0])),
                            input_time=torch.tensor([5]), time_attr="time", batch_size=1)
    batch = next(iter(loader))
    ids = batch["transaction"].n_id.tolist()
    return "earlier_or_equal" if 1 in ids else "strictly_earlier", ids


class TinyHetero(torch.nn.Module):
    def __init__(self, edge_types, hidden=64, heads=4):
        super().__init__()
        self.tx = torch.nn.Linear(8, hidden)
        self.vec = torch.nn.ParameterDict({"user": torch.nn.Parameter(torch.randn(hidden)),
                                           "email_domain": torch.nn.Parameter(torch.randn(hidden))})
        self.convs = torch.nn.ModuleList([
            HeteroConv({et: GATv2Conv((-1, -1), hidden, heads=heads, concat=False, add_self_loops=False)
                        for et in edge_types}, aggr="sum") for _ in range(2)])
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict, num_nodes):
        h = {"transaction": torch.relu(self.tx(x_dict["transaction"]))}
        for nt, p in self.vec.items():
            h[nt] = p.unsqueeze(0).repeat(num_nodes.get(nt, 0), 1)
        for conv in self.convs:
            o = conv(h, edge_index_dict)
            h = {nt: torch.relu(o[nt]) + h[nt] if nt in o else h[nt] for nt in h}
        return h


def check_speed(device, batch_sizes=(256, 1024, 2048), n_tx=200_000):
    data = synthetic_graph(n_tx=n_tx, n_users=15_000, n_emails=120)
    model = TinyHetero(list(data.edge_types)).to(device)
    # lazy init
    warm = next(iter(NeighborLoader(data, num_neighbors=[15, 10], input_nodes=("transaction", torch.arange(64)),
                                    input_time=data["transaction"].time[:64], time_attr="time", batch_size=64))).to(device)
    model(warm.x_dict, warm.edge_index_dict, {nt: warm[nt].num_nodes for nt in warm.node_types})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    seeds = torch.arange(n_tx)[-100_000:]
    rows = []
    for bs in batch_sizes:
        loader = NeighborLoader(data, num_neighbors=[15, 10], input_nodes=("transaction", seeds),
                                input_time=data["transaction"].time[seeds], time_attr="time",
                                batch_size=bs, shuffle=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0, n = time.perf_counter(), 0
        for batch in loader:
            batch = batch.to(device)
            h = model(batch.x_dict, batch.edge_index_dict, {nt: batch[nt].num_nodes for nt in batch.node_types})
            logits = model.out(h["transaction"][: batch["transaction"].batch_size]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["transaction"].y[: batch["transaction"].batch_size].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            n += 1
            if n == 6:
                break
        if device == "cuda":
            torch.cuda.synchronize()
        per_batch = (time.perf_counter() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
        train_seeds = 413_000  # about 70% of the real data's 590k transactions
        rows.append((bs, per_batch, peak, per_batch * train_seeds / bs / 60))
    return rows


def check_explainer():
    from torch_geometric.explain import Explainer, GNNExplainer

    data = synthetic_graph(n_tx=300, n_users=30, n_emails=6)
    net = TinyHetero(list(data.edge_types))
    nn_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    x_dict = {"transaction": data["transaction"].x}
    net(x_dict, data.edge_index_dict, nn_dict)  # lazy init

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x_dict, edge_index_dict):
            h = self.net(x_dict, edge_index_dict, nn_dict)
            return self.net.out(h["transaction"]).squeeze(-1)

    explainer = Explainer(
        model=Wrapper(), algorithm=GNNExplainer(epochs=20), explanation_type="model",
        node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )
    expl = explainer(x_dict, data.edge_index_dict, index=torch.tensor([0, 1]))
    info = {"type": type(expl).__name__,
            "tx_node_mask": tuple(expl["transaction"].node_mask.shape),
            "edge_masks": {str(et): tuple(expl[et].edge_mask.shape) for et in expl.edge_types if hasattr(expl[et], "edge_mask")}}
    try:
        from torch_geometric.explain.metric import fidelity

        info["fidelity"] = tuple(float(v) for v in fidelity(explainer, expl))
    except Exception as exc:  # recorded, not fatal
        info["fidelity_error"] = f"{type(exc).__name__}: {exc}"
    return info


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import pyg_lib

    report("env", True, f"torch {torch.__version__}, pyg_lib {pyg_lib.__version__}, device {device}")
    data = synthetic_graph()
    try:
        r = check_temporal(data)
        report("temporal sampling, time on every node type",
               r["violations"] == 0 and r["has_batch_vector"],
               f"violations={r['violations']} batch_vector={r['has_batch_vector']} nodes={r['node_counts']} edges={r['edge_counts']}")
    except Exception:
        report("temporal sampling, time on every node type", False, traceback.format_exc(limit=2))
    try:
        rule, ids = check_equal_timestamp()
        report("equal-timestamp rule", True, f"{rule} (sampled transaction ids {ids})")
    except Exception:
        report("equal-timestamp rule", False, traceback.format_exc(limit=2))
    try:
        check_temporal(synthetic_graph(entity_time=False))
        report("entity nodes without time", True, "sampling worked without entity time")
    except Exception as exc:
        report("entity nodes without time", False, f"{type(exc).__name__}: {exc}")
    try:
        a = check_temporal(data, "last")["n_id_tx"]
        b = check_temporal(data, "last")["n_id_tx"]
        report("'last' strategy deterministic", bool(torch.equal(a, b)))
    except Exception:
        report("'last' strategy deterministic", False, traceback.format_exc(limit=2))
    try:
        for bs, sec, peak, epoch_min in check_speed(device):
            print(f"INFO speed batch_size={bs}: {sec:.2f}s/batch, peak {peak:.2f} GiB, estimated epoch {epoch_min:.1f} min", flush=True)
    except Exception:
        report("speed and memory", False, traceback.format_exc(limit=3))
    try:
        report("hetero GNNExplainer", True, str(check_explainer()))
    except Exception:
        report("hetero GNNExplainer", False, traceback.format_exc(limit=3))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the spike**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" -W ignore scripts/spike_pyg.py 2>&1 | grep -v -i "warn"
```
Expected: PASS/FAIL/INFO lines for every check. Record the full output.

- [ ] **Step 3: Check the CPU-only Linux wheels that CI needs**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" -m pip download --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 --python-version 3.14 \
  -d /tmp/ci_wheels "pyg-lib==0.9.0+pt214cpu" -f https://data.pyg.org/whl/torch-2.14.0+cpu.html 2>&1 | tail -3
"$PY" -m pip download --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 --python-version 3.14 \
  -d /tmp/ci_wheels torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
```
Expected: both download. If the Linux wheel is missing for Python 3.14, record it and choose a CI Python version for which wheels exist (this changes CI only).

- [ ] **Step 4: Apply the decision rules and write `docs/spike-report.md`**

Write the report with one section per check (result, evidence line copied from the output, decision). Decision rules:
- Temporal sampling: if `violations == 0` and the batch vector exists, **Option B stands** (`train.temporal: true`). If it raises or violates, set `train.temporal: false` (Option A) in Task 3's config, tell the user, amend the spec's section 4 and drop the sampler-based leakage test in Task 9 (the masked-eval tests remain).
- Equal-timestamp rule: record `earlier_or_equal` or `strictly_earlier`; the README states it.
- Entity time: the design gives every entity node a first-seen `time` regardless of this result.
- Speed: choose `train.batch_fraud` (default 2048) as the largest of 256, 1024, 2048 whose peak memory is at most 5 GiB and whose estimated epoch is at most 5 minutes; if even 256 exceeds 10 minutes, use `num_neighbors` `[10, 5]` and say so. Set `train.batch_account` to half of `train.batch_fraud`.
- Explainer: if the hetero `GNNExplainer` call works and exposes `node_mask`/`edge_mask`, set `explain.mode: hetero`; otherwise `explain.mode: occlusion` (Task 14 implements both). Record whether `fidelity` worked on hetero explanations.
- Environment: if `pyg_lib` failed under torch 2.14.1, record it and switch to the fallback stack (re-run Task 1 steps 1-3 with the fallback versions).

- [ ] **Step 5: Commit**

```bash
git add scripts/spike_pyg.py docs/spike-report.md
git commit -m "docs: spike PyG temporal hetero sampling, explainer and speed" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Phase 1: core pipeline

### Task 3: Config, data loading and the fixture generator

**Files:**
- Create: `configs/config.yaml`, `pipeline/__init__.py` (empty), `pipeline/config.py`, `pipeline/data_loading.py`, `tests/fixtures.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_data_loading.py`

**Interfaces:**
- Produces in `pipeline.config`: `PROJECT_ROOT`, `load_config(path=None, overrides=None) -> dict`, `resolve_path(p) -> Path`, `resolve_device(requested: str) -> str`, `config_hash(cfg) -> str`, `VARIANTS`.
- Produces in `pipeline.data_loading`: `REQUIRED_TX`, `load_raw(cfg) -> DataFrame`, `finalize(tx) -> DataFrame` (sorted by `TransactionDT`, `TransactionID`; adds columns `user`, `email`, `device`; guarantees `DeviceInfo`).
- Produces in `tests/fixtures.py`: `make_fixture(n_users=500, n_tx=9000, days=90, seed=0, with_identity=True) -> (tx_df, identity_df_or_None)`, `write_fixture(dir, **kw)`, `fixture_overrides() -> dict` (config overrides sized for the fixture: `splits.embargo_days` 1, `account.min_positives` 3, small batches and neighbours, `train.device` `cpu`).
- Produces in `tests/conftest.py`: fixtures `raw_frames` (session scope), `cfg` (the default config with `fixture_overrides()` applied).

- [ ] **Step 1: Write the config file** (`configs/config.yaml`)

```yaml
data:
  dir: data
  transaction_file: train_transaction.csv
  identity_file: train_identity.csv   # optional
seed: 0

splits:
  train_frac: 0.70
  val_end_frac: 0.85
  embargo_days: 7
  user_fractions: [0.70, 0.15, 0.15]

account:
  min_transactions: 3
  fraud_rate_threshold: 0.2
  min_positives: 30

features:
  max_v_columns: 100
  v_max_missing: 0.5

model:
  hidden: 64
  heads: 4
  layers: 2
  dropout: 0.2
  conv: gatv2            # gatv2 or sage

train:
  variant: multitask     # multitask, fraud_only, account_only or no_graph
  temporal: true         # false = Option A fallback (set by the spike if needed)
  temporal_strategy: uniform
  num_neighbors: [15, 10]
  batch_fraud: 2048
  batch_account: 1024
  lr: 1.0e-3
  weight_decay: 1.0e-5
  epochs: 20
  patience: 3
  focal_alpha: 0.75
  focal_gamma: 2.0
  w_fraud: 1.0
  w_account: 0.5
  seeds: [0, 1, 2]
  device: auto           # auto, cpu or cuda
  max_steps_per_epoch: null

evaluation:
  alert_budgets: [0.005, 0.01, 0.05]
  review_cost: 5.0
  fpr_target: 0.01
  bootstrap_samples: 1000

drift:
  retrain_every_days: 28
  seed: 0

explain:
  mode: hetero           # hetero or occlusion (set by the spike)
  n_explained: 10
  fidelity_sample: 200
  stability_seeds: 3
  epochs: 100
  top_k: 10
```

- [ ] **Step 2: Write the failing tests** (`tests/test_config.py`)

```python
import pytest
import torch

from pipeline.config import config_hash, load_config, resolve_device


def test_default_config_loads_with_spec_values():
    cfg = load_config()
    assert cfg["splits"]["train_frac"] == 0.70 and cfg["splits"]["val_end_frac"] == 0.85
    assert cfg["splits"]["embargo_days"] == 7
    assert cfg["account"]["min_transactions"] == 3 and cfg["account"]["fraud_rate_threshold"] == 0.2
    assert cfg["model"]["hidden"] == 64 and cfg["model"]["heads"] == 4 and cfg["model"]["layers"] == 2
    assert cfg["train"]["num_neighbors"] == [15, 10] and cfg["train"]["focal_alpha"] == 0.75
    assert cfg["train"]["seeds"] == [0, 1, 2]


def test_overrides_merge_deeply():
    cfg = load_config(overrides={"train": {"epochs": 1}, "seed": 5})
    assert cfg["train"]["epochs"] == 1 and cfg["train"]["patience"] == 3 and cfg["seed"] == 5


def test_data_dir_environment_variable_overrides_the_config_but_not_explicit_overrides(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "somewhere")
    assert load_config()["data"]["dir"] == "somewhere"
    assert load_config(overrides={"data": {"dir": "explicit"}})["data"]["dir"] == "explicit"


@pytest.mark.parametrize("overrides, message", [
    ({"splits": {"train_frac": 0.9, "val_end_frac": 0.8}}, "train_frac"),
    ({"splits": {"user_fractions": [0.5, 0.2, 0.1]}}, "user_fractions"),
    ({"train": {"variant": "bogus"}}, "variant"),
    ({"model": {"conv": "gcn"}}, "conv"),
    ({"model": {"layers": 3}}, "layers"),
    ({"splits": {"embargo_days": -1}}, "embargo_days"),
])
def test_invalid_configs_are_rejected_with_a_named_key(overrides, message):
    with pytest.raises(ValueError, match=message):
        load_config(overrides=overrides)


def test_config_hash_is_stable_and_sensitive():
    a, b = load_config(), load_config()
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(overrides={"seed": 1}))


def test_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cpu") == "cpu"


def test_device_cuda_unavailable_message(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device: auto"):
        resolve_device("cuda")
```

- [ ] **Step 3: Run to verify failure**

Run: `"$PY" -m pytest tests/test_config.py -q`
Expected: `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 4: Implement** (`pipeline/config.py`)

```python
"""Configuration loading, validation, hashing and device resolution."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
VARIANTS = ("multitask", "fraud_only", "account_only", "no_graph")
CONVS = ("gatv2", "sage")


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _merge(base: dict, over: dict) -> dict:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def validate(cfg: dict) -> None:
    s = cfg["splits"]
    if not 0 < s["train_frac"] < s["val_end_frac"] < 1:
        raise ValueError("splits.train_frac must be below splits.val_end_frac, both within (0, 1)")
    if s["embargo_days"] < 0:
        raise ValueError("splits.embargo_days must be >= 0")
    if len(s["user_fractions"]) != 3 or abs(sum(s["user_fractions"]) - 1.0) > 1e-6:
        raise ValueError("splits.user_fractions must be three numbers that sum to 1")
    if cfg["train"]["variant"] not in VARIANTS:
        raise ValueError(f"train.variant must be one of {VARIANTS}")
    if cfg["model"]["conv"] not in CONVS:
        raise ValueError(f"model.conv must be one of {CONVS}")
    nn = cfg["train"]["num_neighbors"]
    if not nn or not all(isinstance(n, int) and n > 0 for n in nn):
        raise ValueError("train.num_neighbors must be a list of positive integers")
    if cfg["model"]["layers"] != len(nn):
        raise ValueError("model.layers must equal len(train.num_neighbors)")
    if not 0 < cfg["train"]["focal_alpha"] < 1:
        raise ValueError("train.focal_alpha must be within (0, 1)")


def load_config(path=None, overrides: dict | None = None) -> dict:
    path = resolve_path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if os.environ.get("DATA_DIR"):
        cfg["data"]["dir"] = os.environ["DATA_DIR"]          # the spec's override; explicit `overrides` still win
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    validate(cfg)
    return cfg


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requests device '{requested}' but CUDA is not available. "
            "Set `train.device: auto` (falls back to CPU) or `cpu`."
        )
    return requested


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
```

- [ ] **Step 5: Run to verify pass**

Run: `"$PY" -m pytest tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 6: Write the fixture generator** (`tests/fixtures.py`)

```python
"""IEEE-CIS-shaped synthetic data with a planted fraud signal, for tests only (never real results)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMAILS = np.array(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "protonmail.com", None], dtype=object)
P_LEGIT = np.array([0.45, 0.20, 0.15, 0.10, 0.02, 0.08])
P_FRAUD = np.array([0.15, 0.10, 0.10, 0.25, 0.30, 0.10])


def make_fixture(n_users=500, n_tx=9000, days=90, seed=0, with_identity=True):
    rng = np.random.default_rng(seed)
    horizon = days * 86400
    times = np.sort(rng.choice(horizon, size=n_tx, replace=False)).astype(np.int64)  # unique timestamps
    user = rng.integers(0, n_users, size=n_tx)
    compromised = rng.random(n_users) < 0.12
    start = rng.uniform(0.15, 0.7, size=n_users) * horizon
    in_window = compromised[user] & (times >= start[user])
    y = (rng.random(n_tx) < np.where(in_window, 0.45, 0.01)).astype(int)

    amount = np.exp(rng.normal(3.5, 0.8, size=n_tx) + 0.9 * y)
    email_idx = np.where(y == 1, rng.choice(6, size=n_tx, p=P_FRAUD), rng.choice(6, size=n_tx, p=P_LEGIT))
    email = EMAILS[email_idx]
    tx = pd.DataFrame({
        "TransactionID": 2_987_000 + np.arange(n_tx),
        "isFraud": y,
        "TransactionDT": times,
        "TransactionAmt": amount.round(2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n_tx, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        "card1": 1000 + user,
        "card2": rng.choice([111.0, 222.0, 333.0, np.nan], size=n_tx),
        "card3": 150.0,
        "card4": rng.choice(["visa", "mastercard", "discover"], size=n_tx),
        "card5": rng.choice([226.0, 224.0, 166.0], size=n_tx),
        "card6": rng.choice(["debit", "credit"], size=n_tx),
        "addr1": np.where(rng.random(n_tx) < 0.08, np.nan, 100 + user % 50),
        "addr2": 87.0,
        "dist1": np.where(rng.random(n_tx) < 0.7, np.nan, rng.exponential(50, size=n_tx)),
        "P_emaildomain": email,
        "R_emaildomain": np.where(rng.random(n_tx) < 0.7, None, "gmail.com"),
    })
    for j in range(1, 15):
        tx[f"C{j}"] = rng.poisson(1 + 3 * y).astype(float)
    for j in range(1, 16):
        tx[f"D{j}"] = np.where(rng.random(n_tx) < 0.4, np.nan, rng.exponential(30, size=n_tx))
    for j in range(1, 10):
        tx[f"M{j}"] = rng.choice(["T", "F", None], size=n_tx, p=[0.5, 0.3, 0.2])
    for j in range(1, 41):
        shift = 0.6 * y if j <= 10 else 0.0
        tx[f"V{j}"] = np.where(rng.random(n_tx) < (0.8 if j > 30 else 0.1), np.nan, rng.normal(0, 1, size=n_tx) + shift)

    identity = None
    if with_identity:
        has = rng.random(n_tx) < 0.6
        shared = rng.random(n_tx) < np.where(y == 1, 0.6, 0.02)
        device = np.where(shared, [f"shared_{k}" for k in rng.integers(0, 8, size=n_tx)],
                          [f"dev_{u}_{k}" for u, k in zip(user, rng.integers(0, 2, size=n_tx))])
        identity = pd.DataFrame({
            "TransactionID": tx["TransactionID"][has].to_numpy(),
            "DeviceType": rng.choice(["desktop", "mobile"], size=int(has.sum())),
            "DeviceInfo": device[has],
            "id_01": rng.normal(0, 5, size=int(has.sum())),
        })
    return tx, identity


def write_fixture(directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tx, identity = make_fixture(**kwargs)
    tx.to_csv(directory / "train_transaction.csv", index=False)
    if identity is not None:
        identity.to_csv(directory / "train_identity.csv", index=False)
    return directory


def fixture_overrides():
    return {
        "splits": {"embargo_days": 1},
        "account": {"min_positives": 3},
        "model": {"hidden": 16, "heads": 2},
        "train": {"num_neighbors": [5, 3], "batch_fraud": 256, "batch_account": 128, "epochs": 1,
                  "patience": 2, "device": "cpu", "max_steps_per_epoch": 4},
        "evaluation": {"bootstrap_samples": 50},
        "explain": {"epochs": 10, "fidelity_sample": 5, "n_explained": 2},
    }
```

`tests/conftest.py`:
```python
import pytest

from pipeline.config import load_config
from tests.fixtures import fixture_overrides, make_fixture


@pytest.fixture(scope="session")
def raw_frames():
    return make_fixture()


@pytest.fixture()
def cfg():
    return load_config(overrides=fixture_overrides())
```
Also add an empty `tests/__init__.py` so `tests.fixtures` imports resolve from the repo root.

- [ ] **Step 7: Write the failing data-loading tests** (`tests/test_data_loading.py`)

```python
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.data_loading import finalize, load_raw
from tests.fixtures import write_fixture


def cfg_for(tmp_path):
    return load_config(overrides={"data": {"dir": str(tmp_path)}})


def test_load_raw_merges_identity_sorts_and_adds_keys(tmp_path):
    write_fixture(tmp_path, n_tx=1500, n_users=80, days=30)
    df = load_raw(cfg_for(tmp_path))
    assert list(df["TransactionDT"]) == sorted(df["TransactionDT"])
    assert {"user", "email", "device", "DeviceInfo"} <= set(df.columns)
    assert df["user"].str.contains("_").all()
    assert df["device"].isna().any() and df["device"].notna().any()  # identity covers only some rows


def test_identity_file_is_optional(tmp_path):
    write_fixture(tmp_path, n_tx=800, n_users=40, days=20, with_identity=False)
    df = load_raw(cfg_for(tmp_path))
    assert df["device"].isna().all()


def test_missing_transaction_file_names_the_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_transaction.csv"):
        load_raw(cfg_for(tmp_path))


def test_missing_columns_are_named(tmp_path):
    write_fixture(tmp_path, n_tx=300, n_users=20, days=10)
    path = tmp_path / "train_transaction.csv"
    pd.read_csv(path).drop(columns=["isFraud"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="isFraud"):
        load_raw(cfg_for(tmp_path))


def test_missing_addr1_and_email_do_not_break_keys(raw_frames):
    tx, _ = raw_frames
    df = finalize(tx)
    assert df["user"].notna().all()  # a sentinel keeps users with no addr1 usable
    assert df["email"].isna().any()
```

- [ ] **Step 8: Implement** (`pipeline/data_loading.py`)

```python
"""Load and normalise the IEEE-CIS CSV files."""
from __future__ import annotations

import pandas as pd

from pipeline.config import resolve_path

REQUIRED_TX = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
               "card1", "addr1", "P_emaildomain"]


def finalize(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    if "DeviceInfo" not in tx.columns:
        tx["DeviceInfo"] = pd.NA
    tx["isFraud"] = tx["isFraud"].astype(int)
    tx["TransactionDT"] = tx["TransactionDT"].astype("int64")
    tx = tx.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    tx["user"] = (tx["card1"].fillna(-1).astype("int64").astype(str) + "_"
                  + tx["addr1"].fillna(-1).astype("int64").astype(str))
    tx["email"] = tx["P_emaildomain"]
    tx["device"] = tx["DeviceInfo"]
    return tx


def load_raw(cfg: dict) -> pd.DataFrame:
    directory = resolve_path(cfg["data"]["dir"])
    tx_path = directory / cfg["data"]["transaction_file"]
    if not tx_path.exists():
        raise FileNotFoundError(
            f"Expected {tx_path}. Download train_transaction.csv from the Kaggle competition "
            "'ieee-fraud-detection' (a Kaggle account and acceptance of the rules are required) "
            "and place it in data/ (see README)."
        )
    tx = pd.read_csv(tx_path)
    missing = [c for c in REQUIRED_TX if c not in tx.columns]
    if missing:
        raise ValueError(f"{tx_path} is missing required columns: {missing}")
    id_path = directory / cfg["data"]["identity_file"]
    if id_path.exists():
        tx = tx.merge(pd.read_csv(id_path), on="TransactionID", how="left")
    return finalize(tx)
```

- [ ] **Step 9: Run to verify pass and commit**

Run: `"$PY" -m pytest -q`
Expected: all pass.

```bash
git add configs pipeline tests
git commit -m "feat: add config, data loading and an IEEE-CIS-shaped fixture generator" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Time periods, embargo, user splits, account labels and leakage-safe masks

**Files:**
- Create: `pipeline/splits.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `pipeline.data_loading.finalize`, `pipeline.config` (via the `cfg` fixture).
- Produces in `pipeline.splits`: `DAY = 86400.0`; `Periods(t_train, val_start, val_end, test_start, embargo)` (frozen dataclass of floats); `compute_periods(times, train_frac, val_end_frac, embargo_days) -> Periods`; `period_of(times, periods) -> np.ndarray` of `"train" | "embargo" | "val" | "test"`; `split_users(users, fractions, seed) -> pd.Series` (index = user key, values `"train" | "val" | "test"`); `account_table(df, periods, user_split, min_transactions, fraud_rate_threshold) -> DataFrame` indexed by user with columns `split, n_tx, fraud_rate, eligible, label`; `check_positives(account, min_positives) -> None` (raises `ValueError`); `fraud_masks(df, periods, user_split) -> dict` with boolean arrays `train`, `val`, `test` aligned to `df`.

- [ ] **Step 1: Write the failing tests** (`tests/test_splits.py`)

```python
import numpy as np
import pandas as pd
import pytest

from pipeline.data_loading import finalize
from pipeline.splits import (DAY, account_table, check_positives, compute_periods, fraud_masks,
                             period_of, split_users)


@pytest.fixture()
def df(raw_frames):
    return finalize(raw_frames[0])


@pytest.fixture()
def periods(df, cfg):
    s = cfg["splits"]
    return compute_periods(df["TransactionDT"], s["train_frac"], s["val_end_frac"], s["embargo_days"])


@pytest.fixture()
def users(df, cfg):
    return split_users(df["user"], cfg["splits"]["user_fractions"], cfg["seed"])


def test_periods_are_ordered_and_separated_by_the_embargo(periods):
    assert periods.t_train < periods.val_start < periods.val_end < periods.test_start
    assert periods.val_start - periods.t_train == periods.embargo
    assert periods.test_start - periods.val_end == periods.embargo


def test_embargo_that_empties_a_period_is_rejected(df):
    with pytest.raises(ValueError, match="embargo_days"):
        compute_periods(df["TransactionDT"], 0.7, 0.85, embargo_days=60)


def test_split_users_is_deterministic_and_complete(df, cfg):
    a = split_users(df["user"], cfg["splits"]["user_fractions"], 0)
    b = split_users(df["user"], cfg["splits"]["user_fractions"], 0)
    assert a.equals(b) and set(a.index) == set(df["user"]) and set(a) == {"train", "val", "test"}
    assert 0.6 < (a == "train").mean() < 0.8


def test_no_fraud_train_row_belongs_to_a_held_out_user(df, periods, users):
    train = fraud_masks(df, periods, users)["train"]
    assert train.any() and set(df.loc[train, "user"].map(users)) == {"train"}


def test_every_fraud_train_row_is_inside_the_train_period(df, periods, users):
    train = fraud_masks(df, periods, users)["train"]
    assert (df.loc[train, "TransactionDT"] <= periods.t_train).all()


def test_no_evaluation_row_falls_inside_an_embargo_window(df, periods, users):
    masks = fraud_masks(df, periods, users)
    t = df["TransactionDT"]
    embargo1 = (t > periods.t_train) & (t <= periods.val_start)
    embargo2 = (t > periods.val_end) & (t <= periods.test_start)
    for name in ("val", "test"):
        assert not (masks[name] & (embargo1 | embargo2)).any()
    assert masks["val"].any() and masks["test"].any()
    assert set(period_of(t.to_numpy(), periods)) == {"train", "embargo", "val", "test"}


def test_account_labels_ignore_labels_after_t_train(df, periods, users, cfg):
    a = cfg["account"]
    base = account_table(df, periods, users, a["min_transactions"], a["fraud_rate_threshold"])
    shuffled = df.copy()
    later = shuffled["TransactionDT"] > periods.t_train
    rng = np.random.default_rng(0)
    shuffled.loc[later, "isFraud"] = rng.permutation(shuffled.loc[later, "isFraud"].to_numpy())
    again = account_table(shuffled, periods, users, a["min_transactions"], a["fraud_rate_threshold"])
    assert base.equals(again) and base["label"].sum() > 0


def test_users_without_train_period_history_are_ineligible():
    periods = compute_periods(pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 0.5, 0.8, 0.0)
    rows = [("A", 60, 1), ("A", 70, 1), ("A", 80, 1),          # only after t_train
            ("B", 1, 1), ("B", 2, 1),                          # fewer than min_transactions
            ("C", 3, 1), ("C", 4, 1), ("C", 5, 0)]             # eligible, fraud rate 2/3
    df = pd.DataFrame(rows, columns=["user", "TransactionDT", "isFraud"])
    users = pd.Series({"A": "train", "B": "train", "C": "train"}, name="split")
    t = account_table(df, periods, users, min_transactions=3, fraud_rate_threshold=0.2)
    assert t.loc["A", ["eligible", "label"]].tolist() == [False, 0]
    assert t.loc["B", ["eligible", "label"]].tolist() == [False, 0]
    assert t.loc["C", ["eligible", "label"]].tolist() == [True, 1]


def test_too_few_positives_raises_with_the_split_named():
    t = pd.DataFrame({"split": ["train"] * 5 + ["val"] * 5, "eligible": True,
                      "label": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]})
    with pytest.raises(ValueError, match="val"):
        check_positives(t, min_positives=2)
    check_positives(t, min_positives=1)
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_splits.py -q`
Expected: `ModuleNotFoundError: No module named 'pipeline.splits'`.

- [ ] **Step 3: Implement** (`pipeline/splits.py`)

```python
"""Time periods with a label-delay embargo, user splits, account labels and leakage-safe masks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86400.0


@dataclass(frozen=True)
class Periods:
    t_train: float
    val_start: float
    val_end: float
    test_start: float
    embargo: float


def compute_periods(times, train_frac, val_end_frac, embargo_days) -> Periods:
    t = pd.Series(times)
    t_train = float(t.quantile(train_frac))
    val_end = float(t.quantile(val_end_frac))
    embargo = float(embargo_days) * DAY
    if val_end <= t_train + embargo:
        raise ValueError("embargo_days is too large: the validation period would be empty")
    if float(t.max()) <= val_end + embargo:
        raise ValueError("embargo_days is too large: the test period would be empty")
    return Periods(t_train, t_train + embargo, val_end, val_end + embargo, embargo)


def period_of(times, periods: Periods) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    out = np.full(len(t), "embargo", dtype=object)
    out[t <= periods.t_train] = "train"
    out[(t > periods.val_start) & (t <= periods.val_end)] = "val"
    out[t > periods.test_start] = "test"
    return out


def split_users(users, fractions, seed) -> pd.Series:
    ids = np.array(sorted(pd.unique(pd.Series(users))))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train = int(round(fractions[0] * len(ids)))
    n_val = int(round(fractions[1] * len(ids)))
    labels = np.empty(len(ids), dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"
    return pd.Series(labels, index=ids, name="split")


def account_table(df, periods, user_split, min_transactions, fraud_rate_threshold) -> pd.DataFrame:
    """Account-level proxy label from fraud history up to `t_train` (not credit data)."""
    history = df[df["TransactionDT"] <= periods.t_train]
    agg = history.groupby("user")["isFraud"].agg(n_tx="count", fraud_rate="mean")
    table = pd.DataFrame({"split": user_split}).join(agg)
    table["n_tx"] = table["n_tx"].fillna(0).astype(int)
    table["fraud_rate"] = table["fraud_rate"].fillna(0.0)
    table["eligible"] = table["n_tx"] >= min_transactions
    table["label"] = (table["eligible"] & (table["fraud_rate"] >= fraud_rate_threshold)).astype(int)
    return table


def check_positives(account: pd.DataFrame, min_positives: int) -> None:
    pos = account[account["eligible"]].groupby("split")["label"].sum()
    for split in ("train", "val", "test"):
        n = int(pos.get(split, 0))
        if n < min_positives:
            raise ValueError(
                f"account split '{split}' has {n} positive users, fewer than the required {min_positives}; "
                "adjust account.min_transactions or account.fraud_rate_threshold (by prevalence only)"
            )


def fraud_masks(df, periods, user_split) -> dict:
    period = period_of(df["TransactionDT"].to_numpy(), periods)
    split = df["user"].map(user_split).to_numpy()
    return {"train": (period == "train") & (split == "train"),
            "val": period == "val", "test": period == "test"}
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_splits.py -q`
Expected: all pass.

```bash
git add pipeline/splits.py tests/test_splits.py
git commit -m "feat: add periods with embargo, user splits, account labels and leakage-safe masks" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Feature transformer fitted on the train period

**Files:**
- Create: `pipeline/feature_engineering.py`, `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: `pipeline.splits.Periods`.
- Produces in `pipeline.feature_engineering`: `FeatureTransformer(max_v_columns=100, v_max_missing=0.5)` with `.fit(df) -> self`, `.transform(df) -> np.ndarray` (float32, shape `(len(df), len(feature_names))`, no NaN or inf) and attributes `feature_names: list[str]`, `v_cols`, `log_amt_mean`, `log_amt_std`; `fit_on_train(df, periods, cfg) -> FeatureTransformer` (fits on rows with `TransactionDT <= periods.t_train` only).

- [ ] **Step 1: Write the failing tests** (`tests/test_feature_engineering.py`)

```python
import numpy as np
import pandas as pd
import pytest

from pipeline.data_loading import finalize
from pipeline.feature_engineering import FeatureTransformer, fit_on_train
from pipeline.splits import compute_periods


@pytest.fixture()
def df(raw_frames):
    return finalize(raw_frames[0])


@pytest.fixture()
def periods(df, cfg):
    s = cfg["splits"]
    return compute_periods(df["TransactionDT"], s["train_frac"], s["val_end_frac"], s["embargo_days"])


def test_statistics_come_from_train_rows_only(df, periods, cfg):
    train = df[df["TransactionDT"] <= periods.t_train]
    ft = fit_on_train(df, periods, cfg)
    assert ft.log_amt_mean == pytest.approx(np.log1p(train["TransactionAmt"]).mean())
    poisoned = df.copy()
    later = poisoned["TransactionDT"] > periods.t_train
    poisoned.loc[later, "TransactionAmt"] = poisoned.loc[later, "TransactionAmt"] * 1000
    ft2 = fit_on_train(poisoned, periods, cfg)
    assert ft2.log_amt_mean == ft.log_amt_mean and ft2.feature_names == ft.feature_names


def test_transform_shape_dtype_and_finiteness(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    x = ft.transform(df)
    assert x.shape == (len(df), len(ft.feature_names)) and x.dtype == np.float32
    assert np.isfinite(x).all()


def test_v_columns_respect_the_missing_rate_and_the_cap(df, periods, cfg):
    ft = FeatureTransformer(max_v_columns=5, v_max_missing=0.5).fit(df[df["TransactionDT"] <= periods.t_train])
    assert len(ft.v_cols) == 5
    assert not {f"V{j}" for j in range(31, 41)} & set(ft.v_cols)  # those are 80% missing


def test_unseen_category_gets_zero_frequency(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    row = df.iloc[[0]].copy()
    row["P_emaildomain"] = "never-seen.example"
    x = ft.transform(row)
    assert x[0, ft.feature_names.index("freq_P_emaildomain")] == 0.0


def test_train_features_are_centred(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    x = ft.transform(df[df["TransactionDT"] <= periods.t_train])
    assert abs(x[:, ft.feature_names.index("C1")].mean()) < 1e-3


def test_missing_values_get_indicators_not_nans(df, periods, cfg):
    ft = fit_on_train(df, periods, cfg)
    assert any(name.startswith("na_D") for name in ft.feature_names)
    x = ft.transform(df.head(50))
    assert np.isfinite(x).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_feature_engineering.py -q`
Expected: `ModuleNotFoundError: No module named 'pipeline.feature_engineering'`.

- [ ] **Step 3: Implement** (`pipeline/feature_engineering.py`)

```python
"""Per-transaction features. Every statistic is fitted on the train period only."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CAT_BASE = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
            "P_emaildomain", "R_emaildomain"]


def _cols(df: pd.DataFrame, prefix: str) -> list[str]:
    pat = re.compile(rf"^{prefix}(\d+)$")
    found = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(c))]
    return [c for _, c in sorted(found)]


class FeatureTransformer:
    def __init__(self, max_v_columns: int = 100, v_max_missing: float = 0.5):
        self.max_v_columns = max_v_columns
        self.v_max_missing = v_max_missing

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        self.cat_cols = [c for c in CAT_BASE + _cols(df, "M") if c in df.columns]
        self.freq = {c: df[c].astype("string").fillna("nan").value_counts(normalize=True).to_dict()
                     for c in self.cat_cols}
        v_all = _cols(df, "V")
        missing = df[v_all].isna().mean() if v_all else pd.Series(dtype=float)
        keep = list(missing[missing <= self.v_max_missing].index)
        variances = df[keep].var().sort_values(ascending=False) if keep else pd.Series(dtype=float)
        self.v_cols = list(variances.index[: self.max_v_columns])
        self.num_cols = _cols(df, "C") + _cols(df, "D") + self.v_cols
        self.na_cols = [c for c in _cols(df, "D") + self.v_cols if df[c].isna().any()]
        log_amt = np.log1p(df["TransactionAmt"].astype(float))
        self.log_amt_mean = float(log_amt.mean())
        self.log_amt_std = float(log_amt.std()) or 1.0
        self.stats = {c: (float(df[c].mean()), float(df[c].std()) or 1.0) for c in self.num_cols}
        self.feature_names = (["log_amt", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
                              + [f"freq_{c}" for c in self.cat_cols] + list(self.num_cols)
                              + [f"na_{c}" for c in self.na_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        log_amt = np.log1p(df["TransactionAmt"].astype(float).to_numpy())
        cols.append((log_amt - self.log_amt_mean) / self.log_amt_std)
        hour = (df["TransactionDT"].to_numpy() // 3600) % 24
        dow = (df["TransactionDT"].to_numpy() // 86400) % 7
        cols += [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                 np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
        for c in self.cat_cols:
            mapped = df[c].astype("string").fillna("nan").map(self.freq[c])
            cols.append(mapped.fillna(0.0).to_numpy(dtype=float))
        for c in self.num_cols:
            mean, std = self.stats[c]
            values = df[c].astype(float).to_numpy() if c in df.columns else np.full(len(df), np.nan)
            cols.append(np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0))
        for c in self.na_cols:
            values = df[c].to_numpy() if c in df.columns else np.full(len(df), np.nan)
            cols.append(pd.isna(values).astype(float))
        return np.column_stack(cols).astype(np.float32)


def fit_on_train(df: pd.DataFrame, periods, cfg: dict) -> FeatureTransformer:
    train = df[df["TransactionDT"] <= periods.t_train]
    f = cfg["features"]
    return FeatureTransformer(f["max_v_columns"], f["v_max_missing"]).fit(train)
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_feature_engineering.py -q`
Expected: all pass.

```bash
git add pipeline/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: add the train-fitted feature transformer" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: History-aware features (strictly earlier transactions only)

**Files:**
- Create: `pipeline/history_features.py`, `tests/test_history_features.py`

**Interfaces:**
- Produces in `pipeline.history_features`: `history_features(df) -> DataFrame` indexed like `df`, requiring columns `user`, `email`, `device`, `TransactionDT`, `TransactionAmt`. Columns: for each entity `user`, `email`, `device`: `<e>_n_prior, <e>_n_1d, <e>_n_7d, <e>_amt_mean, <e>_amt_std, <e>_secs_since_prev`; plus `user_distinct_devices` and `user_distinct_emails`. All use only transactions with a strictly smaller `TransactionDT` than the row (never labels); rows without that entity get NaN.

- [ ] **Step 1: Write the failing tests** (`tests/test_history_features.py`)

```python
import numpy as np
import pandas as pd
import pytest

from pipeline.data_loading import finalize
from pipeline.history_features import history_features


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "email", "device", "TransactionDT", "TransactionAmt"])


def test_hand_checked_example():
    df = frame([("u", "e", "A", 0, 10.0), ("u", "e", "A", 100, 20.0), ("u", "e", "B", 100_000, 30.0)])
    h = history_features(df)
    assert h["user_n_prior"].tolist() == [0, 1, 2]
    assert np.isnan(h["user_amt_mean"].iloc[0]) and h["user_amt_mean"].tolist()[1:] == [10.0, 15.0]
    assert h["user_n_1d"].tolist() == [0, 1, 0]       # only the 100 s old row is within a day of row 2? no: see below
    assert h["user_n_7d"].tolist() == [0, 1, 2]
    assert np.isnan(h["user_secs_since_prev"].iloc[0]) and h["user_secs_since_prev"].tolist()[1:] == [100.0, 99_900.0]
    assert h["user_distinct_devices"].tolist() == [0, 1, 1]


def test_equal_timestamps_do_not_count_each_other():
    df = frame([("u", "e", "A", 50, 1.0), ("u", "e", "A", 100, 2.0), ("u", "e", "A", 100, 3.0)])
    assert history_features(df)["user_n_prior"].tolist() == [0, 1, 1]


def test_features_never_depend_on_later_transactions(raw_frames):
    df = finalize(raw_frames[0])
    cutoff = df["TransactionDT"].quantile(0.5)
    base = history_features(df)
    perturbed = df.copy()
    later = perturbed["TransactionDT"] > cutoff
    perturbed.loc[later, "TransactionAmt"] = perturbed.loc[later, "TransactionAmt"] * 10
    again = history_features(perturbed)
    early = (df["TransactionDT"] <= cutoff).to_numpy()
    pd.testing.assert_frame_equal(base[early], again[early])


def test_features_do_not_use_labels(raw_frames):
    df = finalize(raw_frames[0])
    flipped = df.copy()
    flipped["isFraud"] = 1 - flipped["isFraud"]
    pd.testing.assert_frame_equal(history_features(df), history_features(flipped))


def test_missing_entities_give_nan_not_crashes():
    df = frame([("u", None, None, 0, 5.0), ("u", "e", None, 10, 6.0), ("v", None, "D", 20, 7.0)])
    h = history_features(df)
    assert h["email_n_prior"].isna().tolist() == [True, False, True]
    assert h["device_n_prior"].isna().tolist() == [True, True, False]
    assert h["user_n_prior"].notna().all()
```
Note for the executor: in `test_hand_checked_example` the third row (time 100,000) is more than a day (86,400 s) after both earlier rows, so `user_n_1d` is `[0, 1, 0]` (row 1 sees row 0 at a 100 s gap; row 2 sees neither within a day). Delete the stray comment in the test when adding it.

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_history_features.py -q`
Expected: `ModuleNotFoundError: No module named 'pipeline.history_features'`.

- [ ] **Step 3: Implement** (`pipeline/history_features.py`)

```python
"""Label-free, strictly-causal history features for the tabular baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400


def _per_entity(codes: np.ndarray, t: np.ndarray, a: np.ndarray, distinct: dict | None = None) -> dict:
    n = len(t)
    out = {k: np.full(n, np.nan) for k in ("n_prior", "n_1d", "n_7d", "amt_mean", "amt_std", "secs_since_prev")}
    for name in (distinct or {}):
        out[f"distinct_{name}"] = np.full(n, np.nan)
    valid = np.flatnonzero(codes >= 0)
    if len(valid) == 0:
        return out
    order = valid[np.lexsort((t[valid], codes[valid]))]  # by entity, then time
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    for grp in np.split(order, cuts):
        times, amts = t[grp], a[grp]
        prior = np.searchsorted(times, times, side="left")  # count of strictly earlier rows
        cs = np.concatenate([[0.0], np.cumsum(amts)])
        cs2 = np.concatenate([[0.0], np.cumsum(amts ** 2)])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(prior > 0, cs[prior] / np.maximum(prior, 1), np.nan)
            var = np.where(prior > 1, (cs2[prior] - prior * mean ** 2) / np.maximum(prior - 1, 1), np.nan)
        out["n_prior"][grp] = prior
        out["n_1d"][grp] = prior - np.searchsorted(times, times - DAY, side="left")
        out["n_7d"][grp] = prior - np.searchsorted(times, times - 7 * DAY, side="left")
        out["amt_mean"][grp] = mean
        out["amt_std"][grp] = np.sqrt(np.maximum(var, 0.0))
        out["secs_since_prev"][grp] = np.where(prior > 0, times - times[np.maximum(prior - 1, 0)], np.nan)
        for name, other in (distinct or {}).items():
            e = other[grp]
            first = (e >= 0) & ~pd.Series(e).duplicated().to_numpy()
            out[f"distinct_{name}"][grp] = np.concatenate([[0], np.cumsum(first)])[prior]
    return out


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["TransactionDT"].to_numpy(np.int64)
    a = df["TransactionAmt"].to_numpy(float)
    user = pd.factorize(df["user"])[0]
    email = pd.factorize(df["email"])[0]
    device = pd.factorize(df["device"])[0]
    blocks = {
        "user": _per_entity(user, t, a, distinct={"devices": device, "emails": email}),
        "email": _per_entity(email, t, a),
        "device": _per_entity(device, t, a),
    }
    cols = {f"{ent}_{name}": arr for ent, block in blocks.items() for name, arr in block.items()}
    return pd.DataFrame(cols, index=df.index)
```
The column names produced for the distinct counts are `user_distinct_devices` and `user_distinct_emails`.

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_history_features.py -q`
Expected: all pass.

```bash
git add pipeline/history_features.py tests/test_history_features.py
git commit -m "feat: add strictly-causal history features for the tabular baseline" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Graph builder and the prepared pipeline

**Files:**
- Create: `pipeline/graph_builder.py`, `pipeline/prepare.py`, `tests/test_graph_builder.py`

**Interfaces:**
- Consumes: Tasks 3-6.
- Produces in `pipeline.graph_builder`: `EDGE_TYPES` (the six edge-type tuples listed below) and `build_graph(df, x, periods, user_split, account, entity_codes=None) -> HeteroData`. Node types: `transaction` (`x` float32, `y` long, `time` long, boolean `train_mask`, `val_mask`, `test_mask`), `user`, `email_domain`, `device` (each: `num_nodes`, `time` = first-seen `TransactionDT` as long; `user` also has `y` (account label), and boolean `train_mask`, `val_mask`, `test_mask` that are true only for *eligible* users in that split). Edge types: `("user","owns","transaction")`, `("transaction","rev_owns","user")`, `("transaction","uses_email","email_domain")`, `("email_domain","rev_uses_email","transaction")`, `("transaction","uses_device","device")`, `("device","rev_uses_device","transaction")`. Rows without an email or device get no edge of that kind. `data.t_train` is the integer `periods.t_train`.
- Produces in `pipeline.prepare`: `Prepared` (dataclass: `cfg, df, periods, user_split, account, transformer, x, hist, graph, entity_index`) and `prepare(cfg, df=None) -> Prepared` (loads data when `df` is None; accepts a raw frame and finalises it). `entity_index` maps `"user" | "email_domain" | "device"` to a `pd.Index` of keys in node-id order.

- [ ] **Step 1: Write the failing tests** (`tests/test_graph_builder.py`)

```python
import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.graph_builder import EDGE_TYPES
from pipeline.prepare import prepare


@pytest.fixture(scope="module")
def prep(raw_frames):
    from pipeline.config import load_config
    from tests.fixtures import fixture_overrides

    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_node_counts(prep):
    g, df = prep.graph, prep.df
    assert g["transaction"].num_nodes == len(df)
    assert g["user"].num_nodes == df["user"].nunique()
    assert g["email_domain"].num_nodes == df["email"].nunique()
    assert g["device"].num_nodes == df["device"].nunique()


def test_times_are_transaction_times_and_first_seen_for_entities(prep):
    g, df = prep.graph, prep.df
    assert torch.equal(g["transaction"].time, torch.as_tensor(df["TransactionDT"].to_numpy(np.int64)))
    first = df.groupby("user")["TransactionDT"].min()
    names = prep.entity_index["user"]
    for node in (0, 5, 17):
        assert int(g["user"].time[node]) == int(first[names[node]])


def test_rows_without_entities_have_no_edges(prep):
    g, df = prep.graph, prep.df
    ei = g["transaction", "uses_email", "email_domain"].edge_index
    assert ei.shape[1] == df["email"].notna().sum()
    no_email = set(np.flatnonzero(df["email"].isna().to_numpy()))
    assert not no_email & set(ei[0].tolist())
    dv = g["transaction", "uses_device", "device"].edge_index
    assert dv.shape[1] == df["device"].notna().sum() and df["device"].isna().any()


def test_every_row_has_a_user_edge_and_reverse_edges_mirror(prep):
    g = prep.graph
    fwd = g["user", "owns", "transaction"].edge_index
    rev = g["transaction", "rev_owns", "user"].edge_index
    assert fwd.shape[1] == g["transaction"].num_nodes
    assert torch.equal(fwd.flip(0), rev)
    assert set(g.edge_types) == set(EDGE_TYPES)


def test_masks_follow_the_splits(prep):
    g, df = prep.graph, prep.df
    train = g["transaction"].train_mask.numpy()
    assert set(df.loc[train, "user"].map(prep.user_split)) == {"train"}
    assert (df.loc[train, "TransactionDT"] <= prep.periods.t_train).all()
    for name in ("train", "val", "test"):
        m = g["user"][f"{name}_mask"].numpy()
        assert set(prep.account.loc[prep.entity_index["user"][m], "split"]) <= {name}
        assert prep.account.loc[prep.entity_index["user"][m], "eligible"].all()
    assert g.t_train == int(prep.periods.t_train)


def test_features_and_history_are_aligned_with_rows(prep):
    assert prep.x.shape[0] == len(prep.df) == prep.hist.shape[0]
    assert prep.graph["transaction"].x.shape[1] == len(prep.transformer.feature_names)
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_graph_builder.py -q`
Expected: `ModuleNotFoundError: No module named 'pipeline.graph_builder'`.

- [ ] **Step 3: Implement** (`pipeline/graph_builder.py`)

```python
"""Build the time-stamped heterogeneous graph."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from pipeline.splits import fraud_masks

I64MAX = np.iinfo(np.int64).max
EDGE_TYPES = [
    ("user", "owns", "transaction"), ("transaction", "rev_owns", "user"),
    ("transaction", "uses_email", "email_domain"), ("email_domain", "rev_uses_email", "transaction"),
    ("transaction", "uses_device", "device"), ("device", "rev_uses_device", "transaction"),
]
ENTITIES = (("user", "user", "owns", "rev_owns", True),
            ("email_domain", "email", "uses_email", "rev_uses_email", False),
            ("device", "device", "uses_device", "rev_uses_device", False))


def build_graph(df, x, periods, user_split, account, entity_codes=None):
    n = len(df)
    times = df["TransactionDT"].to_numpy(np.int64)
    data = HeteroData()
    data["transaction"].x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    data["transaction"].y = torch.as_tensor(df["isFraud"].to_numpy(np.int64))
    data["transaction"].time = torch.as_tensor(times)
    masks = fraud_masks(df, periods, user_split)
    for name, mask in masks.items():
        data["transaction"][f"{name}_mask"] = torch.as_tensor(mask)

    rows_all = np.arange(n)
    index = {}
    for node_type, col, fwd_name, rev_name, owns in ENTITIES:
        codes, uniques = pd.factorize(df[col])
        index[node_type] = pd.Index(uniques)
        n_ent = len(uniques)
        data[node_type].num_nodes = n_ent
        valid = codes >= 0
        first = np.full(n_ent, I64MAX, dtype=np.int64)
        np.minimum.at(first, codes[valid], times[valid])
        data[node_type].time = torch.as_tensor(first)
        rows = rows_all[valid]
        ent = torch.as_tensor(codes[valid], dtype=torch.long)
        tx = torch.as_tensor(rows, dtype=torch.long)
        if owns:
            data[node_type, fwd_name, "transaction"].edge_index = torch.stack([ent, tx])
            data["transaction", rev_name, node_type].edge_index = torch.stack([tx, ent])
        else:
            data["transaction", fwd_name, node_type].edge_index = torch.stack([tx, ent])
            data[node_type, rev_name, "transaction"].edge_index = torch.stack([ent, tx])

    table = account.reindex(index["user"])
    data["user"].y = torch.as_tensor(table["label"].to_numpy(np.int64))
    eligible = table["eligible"].to_numpy(bool)
    for name in ("train", "val", "test"):
        data["user"][f"{name}_mask"] = torch.as_tensor((table["split"].to_numpy() == name) & eligible)
    data.t_train = int(periods.t_train)
    return data, index
```
`build_graph` returns the graph and the entity index (`dict` of `pd.Index`). In the test above `prep.entity_index` is that dict.

`pipeline/prepare.py`:
```python
"""One call that turns raw data into everything the training, baseline and serving code needs."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.data_loading import finalize, load_raw
from pipeline.feature_engineering import fit_on_train
from pipeline.graph_builder import build_graph
from pipeline.history_features import history_features
from pipeline.splits import account_table, check_positives, compute_periods, split_users


@dataclass
class Prepared:
    cfg: dict
    df: pd.DataFrame
    periods: object
    user_split: pd.Series
    account: pd.DataFrame
    transformer: object
    x: object
    hist: pd.DataFrame
    graph: object
    entity_index: dict


def prepare(cfg: dict, df: pd.DataFrame | None = None) -> Prepared:
    df = load_raw(cfg) if df is None else finalize(df)
    s, a = cfg["splits"], cfg["account"]
    periods = compute_periods(df["TransactionDT"], s["train_frac"], s["val_end_frac"], s["embargo_days"])
    user_split = split_users(df["user"], s["user_fractions"], cfg["seed"])
    account = account_table(df, periods, user_split, a["min_transactions"], a["fraud_rate_threshold"])
    check_positives(account, a["min_positives"])
    transformer = fit_on_train(df, periods, cfg)
    x = transformer.transform(df)
    hist = history_features(df)
    graph, entity_index = build_graph(df, x, periods, user_split, account)
    return Prepared(cfg, df, periods, user_split, account, transformer, x, hist, graph, entity_index)
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_graph_builder.py -q`
Expected: all pass. If `check_positives` raises on the fixture, raise the fixture's `n_users` or lower `account.min_positives` in `fixture_overrides()` and ledger it.

```bash
git add pipeline/graph_builder.py pipeline/prepare.py tests/test_graph_builder.py
git commit -m "feat: add the time-stamped hetero graph builder and the prepared pipeline" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Model: backbone, heads and focal loss

**Files:**
- Create: `models/__init__.py` (empty), `models/gnn_backbone.py`, `models/multi_task_heads.py`, `models/losses.py`, `tests/test_models.py`

**Interfaces:**
- Produces in `models.gnn_backbone`: `HeteroGNN(tx_in, edge_types, hidden=64, heads=4, layers=2, dropout=0.2, conv="gatv2", entity_types=("user","email_domain","device"))` with `forward(x_dict, edge_index_dict, num_nodes_dict) -> dict[str, Tensor]` (embeddings per node type, hidden-dim) and `encode_tabular(x_tx) -> Tensor` (the no-graph path). PyG convolutions initialise lazily, so run one forward pass before creating an optimiser.
- Produces in `models.multi_task_heads`: `TaskHeads(hidden, dropout=0.2)` with `fraud_logits(h_tx) -> Tensor[N]` and `account_logits(h_user) -> Tensor[N]`.
- Produces in `models.losses`: `focal_loss(logits, targets, alpha=0.75, gamma=2.0, reduction="mean")` (targets float 0/1).

- [ ] **Step 1: Write the failing tests** (`tests/test_models.py`)

```python
import pytest
import torch
import torch.nn.functional as F

from models.gnn_backbone import HeteroGNN
from models.losses import focal_loss
from models.multi_task_heads import TaskHeads
from pipeline.graph_builder import EDGE_TYPES


def tiny_batch(n_tx=6, n_user=3, n_email=2, n_device=0, f=5):
    x = {"transaction": torch.randn(n_tx, f)}
    tx = torch.arange(n_tx)
    user = tx % n_user
    email = tx % n_email
    edges = {
        ("user", "owns", "transaction"): torch.stack([user, tx]),
        ("transaction", "rev_owns", "user"): torch.stack([tx, user]),
        ("transaction", "uses_email", "email_domain"): torch.stack([tx, email]),
        ("email_domain", "rev_uses_email", "transaction"): torch.stack([email, tx]),
    }
    return x, edges, {"transaction": n_tx, "user": n_user, "email_domain": n_email, "device": n_device}


@pytest.mark.parametrize("conv", ["gatv2", "sage"])
def test_forward_shapes_for_each_conv_including_an_empty_node_type(conv):
    x, edges, nn_ = tiny_batch()
    model = HeteroGNN(5, EDGE_TYPES, hidden=16, heads=2, layers=2, dropout=0.0, conv=conv)
    h = model(x, edges, nn_)
    assert h["transaction"].shape == (6, 16) and h["user"].shape == (3, 16)
    assert h["device"].shape == (0, 16)
    assert torch.isfinite(h["transaction"]).all()


def test_gradients_flow_to_the_transaction_projection_and_type_vectors():
    x, edges, nn_ = tiny_batch()
    model = HeteroGNN(5, EDGE_TYPES, hidden=8, heads=2, dropout=0.0)
    h = model(x, edges, nn_)
    h["transaction"].sum().backward()
    assert model.tx_proj.weight.grad is not None and model.type_vec["user"].grad is not None


def test_tabular_path_ignores_edges_and_has_the_same_width():
    x, _, _ = tiny_batch()
    model = HeteroGNN(5, EDGE_TYPES, hidden=8, heads=2, dropout=0.0)
    assert model.encode_tabular(x["transaction"]).shape == (6, 8)


def test_heads_return_one_logit_per_node():
    heads = TaskHeads(8, dropout=0.0)
    assert heads.fraud_logits(torch.randn(4, 8)).shape == (4,)
    assert heads.account_logits(torch.randn(3, 8)).shape == (3,)


def test_focal_loss_reduces_to_scaled_cross_entropy_at_gamma_zero():
    logits, y = torch.randn(50), (torch.rand(50) > 0.5).float()
    got = focal_loss(logits, y, alpha=0.5, gamma=0.0)
    assert torch.allclose(got, 0.5 * F.binary_cross_entropy_with_logits(logits, y))


def test_alpha_weights_positives_and_negatives():
    logits = torch.tensor([0.3, 0.3])
    pos = focal_loss(logits[:1], torch.tensor([1.0]), alpha=0.75, gamma=0.0)
    neg = focal_loss(logits[1:], torch.tensor([0.0]), alpha=0.75, gamma=0.0)
    assert pos == pytest.approx(0.75 * F.binary_cross_entropy_with_logits(logits[:1], torch.tensor([1.0])).item())
    assert neg == pytest.approx(0.25 * F.binary_cross_entropy_with_logits(logits[1:], torch.tensor([0.0])).item())


def test_gamma_downweights_easy_examples_and_none_reduction_keeps_shape():
    logits, y = torch.tensor([6.0, -6.0]), torch.tensor([1.0, 0.0])  # both confidently correct
    assert focal_loss(logits, y, 0.75, 2.0).item() < focal_loss(logits, y, 0.75, 0.0).item()
    assert focal_loss(logits, y, 0.75, 2.0, reduction="none").shape == (2,)
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_models.py -q`
Expected: `ModuleNotFoundError: No module named 'models'`.

- [ ] **Step 3: Implement**

`models/losses.py`:
```python
"""Focal loss for heavily imbalanced binary targets."""
import torch
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.75, gamma=2.0, reduction="mean"):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
```

`models/multi_task_heads.py`:
```python
"""Transaction (fraud) and account (risk) heads on the shared embeddings."""
import torch.nn as nn


def _mlp(hidden, dropout):
    return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class TaskHeads(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.fraud = _mlp(hidden, dropout)
        self.account = _mlp(hidden, dropout)

    def fraud_logits(self, h_tx):
        return self.fraud(h_tx).squeeze(-1)

    def account_logits(self, h_user):
        return self.account(h_user).squeeze(-1)
```

`models/gnn_backbone.py`:
```python
"""Heterogeneous GATv2/SAGE backbone with residual connections and learned per-type input vectors."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv


def _make_conv(kind, hidden, heads, dropout):
    if kind == "gatv2":
        return GATv2Conv((-1, -1), hidden, heads=heads, concat=False, add_self_loops=False, dropout=dropout)
    return SAGEConv((-1, -1), hidden)


class HeteroGNN(nn.Module):
    def __init__(self, tx_in, edge_types, hidden=64, heads=4, layers=2, dropout=0.2, conv="gatv2",
                 entity_types=("user", "email_domain", "device")):
        super().__init__()
        self.hidden, self.dropout = hidden, dropout
        self.entity_types = tuple(entity_types)
        self.tx_proj = nn.Linear(tx_in, hidden)
        self.type_vec = nn.ParameterDict({nt: nn.Parameter(torch.randn(hidden) * 0.1) for nt in self.entity_types})
        self.convs = nn.ModuleList([
            HeteroConv({et: _make_conv(conv, hidden, heads, dropout) for et in edge_types}, aggr="sum")
            for _ in range(layers)])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in ("transaction",) + self.entity_types})
            for _ in range(layers)])

    def encode_tabular(self, x_tx):
        return F.relu(self.tx_proj(x_tx))

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        h = {"transaction": self.encode_tabular(x_dict["transaction"])}
        for nt in self.entity_types:
            h[nt] = self.type_vec[nt].unsqueeze(0).repeat(int(num_nodes_dict.get(nt, 0)), 1)
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict)
            h = {nt: norm[nt](F.dropout(F.relu(out[nt]), self.dropout, self.training) + h[nt]) if nt in out else h[nt]
                 for nt in h}
        return h
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_models.py -q`
Expected: all pass.

```bash
git add models tests/test_models.py
git commit -m "feat: add the hetero GNN backbone, task heads and focal loss" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Trainer: alternating batches, variants, temporal sampling, early stopping

**Files:**
- Create: `scripts/__init__.py` (empty), `scripts/train.py`, `tests/test_train.py`
- Depends on: `pipeline/metrics.py` (`pr_auc`), which Task 10 creates. Do Task 10 first, or create `pipeline/metrics.py` with just `pr_auc` and `roc_auc` now (Task 10 extends it and keeps these signatures).

**Interfaces:**
- Consumes: `Prepared`, `HeteroGNN`, `TaskHeads`, `focal_loss`, `pr_auc`, config keys under `train` and `model`.
- Produces in `scripts.train`:
  - `RunSpec(variant, seed, fraud_train_idx, fraud_val_idx, account_train_idx, account_val_idx, epochs=None)` (index arrays are `np.ndarray` of node ids; `epochs` set means fixed epochs and no validation).
  - `default_spec(prepared, variant, seed) -> RunSpec` (indices from the graph masks).
  - `make_loader(prepared, node_type, idx, batch_size, shuffle) -> NeighborLoader` (temporal when `train.temporal`; account seeds all carry time `graph.t_train`).
  - `build_modules(prepared, device) -> (HeteroGNN, TaskHeads)`; `warm_up(model, heads, prepared, device) -> None` (one forward pass so PyG's lazy layers get their shapes); `load_checkpoint(path, prepared, device) -> (model, heads, meta: dict)`.
  - `predict_scores(prepared, model, heads, kind, idx, device, batch_size=None, variant="multitask") -> np.ndarray` of probabilities aligned to `idx` (`kind` is `"fraud"` or `"account"`; `variant="no_graph"` uses the tabular path).
  - `TrainResult(model, heads, best_epoch, best_score, history)` and `train_variant(prepared, spec, device="cpu", checkpoint_path=None) -> TrainResult`.
  - `main(argv=None)` CLI: `--variant --seed --config --device`; saves `models/<variant>_seed<seed>.pt`.

- [ ] **Step 1: Write the failing tests** (`tests/test_train.py`)

```python
import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.train import (default_spec, load_checkpoint, make_loader, predict_scores, train_variant)
from tests.fixtures import fixture_overrides


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


@pytest.mark.parametrize("variant", ["multitask", "fraud_only", "account_only", "no_graph"])
def test_every_variant_trains_one_epoch(prep, variant):
    res = train_variant(prep, default_spec(prep, variant, 0), "cpu")
    assert len(res.history) == 1 and np.isfinite(res.history[0]["train_loss"])


def test_checkpoint_round_trip_reproduces_predictions(prep, tmp_path):
    spec = default_spec(prep, "multitask", 0)
    res = train_variant(prep, spec, "cpu", tmp_path / "m.pt")
    idx = spec.fraud_val_idx[:64]
    before = predict_scores(prep, res.model, res.heads, "fraud", idx, "cpu")
    model, heads, meta = load_checkpoint(tmp_path / "m.pt", prep, "cpu")
    after = predict_scores(prep, model, heads, "fraud", idx, "cpu")
    assert np.allclose(before, after, atol=1e-5) and meta["variant"] == "multitask"


def test_predictions_are_probabilities_aligned_with_the_requested_rows(prep):
    spec = default_spec(prep, "multitask", 0)
    res = train_variant(prep, spec, "cpu")
    idx = spec.fraud_val_idx[:40]
    p = predict_scores(prep, res.model, res.heads, "fraud", idx, "cpu")
    assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all()
    u = predict_scores(prep, res.model, res.heads, "account", spec.account_val_idx, "cpu")
    assert u.shape == (len(spec.account_val_idx),)


def test_fixed_epochs_skip_validation(prep):
    spec = default_spec(prep, "fraud_only", 0)
    spec.epochs = 2
    res = train_variant(prep, spec, "cpu")
    assert [h["epoch"] for h in res.history] == [1, 2] and res.best_epoch == 2
    assert all(h.get("val_score") is None for h in res.history)


def test_training_is_reproducible_for_a_seed(prep):
    a = train_variant(prep, default_spec(prep, "fraud_only", 3), "cpu")
    b = train_variant(prep, default_spec(prep, "fraud_only", 3), "cpu")
    assert a.history[0]["train_loss"] == pytest.approx(b.history[0]["train_loss"], rel=1e-4)


def test_sampled_neighbours_never_come_from_the_future(prep):
    if not prep.cfg["train"]["temporal"]:
        pytest.skip("temporal sampling disabled (Option A fallback)")
    idx = np.flatnonzero(prep.graph["transaction"].val_mask.numpy())[:64]
    batch = next(iter(make_loader(prep, "transaction", idx, 64, False)))
    seed_time = batch["transaction"].time[: batch["transaction"].batch_size]
    for nt in batch.node_types:
        assert (batch[nt].time <= seed_time[batch[nt].batch]).all(), nt
    users = np.flatnonzero(prep.graph["user"].val_mask.numpy())[:16]
    ub = next(iter(make_loader(prep, "user", users, 16, False)))
    assert (ub["transaction"].time <= prep.graph.t_train).all()


def test_empty_seed_set_gives_empty_predictions(prep):
    spec = default_spec(prep, "fraud_only", 0)
    res = train_variant(prep, spec, "cpu")
    assert predict_scores(prep, res.model, res.heads, "fraud", np.array([], dtype=int), "cpu").shape == (0,)
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_train.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.train'` (or `pipeline.metrics`).

- [ ] **Step 3: Implement** (`scripts/train.py`)

```python
"""Train the GNN variants with alternating transaction/user mini-batches and temporal neighbour sampling."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from models.gnn_backbone import HeteroGNN
from models.losses import focal_loss
from models.multi_task_heads import TaskHeads
from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.metrics import pr_auc

FRAUD_VARIANTS = ("multitask", "fraud_only", "no_graph")
ACCOUNT_VARIANTS = ("multitask", "account_only")


@dataclass
class RunSpec:
    variant: str
    seed: int
    fraud_train_idx: np.ndarray
    fraud_val_idx: np.ndarray
    account_train_idx: np.ndarray
    account_val_idx: np.ndarray
    epochs: int | None = None


@dataclass
class TrainResult:
    model: torch.nn.Module
    heads: torch.nn.Module
    best_epoch: int
    best_score: float
    history: list


def default_spec(prepared, variant: str, seed: int) -> RunSpec:
    g = prepared.graph

    def idx(node_type, name):
        return np.flatnonzero(g[node_type][f"{name}_mask"].numpy())

    return RunSpec(variant, seed, idx("transaction", "train"), idx("transaction", "val"),
                   idx("user", "train"), idx("user", "val"))


def _times(prepared, node_type, idx):
    if node_type == "transaction":
        return prepared.graph["transaction"].time.numpy()[idx]
    return np.full(len(idx), int(prepared.graph.t_train), dtype=np.int64)


def make_loader(prepared, node_type, idx, batch_size, shuffle):
    t = prepared.cfg["train"]
    idx = np.asarray(idx)
    kwargs = dict(data=prepared.graph, num_neighbors=list(t["num_neighbors"]),
                  input_nodes=(node_type, torch.as_tensor(idx, dtype=torch.long)),
                  batch_size=batch_size, shuffle=shuffle)
    if t["temporal"]:
        kwargs.update(time_attr="time", temporal_strategy=t["temporal_strategy"],
                      input_time=torch.as_tensor(_times(prepared, node_type, idx), dtype=torch.long))
    return NeighborLoader(**kwargs)


def build_modules(prepared, device):
    m, g = prepared.cfg["model"], prepared.graph
    model = HeteroGNN(g["transaction"].x.shape[1], list(g.edge_types), m["hidden"], m["heads"],
                      m["layers"], m["dropout"], m["conv"])
    return model.to(device), TaskHeads(m["hidden"], m["dropout"]).to(device)


def _graph_logits(model, heads, batch, node_type, kind, device):
    batch = batch.to(device)
    num_nodes = {nt: batch[nt].num_nodes for nt in batch.node_types}
    h = model(batch.x_dict, batch.edge_index_dict, num_nodes)
    bs = batch[node_type].batch_size
    z = h[node_type][:bs]
    logits = heads.fraud_logits(z) if kind == "fraud" else heads.account_logits(z)
    return logits, batch[node_type].y[:bs].float()


def warm_up(model, heads, prepared, device):
    """PyG's lazy layers need one forward pass before parameters exist (optimiser, load_state_dict)."""
    g = prepared.graph
    idx = np.flatnonzero(g["transaction"].train_mask.numpy())[:8]
    if len(idx) == 0:
        idx = np.arange(min(8, g["transaction"].num_nodes))
    with torch.no_grad():
        _graph_logits(model, heads, next(iter(make_loader(prepared, "transaction", idx, 8, False))),
                      "transaction", "fraud", device)


def _cycle(loader):
    while True:
        yield from loader


def _tabular_batches(x, y, idx, batch_size, shuffle, rng):
    order = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(order), batch_size):
        sel = order[i:i + batch_size]
        yield x[sel], y[sel]


@torch.no_grad()
def predict_scores(prepared, model, heads, kind, idx, device, batch_size=None, variant="multitask"):
    idx = np.asarray(idx)
    if len(idx) == 0:
        return np.zeros(0)
    model.eval()
    heads.eval()
    out = []
    if variant == "no_graph":
        x = prepared.graph["transaction"].x
        for i in range(0, len(idx), 4096):
            z = model.encode_tabular(x[idx[i:i + 4096]].to(device))
            out.append(torch.sigmoid(heads.fraud_logits(z)).cpu().numpy())
        return np.concatenate(out)
    node_type = "transaction" if kind == "fraud" else "user"
    t = prepared.cfg["train"]
    bs = batch_size or (t["batch_fraud"] if kind == "fraud" else t["batch_account"])
    for batch in make_loader(prepared, node_type, idx, bs, shuffle=False):
        logits, _ = _graph_logits(model, heads, batch, node_type, kind, device)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def _validation_score(prepared, model, heads, spec, device, use_fraud, use_account):
    g, scores = prepared.graph, []
    if use_fraud and len(spec.fraud_val_idx):
        p = predict_scores(prepared, model, heads, "fraud", spec.fraud_val_idx, device, variant=spec.variant)
        scores.append(pr_auc(g["transaction"].y.numpy()[spec.fraud_val_idx], p))
    if use_account and len(spec.account_val_idx):
        p = predict_scores(prepared, model, heads, "account", spec.account_val_idx, device)
        scores.append(pr_auc(g["user"].y.numpy()[spec.account_val_idx], p))
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else -1.0


def train_variant(prepared, spec: RunSpec, device="cpu", checkpoint_path=None) -> TrainResult:
    cfg, g = prepared.cfg, prepared.graph
    t = cfg["train"]
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    use_fraud = spec.variant in FRAUD_VARIANTS
    use_account = spec.variant in ACCOUNT_VARIANTS
    w_f = t["w_fraud"] if use_fraud else 0.0
    w_a = t["w_account"] if use_account else 0.0
    alpha, gamma = t["focal_alpha"], t["focal_gamma"]
    graph_mode = spec.variant != "no_graph"
    model, heads = build_modules(prepared, device)
    if graph_mode:
        warm_up(model, heads, prepared, device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()),
                            lr=t["lr"], weight_decay=t["weight_decay"])

    fraud_loader = (make_loader(prepared, "transaction", spec.fraud_train_idx, t["batch_fraud"], True)
                    if use_fraud and graph_mode else None)
    account_loader = (make_loader(prepared, "user", spec.account_train_idx, t["batch_account"], True)
                      if use_account else None)
    x_tx, y_tx = g["transaction"].x, g["transaction"].y
    cap = t["max_steps_per_epoch"]
    best, best_epoch, bad, best_state, history = -np.inf, 0, 0, None, []

    for epoch in range(1, (spec.epochs or t["epochs"]) + 1):
        model.train()
        heads.train()
        losses = []
        if not graph_mode:
            for step, (xb, yb) in enumerate(_tabular_batches(x_tx, y_tx, spec.fraud_train_idx, t["batch_fraud"], True, rng)):
                if cap is not None and step >= cap:
                    break
                logits = heads.fraud_logits(model.encode_tabular(xb.to(device)))
                loss = w_f * focal_loss(logits, yb.float().to(device), alpha, gamma)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        else:
            main, kind, node_type, w_main = ((fraud_loader, "fraud", "transaction", w_f) if fraud_loader is not None
                                             else (account_loader, "account", "user", w_a))
            side = _cycle(account_loader) if fraud_loader is not None and account_loader is not None else None
            for step, batch in enumerate(main):
                if cap is not None and step >= cap:
                    break
                logits, y = _graph_logits(model, heads, batch, node_type, kind, device)
                loss = w_main * focal_loss(logits, y, alpha, gamma)
                if side is not None:
                    l2, y2 = _graph_logits(model, heads, next(side), "user", "account", device)
                    loss = loss + w_a * focal_loss(l2, y2, alpha, gamma)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan")}
        if spec.epochs is not None:  # fixed epochs: no validation, keep the last model
            history.append(entry)
            best_epoch = epoch
            continue
        entry["val_score"] = _validation_score(prepared, model, heads, spec, device, use_fraud, use_account)
        history.append(entry)
        if entry["val_score"] > best + 1e-9:
            best, best_epoch, bad = entry["val_score"], epoch, 0
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(heads.state_dict()))
        else:
            bad += 1
            if bad >= t["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state[0])
        heads.load_state_dict(best_state[1])
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "heads": heads.state_dict(), "variant": spec.variant,
                    "seed": spec.seed, "best_epoch": best_epoch, "config_hash": config_hash(cfg)}, checkpoint_path)
    return TrainResult(model, heads, best_epoch, float(best) if best_state is not None else float("nan"), history)


def load_checkpoint(path, prepared, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, heads = build_modules(prepared, device)
    warm_up(model, heads, prepared, device)
    model.load_state_dict(ckpt["model"])
    heads.load_state_dict(ckpt["heads"])
    return model.eval(), heads.eval(), {k: ckpt[k] for k in ("variant", "seed", "best_epoch", "config_hash")}


def main(argv=None):
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Train one GNN variant.")
    p.add_argument("--variant", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    variant = args.variant or cfg["train"]["variant"]
    device = resolve_device(args.device or cfg["train"]["device"])
    prepared = prepare(cfg)
    path = resolve_path("models") / f"{variant}_seed{args.seed}.pt"
    res = train_variant(prepared, default_spec(prepared, variant, args.seed), device, path)
    print(f"variant={variant} seed={args.seed} best_epoch={res.best_epoch} val_score={res.best_score:.4f} saved={path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `"$PY" -m pytest tests/test_train.py -q`
Expected: all pass. If `test_sampled_neighbours_never_come_from_the_future` fails, that is a leakage bug: stop, ledger it, and fix the sampler arguments before continuing. If `batch[nt].batch` is absent, apply the Task 2 spike's finding about the batch vector.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/train.py tests/test_train.py pipeline/metrics.py
git commit -m "feat: add the multi-task trainer with temporal sampling, variants and checkpoints" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Metrics: ranking, finance-shaped, calibration, bootstrap, weekly

**Files:**
- Create or extend: `pipeline/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces in `pipeline.metrics` (all take NumPy arrays; scores are probabilities):
  - `pr_auc(y, s) -> float`, `roc_auc(y, s) -> float` (NaN when `y` has a single class).
  - `recall_at_fpr(y_val, s_val, y_test, s_test, fpr=0.01) -> {"threshold", "recall", "realised_fpr"}` (threshold from validation negatives, applied to test).
  - `alert_budget(y, s, amount, budget) -> {"budget", "k", "recall", "precision", "amount_recall"}`.
  - `expected_cost(y_val, s_val, a_val, y_test, s_test, a_test, review_cost) -> {"threshold", "cost", "cost_flag_none", "cost_flag_all"}`.
  - `calibration(y, p, bins=10) -> ({"brier", "ece"}, DataFrame[bin_lo, bin_hi, mean_pred, frac_pos, count])`.
  - `bootstrap_metric(fn, y, s, n_boot=1000, seed=0, alpha=0.05) -> (lo, hi)`.
  - `weekly_table(times, y, s, start, n_boot=0, seed=0) -> DataFrame[week, n, n_pos, prevalence, pr_auc, pr_auc_lo, pr_auc_hi]`.

- [ ] **Step 1: Write the failing tests** (`tests/test_metrics.py`)

```python
import numpy as np
import pandas as pd
import pytest

from pipeline.metrics import (alert_budget, bootstrap_metric, calibration, expected_cost, pr_auc,
                              recall_at_fpr, roc_auc, weekly_table)

Y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
PERFECT = np.array([0.9, 0.8, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0])


def test_ranking_metrics_on_perfect_scores():
    assert pr_auc(Y, PERFECT) == pytest.approx(1.0) and roc_auc(Y, PERFECT) == pytest.approx(1.0)


def test_degenerate_inputs():
    assert np.isnan(pr_auc(np.zeros(5), np.random.rand(5))) and np.isnan(roc_auc(np.ones(5), np.random.rand(5)))
    assert pr_auc(Y, np.full(10, 0.5)) == pytest.approx(Y.mean())  # constant scores: precision = prevalence
    assert np.isnan(recall_at_fpr(Y, PERFECT, np.zeros(10), PERFECT)["recall"])  # test set with no positives


def test_recall_at_fpr_uses_the_validation_threshold():
    y_val = np.array([0] * 100 + [1] * 10)
    s_val = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.6, 1.0, 10)])
    r = recall_at_fpr(y_val, s_val, y_val, s_val, fpr=0.01)
    assert r["recall"] == 1.0 and r["realised_fpr"] <= 0.02
    assert 0.49 <= r["threshold"] <= 0.5


def test_alert_budget_counts_exactly():
    amount = np.array([100, 10, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    r = alert_budget(Y, PERFECT, amount, 0.2)
    assert r["k"] == 2 and r["recall"] == 1.0 and r["precision"] == 1.0 and r["amount_recall"] == 1.0
    r5 = alert_budget(Y, np.array([0.1, 0.9, 0.8, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0]), amount, 0.2)
    assert r5["recall"] == 0.5 and r5["amount_recall"] == pytest.approx(10 / 110)


def test_expected_cost_beats_flagging_nothing_with_a_good_model():
    amount = np.full(10, 50.0)
    c = expected_cost(Y, PERFECT, amount, Y, PERFECT, amount, review_cost=5.0)
    assert c["cost"] < c["cost_flag_none"] and c["cost_flag_none"] == 100.0 and c["cost_flag_all"] == 40.0


def test_calibration_of_a_calibrated_and_a_miscalibrated_score():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    good, table = calibration(y, p)
    bad, _ = calibration(y, np.clip(p * 0.3, 0, 1))
    assert good["ece"] < 0.03 < bad["ece"] and good["brier"] < bad["brier"]
    assert list(table.columns) == ["bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"]


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int)
    s = y * 0.4 + rng.random(500) * 0.6
    a = bootstrap_metric(pr_auc, y, s, n_boot=200, seed=0)
    assert a == bootstrap_metric(pr_auc, y, s, n_boot=200, seed=0)
    assert a[0] <= pr_auc(y, s) <= a[1]


def test_weekly_table_marks_weeks_without_positives():
    day = 86400
    times = np.concatenate([np.arange(0, 7 * day, day), np.arange(7 * day, 14 * day, day)])
    y = np.concatenate([[1, 0, 0, 0, 1, 0, 0], np.zeros(7)])
    s = np.random.default_rng(0).random(14)
    t = weekly_table(times, y, s, start=0)
    assert t["week"].tolist() == [0, 1] and t["n"].tolist() == [7, 7]
    assert not np.isnan(t.loc[0, "pr_auc"]) and np.isnan(t.loc[1, "pr_auc"]) and t.loc[1, "n_pos"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_metrics.py -q`
Expected: `ImportError` for the missing functions (if only `pr_auc`/`roc_auc` exist from Task 9).

- [ ] **Step 3: Implement** (`pipeline/metrics.py`)

```python
"""Ranking, finance-shaped, calibration and bootstrap metrics. Scores are probabilities in [0, 1]."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

NAN = float("nan")


def _single_class(y) -> bool:
    y = np.asarray(y)
    return y.sum() == 0 or y.sum() == len(y)


def pr_auc(y, s) -> float:
    return NAN if _single_class(y) else float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return NAN if _single_class(y) else float(roc_auc_score(y, s))


def recall_at_fpr(y_val, s_val, y_test, s_test, fpr=0.01) -> dict:
    y_val, s_val, y_test, s_test = map(np.asarray, (y_val, s_val, y_test, s_test))
    negatives = s_val[y_val == 0]
    if len(negatives) == 0 or (y_test == 1).sum() == 0 or (y_test == 0).sum() == 0:
        return {"threshold": NAN, "recall": NAN, "realised_fpr": NAN}
    thr = float(np.quantile(negatives, 1 - fpr))
    flagged = s_test > thr
    return {"threshold": thr, "recall": float(flagged[y_test == 1].mean()),
            "realised_fpr": float(flagged[y_test == 0].mean())}


def alert_budget(y, s, amount, budget) -> dict:
    y, s, amount = np.asarray(y), np.asarray(s), np.asarray(amount, dtype=float)
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-s, kind="stable")[:k]
    pos = y == 1
    total = amount[pos].sum()
    return {"budget": budget, "k": k,
            "recall": float(y[top].sum() / pos.sum()) if pos.any() else NAN,
            "precision": float(y[top].mean()),
            "amount_recall": float(amount[top][y[top] == 1].sum() / total) if total > 0 else NAN}


def _cost(thr, y, s, amount, review_cost) -> float:
    flagged = s > thr
    return float(amount[(y == 1) & ~flagged].sum() + review_cost * ((y == 0) & flagged).sum())


def expected_cost(y_val, s_val, a_val, y_test, s_test, a_test, review_cost) -> dict:
    y_val, s_val, a_val = np.asarray(y_val), np.asarray(s_val), np.asarray(a_val, dtype=float)
    y_test, s_test, a_test = np.asarray(y_test), np.asarray(s_test), np.asarray(a_test, dtype=float)
    candidates = np.append(np.unique(np.quantile(s_val, np.linspace(0, 1, 201))), s_val.max() + 1.0)
    costs = [_cost(c, y_val, s_val, a_val, review_cost) for c in candidates]
    thr = float(candidates[int(np.argmin(costs))])
    return {"threshold": thr, "cost": _cost(thr, y_test, s_test, a_test, review_cost),
            "cost_flag_none": float(a_test[y_test == 1].sum()),
            "cost_flag_all": float(review_cost * (y_test == 0).sum())}


def calibration(y, p, bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            conf, acc, n_b = float(p[m].mean()), float(y[m].mean()), int(m.sum())
            ece += n_b / len(y) * abs(conf - acc)
            rows.append((edges[b], edges[b + 1], conf, acc, n_b))
    table = pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"])
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}, table


def bootstrap_metric(fn, y, s, n_boot=1000, seed=0, alpha=0.05):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], s[i])
        if not np.isnan(v):
            values.append(v)
    if not values:
        return NAN, NAN
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def weekly_table(times, y, s, start, n_boot=0, seed=0) -> pd.DataFrame:
    times, y, s = np.asarray(times), np.asarray(y), np.asarray(s)
    week = np.floor((times - start) / (7 * 86400)).astype(int)
    rows = []
    for w in sorted(set(week[times >= start])):
        m = (week == w) & (times >= start)
        lo = hi = NAN
        if n_boot:
            lo, hi = bootstrap_metric(pr_auc, y[m], s[m], n_boot, seed)
        rows.append({"week": int(w), "n": int(m.sum()), "n_pos": int(y[m].sum()),
                     "prevalence": float(y[m].mean()), "pr_auc": pr_auc(y[m], s[m]),
                     "pr_auc_lo": lo, "pr_auc_hi": hi})
    return pd.DataFrame(rows, columns=["week", "n", "n_pos", "prevalence", "pr_auc", "pr_auc_lo", "pr_auc_hi"])
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_metrics.py -q` then `"$PY" -m pytest -q`
Expected: all pass.

```bash
git add pipeline/metrics.py tests/test_metrics.py
git commit -m "feat: add ranking, alert-budget, cost, calibration, bootstrap and weekly metrics" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Baselines: naive and history-aware trees, account baselines

**Files:**
- Create: `scripts/baseline.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `Prepared`, `pipeline.metrics.pr_auc`.
- Produces in `scripts.baseline`:
  - `fit_hgb(X, y, seed, max_iter=200) -> HistGradientBoostingClassifier`.
  - `fraud_baselines(prepared, seed, max_iter=200) -> dict`: keys `"naive_tabular"`, `"history_tabular"`; each `{"val": np.ndarray, "test": np.ndarray, "model": fitted}` with probabilities for the validation and test fraud rows (row order = `np.flatnonzero(graph["transaction"].val_mask)` / `test_mask`).
  - `user_aggregates(df, t_train) -> DataFrame` (per user, from rows up to `t_train`).
  - `account_baselines(prepared, seed, max_iter=200) -> dict`: keys `"hgb_user_aggregates"`, `"mean_fraud_score"`; each `{"val": np.ndarray, "test": np.ndarray}` over the eligible users of that split in `account.index` order.

- [ ] **Step 1: Write the failing tests** (`tests/test_baseline.py`)

```python
import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.metrics import pr_auc
from pipeline.prepare import prepare
from scripts.baseline import account_baselines, fraud_baselines, user_aggregates
from tests.fixtures import fixture_overrides


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_fraud_baselines_return_aligned_probabilities(prep):
    out = fraud_baselines(prep, seed=0, max_iter=30)
    val = np.flatnonzero(prep.graph["transaction"].val_mask.numpy())
    test = np.flatnonzero(prep.graph["transaction"].test_mask.numpy())
    assert set(out) == {"naive_tabular", "history_tabular"}
    for res in out.values():
        assert res["val"].shape == (len(val),) and res["test"].shape == (len(test),)
        assert ((res["test"] >= 0) & (res["test"] <= 1)).all()


def test_the_planted_fixture_signal_is_learnable(prep):
    out = fraud_baselines(prep, seed=0, max_iter=60)
    y = prep.df["isFraud"].to_numpy()[np.flatnonzero(prep.graph["transaction"].test_mask.numpy())]
    assert pr_auc(y, out["naive_tabular"]["test"]) > y.mean()  # better than chance on synthetic data only


def test_user_aggregates_use_only_history_up_to_t_train(prep):
    agg = user_aggregates(prep.df, prep.periods.t_train)
    early = prep.df[prep.df["TransactionDT"] <= prep.periods.t_train]
    assert agg["n_tx"].sum() == len(early)
    assert {"n_tx", "amt_mean", "n_devices", "n_emails", "tx_per_day"} <= set(agg.columns)


def test_account_baselines_cover_every_eligible_val_and_test_user(prep):
    out = account_baselines(prep, seed=0, max_iter=30)
    acc = prep.account
    for split in ("val", "test"):
        n = int(((acc["split"] == split) & acc["eligible"]).sum())
        for name in ("hgb_user_aggregates", "mean_fraud_score"):
            assert out[name][split].shape == (n,)
    assert np.isfinite(out["mean_fraud_score"]["test"]).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_baseline.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.baseline'`.

- [ ] **Step 3: Implement** (`scripts/baseline.py`)

```python
"""Tabular baselines on the same splits: naive, history-aware, user-aggregate and trivial account scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


def fit_hgb(X, y, seed, max_iter=200):
    return HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed).fit(X, y)


def _tx_idx(prepared, split):
    return np.flatnonzero(prepared.graph["transaction"][f"{split}_mask"].numpy())


def _history_matrix(prepared):
    return np.hstack([prepared.x, prepared.hist.to_numpy(np.float32)])


def fraud_baselines(prepared, seed, max_iter=200) -> dict:
    y = prepared.df["isFraud"].to_numpy()
    tr, va, te = _tx_idx(prepared, "train"), _tx_idx(prepared, "val"), _tx_idx(prepared, "test")
    out = {}
    for name, X in (("naive_tabular", prepared.x), ("history_tabular", _history_matrix(prepared))):
        model = fit_hgb(X[tr], y[tr], seed, max_iter)
        out[name] = {"val": model.predict_proba(X[va])[:, 1], "test": model.predict_proba(X[te])[:, 1],
                     "model": model}
    return out


def user_aggregates(df: pd.DataFrame, t_train: float) -> pd.DataFrame:
    h = df[df["TransactionDT"] <= t_train]
    g = h.groupby("user")
    agg = pd.DataFrame({
        "n_tx": g.size(), "amt_mean": g["TransactionAmt"].mean(), "amt_std": g["TransactionAmt"].std(),
        "amt_max": g["TransactionAmt"].max(), "n_devices": g["device"].nunique(),
        "n_emails": g["email"].nunique(),
        "span": g["TransactionDT"].max() - g["TransactionDT"].min(),
    })
    agg["tx_per_day"] = agg["n_tx"] / np.maximum(agg["span"] / 86400.0, 1.0)
    return agg


def _eligible_users(prepared):
    acc = prepared.account
    return {s: acc.index[(acc["split"] == s) & acc["eligible"]] for s in ("train", "val", "test")}


def _mean_fraud_score(prepared, seed, max_iter):
    """Per-user mean of out-of-fold fraud scores from the history-aware model (no in-sample label leakage)."""
    df, periods = prepared.df, prepared.periods
    y = df["isFraud"].to_numpy()
    X = _history_matrix(prepared)
    train_mask = prepared.graph["transaction"].train_mask.numpy()
    in_history = (df["TransactionDT"] <= periods.t_train).to_numpy()
    scores = np.full(len(df), np.nan)
    tr = np.flatnonzero(train_mask)
    groups = df["user"].to_numpy()[tr]
    for fit_i, pred_i in GroupKFold(n_splits=5).split(tr, groups=groups):
        model = fit_hgb(X[tr[fit_i]], y[tr[fit_i]], seed, max_iter)
        scores[tr[pred_i]] = model.predict_proba(X[tr[pred_i]])[:, 1]
    final = fit_hgb(X[tr], y[tr], seed, max_iter)
    other = np.flatnonzero(in_history & ~train_mask)  # train-period rows of held-out users
    scores[other] = final.predict_proba(X[other])[:, 1]
    return pd.Series(scores[in_history]).groupby(df["user"].to_numpy()[in_history]).mean()


def account_baselines(prepared, seed, max_iter=200) -> dict:
    users = _eligible_users(prepared)
    acc = prepared.account
    agg = user_aggregates(prepared.df, prepared.periods.t_train)
    model = fit_hgb(agg.loc[users["train"]].to_numpy(), acc.loc[users["train"], "label"].to_numpy(), seed, max_iter)
    hgb = {s: model.predict_proba(agg.loc[users[s]].to_numpy())[:, 1] for s in ("val", "test")}
    mean_score = _mean_fraud_score(prepared, seed, max_iter)
    trivial = {s: mean_score.reindex(users[s]).fillna(0.0).to_numpy() for s in ("val", "test")}
    return {"hgb_user_aggregates": hgb, "mean_fraud_score": trivial}
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_baseline.py -q`
Expected: all pass.

```bash
git add scripts/baseline.py tests/test_baseline.py
git commit -m "feat: add naive and history-aware tree baselines and account baselines" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Evaluation orchestration and result tables

**Files:**
- Create: `scripts/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Tasks 9-11 and `pipeline.config.config_hash`.
- Produces in `scripts.evaluate`:
  - `evaluate_fraud(model, seed, p_val, p_test, prepared) -> (row: dict, weekly: DataFrame, calibration_table: DataFrame)`; the row has keys `model, seed, head, config_hash, n_test, prevalence, pr_auc, pr_auc_lo, pr_auc_hi, roc_auc, recall_at_fpr, realised_fpr, recall@<b>, precision@<b>, amount_recall@<b>` (for each alert budget `b`, formatted with `:g`), `cost, cost_flag_none, cost_flag_all, brier, ece`.
  - `evaluate_account(model, seed, p_val, p_test, prepared) -> dict` (keys `model, seed, head, config_hash, n_test, prevalence, pr_auc, roc_auc`).
  - `evaluate_all(prepared, out_dir, variants=("multitask","fraud_only","account_only","no_graph"), seeds=None, device=None, log=print) -> dict[str, Path]` writing `results_fraud.csv`, `results_account.csv`, `results_weekly.csv`, `calibration.csv` into `out_dir` (rows for the baselines and every variant x seed; every row carries the same `config_hash`).

- [ ] **Step 1: Write the failing tests** (`tests/test_evaluate.py`)

```python
import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.evaluate import evaluate_account, evaluate_all, evaluate_fraud
from tests.fixtures import fixture_overrides


@pytest.fixture(scope="module")
def prep(raw_frames):
    return prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])


def test_fraud_row_has_every_metric_family(prep):
    va = np.flatnonzero(prep.graph["transaction"].val_mask.numpy())
    te = np.flatnonzero(prep.graph["transaction"].test_mask.numpy())
    y = prep.df["isFraud"].to_numpy()
    rng = np.random.default_rng(0)
    row, weekly, cal = evaluate_fraud("demo", 0, np.clip(y[va] * 0.5 + rng.random(len(va)) * 0.5, 0, 1),
                                      np.clip(y[te] * 0.5 + rng.random(len(te)) * 0.5, 0, 1), prep)
    for key in ("pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "recall_at_fpr", "realised_fpr", "recall@0.01",
                "precision@0.05", "amount_recall@0.005", "cost", "cost_flag_none", "brier", "ece", "config_hash"):
        assert key in row, key
    assert row["pr_auc_lo"] <= row["pr_auc"] <= row["pr_auc_hi"] and row["head"] == "fraud"
    assert {"week", "pr_auc"} <= set(weekly.columns) and len(cal) > 0


def test_account_row(prep):
    users = prep.account
    n_val = int(((users["split"] == "val") & users["eligible"]).sum())
    n_test = int(((users["split"] == "test") & users["eligible"]).sum())
    row = evaluate_account("demo", 0, np.random.rand(n_val), np.random.rand(n_test), prep)
    assert row["head"] == "account" and row["n_test"] == n_test and 0 <= row["prevalence"] <= 1


def test_results_carry_one_config_hash(prep, tmp_path):
    paths = evaluate_all(prep, tmp_path, variants=("multitask", "no_graph"), seeds=[0], device="cpu", log=lambda *_: None)
    fraud = pd.read_csv(paths["fraud"])
    account = pd.read_csv(paths["account"])
    assert fraud["config_hash"].nunique() == 1 and account["config_hash"].nunique() == 1
    assert fraud["config_hash"].iloc[0] == account["config_hash"].iloc[0]
    assert {"naive_tabular", "history_tabular", "multitask", "no_graph"} <= set(fraud["model"])
    assert {"hgb_user_aggregates", "mean_fraud_score", "multitask"} <= set(account["model"])
    assert fraud["pr_auc"].notna().all()
    assert (tmp_path / "results_weekly.csv").exists() and (tmp_path / "calibration.csv").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_evaluate.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.evaluate'`.

- [ ] **Step 3: Implement** (`scripts/evaluate.py`)

```python
"""Train every variant and baseline, evaluate on the test period and write the result tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.metrics import (alert_budget, bootstrap_metric, calibration, expected_cost, pr_auc,
                              recall_at_fpr, roc_auc, weekly_table)
from scripts.baseline import account_baselines, fraud_baselines
from scripts.train import ACCOUNT_VARIANTS, FRAUD_VARIANTS, default_spec, predict_scores, train_variant


def _fraud_arrays(prepared):
    g, df = prepared.graph, prepared.df
    va = np.flatnonzero(g["transaction"].val_mask.numpy())
    te = np.flatnonzero(g["transaction"].test_mask.numpy())
    y, amount, times = df["isFraud"].to_numpy(), df["TransactionAmt"].to_numpy(float), df["TransactionDT"].to_numpy()
    return {"y_val": y[va], "a_val": amount[va], "y_test": y[te], "a_test": amount[te], "t_test": times[te]}


def evaluate_fraud(model, seed, p_val, p_test, prepared):
    cfg = prepared.cfg
    e, a = cfg["evaluation"], _fraud_arrays(prepared)
    y_val, y_test = a["y_val"], a["y_test"]
    lo, hi = bootstrap_metric(pr_auc, y_test, p_test, e["bootstrap_samples"], seed)
    row = {"model": model, "seed": seed, "head": "fraud", "config_hash": config_hash(cfg),
           "n_test": int(len(y_test)), "prevalence": float(y_test.mean()),
           "pr_auc": pr_auc(y_test, p_test), "pr_auc_lo": lo, "pr_auc_hi": hi, "roc_auc": roc_auc(y_test, p_test)}
    r = recall_at_fpr(y_val, p_val, y_test, p_test, e["fpr_target"])
    row.update(recall_at_fpr=r["recall"], realised_fpr=r["realised_fpr"])
    for b in e["alert_budgets"]:
        ab = alert_budget(y_test, p_test, a["a_test"], b)
        row.update({f"recall@{b:g}": ab["recall"], f"precision@{b:g}": ab["precision"],
                    f"amount_recall@{b:g}": ab["amount_recall"]})
    c = expected_cost(y_val, p_val, a["a_val"], y_test, p_test, a["a_test"], e["review_cost"])
    row.update(cost=c["cost"], cost_flag_none=c["cost_flag_none"], cost_flag_all=c["cost_flag_all"])
    cal, table = calibration(y_test, p_test)
    row.update(cal)
    weekly = weekly_table(a["t_test"], y_test, p_test, start=prepared.periods.test_start)
    weekly.insert(0, "model", model)
    weekly.insert(1, "seed", seed)
    table.insert(0, "model", model)
    table.insert(1, "seed", seed)
    return row, weekly, table


def evaluate_account(model, seed, p_val, p_test, prepared):
    acc = prepared.account
    y_test = acc.loc[(acc["split"] == "test") & acc["eligible"], "label"].to_numpy()
    return {"model": model, "seed": seed, "head": "account", "config_hash": config_hash(prepared.cfg),
            "n_test": int(len(y_test)), "prevalence": float(y_test.mean()),
            "pr_auc": pr_auc(y_test, p_test), "roc_auc": roc_auc(y_test, p_test)}


def evaluate_all(prepared, out_dir, variants=("multitask", "fraud_only", "account_only", "no_graph"),
                 seeds=None, device=None, log=print) -> dict:
    cfg = prepared.cfg
    seeds = cfg["train"]["seeds"] if seeds is None else seeds
    device = resolve_device(device or cfg["train"]["device"])
    fraud_rows, account_rows, weekly, cals = [], [], [], []

    def add_fraud(name, seed, p_val, p_test):
        row, wk, cal = evaluate_fraud(name, seed, p_val, p_test, prepared)
        fraud_rows.append(row), weekly.append(wk), cals.append(cal)

    users = {s: prepared.account.index[(prepared.account["split"] == s) & prepared.account["eligible"]]
             for s in ("val", "test")}
    for seed in seeds:
        log(f"baselines seed={seed}")
        for name, res in fraud_baselines(prepared, seed).items():
            add_fraud(name, seed, res["val"], res["test"])
        for name, res in account_baselines(prepared, seed).items():
            account_rows.append(evaluate_account(name, seed, res["val"], res["test"], prepared))
        for variant in variants:
            log(f"training {variant} seed={seed}")
            spec = default_spec(prepared, variant, seed)
            res = train_variant(prepared, spec, device)
            if variant in FRAUD_VARIANTS:
                va = np.flatnonzero(prepared.graph["transaction"].val_mask.numpy())
                te = np.flatnonzero(prepared.graph["transaction"].test_mask.numpy())
                add_fraud(variant, seed,
                          predict_scores(prepared, res.model, res.heads, "fraud", va, device, variant=variant),
                          predict_scores(prepared, res.model, res.heads, "fraud", te, device, variant=variant))
            if variant in ACCOUNT_VARIANTS:
                index = prepared.entity_index["user"]
                pos = {name: i for i, name in enumerate(index)}
                p = {s: predict_scores(prepared, res.model, res.heads, "account",
                                       np.array([pos[u] for u in users[s]]), device) for s in ("val", "test")}
                account_rows.append(evaluate_account(variant, seed, p["val"], p["test"], prepared))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"fraud": out_dir / "results_fraud.csv", "account": out_dir / "results_account.csv",
             "weekly": out_dir / "results_weekly.csv", "calibration": out_dir / "calibration.csv"}
    pd.DataFrame(fraud_rows).to_csv(paths["fraud"], index=False)
    pd.DataFrame(account_rows).to_csv(paths["account"], index=False)
    pd.concat(weekly, ignore_index=True).to_csv(paths["weekly"], index=False)
    pd.concat(cals, ignore_index=True).to_csv(paths["calibration"], index=False)
    return paths


def main(argv=None):
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Evaluate every variant and baseline.")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="plots")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args = p.parse_args(argv)
    prepared = prepare(load_config(args.config))
    paths = evaluate_all(prepared, resolve_path(args.out), seeds=args.seeds)
    print(pd.read_csv(paths["fraud"])[["model", "seed", "pr_auc", "recall@0.01", "amount_recall@0.01"]].to_string(index=False))


if __name__ == "__main__":
    main()
```
Note: `evaluate_all` orders the account-user index by `account.index`, which `prepare` builds from `user_split` (sorted users); `entity_index["user"]` is in order of first appearance. The code maps names through `pos` so the two orders never need to agree.

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_evaluate.py -q`
Expected: all pass (about a minute on CPU).

```bash
git add scripts/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation orchestration and result tables with a config hash" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Drift and retraining study

**Files:**
- Create: `scripts/drift_study.py`, `tests/test_drift_study.py`

**Interfaces:**
- Consumes: Tasks 9-11 and `pipeline.metrics.weekly_table`.
- Produces in `scripts.drift_study`:
  - `retrain_cutoffs(periods, max_time, every_days) -> list[float]` (`val_start + k * every_days * 86400` for k = 1, 2, ... while below `max_time`).
  - `combine_by_cutoff(times, cutoffs, score_sets) -> np.ndarray` (`score_sets[0]` is the base model's scores for every row; `score_sets[k]` the scores of the model retrained at `cutoffs[k-1]`; a row with `time > cutoffs[k-1]` takes `score_sets[k]`).
  - `drift_study(prepared, seed=0, kinds=("history_tabular","gnn_fraud_only"), device=None, log=print) -> DataFrame` with columns `kind, schedule ("static" or "retrain"), week, n, n_pos, prevalence, pr_auc, pr_auc_lo, pr_auc_hi`, over validation and test rows only.
  - `summarize_drift(table) -> DataFrame[kind, schedule, mean_pr_auc, slope_per_week]` (slope by least squares over weeks with a defined PR-AUC).

- [ ] **Step 1: Write the failing tests** (`tests/test_drift_study.py`)

```python
import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.prepare import prepare
from pipeline.splits import DAY, Periods
from scripts.drift_study import combine_by_cutoff, drift_study, retrain_cutoffs, summarize_drift
from tests.fixtures import fixture_overrides, make_fixture


def test_retrain_cutoffs_step_from_val_start_until_the_end():
    p = Periods(t_train=0.0, val_start=10 * DAY, val_end=50 * DAY, test_start=60 * DAY, embargo=10 * DAY)
    assert retrain_cutoffs(p, max_time=100 * DAY, every_days=28) == [38 * DAY, 66 * DAY, 94 * DAY]
    assert retrain_cutoffs(p, max_time=30 * DAY, every_days=28) == []


def test_combine_by_cutoff_switches_models_at_each_cutoff():
    times = np.array([1, 5, 11, 15, 25])
    base, m1, m2 = np.full(5, 0.0), np.full(5, 1.0), np.full(5, 2.0)
    assert combine_by_cutoff(times, [10, 20], [base, m1, m2]).tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]
    assert combine_by_cutoff(times, [], [base]).tolist() == [0.0] * 5


def test_summarize_drift_reports_a_negative_slope_for_a_decaying_curve():
    table = pd.DataFrame({"kind": "k", "schedule": "static", "week": [0, 1, 2, 3],
                          "pr_auc": [0.8, 0.6, 0.4, np.nan]})
    s = summarize_drift(table).iloc[0]
    assert s["slope_per_week"] == pytest.approx(-0.2) and s["mean_pr_auc"] == pytest.approx(0.6)


def test_a_planted_concept_flip_shows_decay_that_retraining_repairs():
    tx, _ = make_fixture(n_users=800, n_tx=20000, days=240, seed=1)
    flip = tx["TransactionDT"].quantile(0.75)                       # inside the validation period
    late = (tx["TransactionDT"] > flip).to_numpy()
    low_c1 = (tx["C1"] <= tx.loc[late, "C1"].quantile(0.3)).to_numpy()
    rng = np.random.default_rng(0)
    y = tx["isFraud"].to_numpy().copy()
    y[late] = (low_c1[late] & (rng.random(int(late.sum())) < 0.4)).astype(int)   # after the flip LOW C1 means fraud
    tx["isFraud"] = y
    overrides = fixture_overrides()
    overrides["drift"] = {"retrain_every_days": 14, "seed": 0}
    prep = prepare(load_config(overrides=overrides), df=tx)
    table = drift_study(prep, seed=0, kinds=("history_tabular",), log=lambda *_: None)
    flip_week = (flip - prep.periods.val_start) / (7 * DAY)

    def mean_pr(schedule, keep):
        g = table[(table["schedule"] == schedule) & keep(table["week"])]
        return g["pr_auc"].mean()

    before = mean_pr("static", lambda w: w < flip_week - 0.5)
    late_static = mean_pr("static", lambda w: w >= flip_week + 3)
    late_retrain = mean_pr("retrain", lambda w: w >= flip_week + 3)
    assert late_static < before and late_retrain > late_static
    # If this fails, strengthen the planted flip (more rows, a cleaner rule) before touching drift_study:
    # the test exists to prove the study can see decay and recovery.


def test_history_tabular_drift_study_on_the_fixture(raw_frames):
    prep = prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])
    table = drift_study(prep, seed=0, kinds=("history_tabular",), log=lambda *_: None)
    assert {"static", "retrain"} == set(table["schedule"]) and set(table["kind"]) == {"history_tabular"}
    for _, g in table.groupby("schedule"):
        assert g["week"].is_monotonic_increasing
    assert table["n"].sum() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_drift_study.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.drift_study'`.

- [ ] **Step 3: Implement** (`scripts/drift_study.py`)

```python
"""How does performance decay after training, and does periodic retraining help?"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline.config import load_config, resolve_device, resolve_path
from pipeline.metrics import weekly_table
from pipeline.splits import DAY, period_of
from scripts.baseline import _history_matrix, fit_hgb
from scripts.train import RunSpec, default_spec, predict_scores, train_variant


def retrain_cutoffs(periods, max_time, every_days):
    cuts, k = [], 1
    while periods.val_start + k * every_days * DAY < max_time:
        cuts.append(periods.val_start + k * every_days * DAY)
        k += 1
    return cuts


def combine_by_cutoff(times, cutoffs, score_sets):
    out = np.asarray(score_sets[0], dtype=float).copy()
    for k, cutoff in enumerate(cutoffs, start=1):
        out = np.where(np.asarray(times) > cutoff, score_sets[k], out)
    return out


def summarize_drift(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, schedule), g in table.groupby(["kind", "schedule"]):
        g = g.dropna(subset=["pr_auc"])
        slope = float(np.polyfit(g["week"], g["pr_auc"], 1)[0]) if len(g) >= 2 else float("nan")
        rows.append({"kind": kind, "schedule": schedule,
                     "mean_pr_auc": float(g["pr_auc"].mean()) if len(g) else float("nan"), "slope_per_week": slope})
    return pd.DataFrame(rows)


def _train_rows(prepared, cutoff):
    """Rows whose labels would be known at `cutoff`: older than the embargo, from train users only."""
    df, p = prepared.df, prepared.periods
    known = (df["TransactionDT"] <= cutoff - p.embargo).to_numpy()
    return np.flatnonzero(known & (df["user"].map(prepared.user_split) == "train").to_numpy())


def drift_study(prepared, seed=0, kinds=("history_tabular", "gnn_fraud_only"), device=None, log=print):
    cfg, df, p = prepared.cfg, prepared.df, prepared.periods
    device = resolve_device(device or cfg["train"]["device"])
    times, y = df["TransactionDT"].to_numpy(), df["isFraud"].to_numpy()
    eval_rows = np.flatnonzero(np.isin(period_of(times, p), ["val", "test"]))
    cutoffs = retrain_cutoffs(p, times.max(), cfg["drift"]["retrain_every_days"])
    n_boot = cfg["evaluation"]["bootstrap_samples"]
    out = []
    for kind in kinds:
        log(f"drift study: {kind}, {len(cutoffs)} retrain point(s)")
        scores = []
        if kind == "history_tabular":
            X = _history_matrix(prepared)
            base_rows = np.flatnonzero(prepared.graph["transaction"].train_mask.numpy())
            scores.append(fit_hgb(X[base_rows], y[base_rows], seed).predict_proba(X[eval_rows])[:, 1])
            for c in cutoffs:
                rows = _train_rows(prepared, c)
                scores.append(fit_hgb(X[rows], y[rows], seed).predict_proba(X[eval_rows])[:, 1])
        elif kind == "gnn_fraud_only":
            base = train_variant(prepared, default_spec(prepared, "fraud_only", seed), device)
            scores.append(predict_scores(prepared, base.model, base.heads, "fraud", eval_rows, device))
            empty = np.zeros(0, dtype=int)
            for c in cutoffs:
                spec = RunSpec("fraud_only", seed, _train_rows(prepared, c), empty, empty, empty,
                               epochs=max(base.best_epoch, 1))
                m = train_variant(prepared, spec, device)
                scores.append(predict_scores(prepared, m.model, m.heads, "fraud", eval_rows, device))
        else:
            raise ValueError(f"unknown drift kind {kind!r}")
        for schedule, s in (("static", scores[0]), ("retrain", combine_by_cutoff(times[eval_rows], cutoffs, scores))):
            wk = weekly_table(times[eval_rows], y[eval_rows], s, start=p.val_start, n_boot=n_boot, seed=seed)
            wk.insert(0, "kind", kind)
            wk.insert(1, "schedule", schedule)
            out.append(wk)
    return pd.concat(out, ignore_index=True)


def main(argv=None):
    from pipeline.prepare import prepare

    ap = argparse.ArgumentParser(description="Drift and retraining study.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="plots")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    prepared = prepare(cfg)
    table = drift_study(prepared, seed=cfg["drift"]["seed"] if args.seed is None else args.seed)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "drift_weekly.csv", index=False)
    summarize_drift(table).to_csv(out / "drift_summary.csv", index=False)
    print(summarize_drift(table).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_drift_study.py -q`
Expected: all pass.

```bash
git add scripts/drift_study.py tests/test_drift_study.py
git commit -m "feat: add the drift and retraining study" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Explainability with stability and fidelity checks

**Files:**
- Create: `scripts/explain.py`, `tests/test_explain.py`

**Interfaces:**
- Consumes: Tasks 8-9 (`load_checkpoint`, `make_loader`), `Prepared`, config `explain.*`. The mode (`hetero` or `occlusion`) comes from the Task 2 spike via `explain.mode`.
- Produces in `scripts.explain`:
  - `FraudLogitWrapper(model, heads, num_nodes_dict)`: `forward(x_dict, edge_index_dict) -> Tensor[N_tx]` of fraud logits (what `Explainer` needs).
  - `top_features(importance, names, k) -> list[tuple[str, float]]` (descending).
  - `topk_overlap(importances, k) -> float` (mean pairwise Jaccard of the top-k index sets; 1.0 when identical).
  - `occlusion_importance(wrapper, x_dict, edge_index_dict, row=0) -> np.ndarray` (per-feature absolute logit change when that feature of `row` is zeroed to its train mean).
  - `deletion_fidelity(wrapper, x_dict, edge_index_dict, importance, k, row=0, seed=0) -> dict` with `drop_top_k` (logit drop after zeroing the top-k features) and `drop_random_k` (mean over 20 random sets of k).
  - `explain_transaction(prepared, model, heads, tx_row, mode, seed, device) -> dict` with `importance` (length = number of features), `logit`, and for `hetero` mode also `explainer`/`explanation`.
  - `top_edges(out, k, entity_index=None) -> list[(edge_type, src, dst, importance)]` (hetero explanations only), `pyg_fidelity(out) -> dict | None` (`fidelity_plus`, `fidelity_minus`, `characterization`; `None` with a warning when PyG cannot compute them), `explain_user(prepared, model, heads, user_node, seed, device, k=5) -> list` (influential edges behind one user's account-risk score; hetero mode).
  - `explain_report(prepared, checkpoint, out_dir, device=None) -> dict` writing `reports/explanations.md` and `reports/feature_importance.png`; returns `stability`, `fidelity_top`, `fidelity_random`, `n_fidelity` (a sample of `explain.fidelity_sample` test transactions, 200 by default) and, in hetero mode, `pyg_fidelity_plus`, `pyg_fidelity_minus`, `pyg_characterization` (else `None`).

- [ ] **Step 1: Write the failing tests** (`tests/test_explain.py`)

```python
import numpy as np
import pytest
import torch

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.explain import (deletion_fidelity, explain_transaction, explain_user, occlusion_importance,
                             pyg_fidelity, top_edges, top_features, topk_overlap)
from scripts.train import default_spec, train_variant
from tests.fixtures import fixture_overrides


class Linear(torch.nn.Module):
    """Toy 'model': logit of node i is x[i] @ w, so occlusion importance is exactly |x_j * w_j|."""

    def __init__(self, w):
        super().__init__()
        self.w = torch.tensor(w, dtype=torch.float32)

    def forward(self, x_dict, edge_index_dict):
        return x_dict["transaction"] @ self.w


def test_occlusion_importance_matches_a_linear_model_exactly():
    x = {"transaction": torch.tensor([[1.0, 2.0, 3.0]])}
    imp = occlusion_importance(Linear([0.5, -1.0, 0.1]), x, {})
    assert np.allclose(imp, [0.5, 2.0, 0.3])


def test_top_features_orders_by_importance():
    assert top_features(np.array([0.1, 0.9, 0.5]), ["a", "b", "c"], 2) == [("b", 0.9), ("c", 0.5)]


def test_topk_overlap_is_one_for_identical_and_lower_for_different_rankings():
    a = np.array([5.0, 4.0, 3.0, 0.1, 0.0])
    b = np.array([0.0, 0.1, 3.0, 4.0, 5.0])
    assert topk_overlap([a, a, a], k=2) == 1.0 and topk_overlap([a, b], k=2) == 0.0


def test_deleting_the_top_features_hurts_more_than_deleting_random_ones():
    x = {"transaction": torch.tensor([[3.0, 2.0, 0.1, 0.1, 0.1, 0.1]])}
    model = Linear([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    imp = occlusion_importance(model, x, {})
    f = deletion_fidelity(model, x, {}, imp, k=2)
    assert f["drop_top_k"] > f["drop_random_k"]


@pytest.fixture(scope="module")
def trained(raw_frames):
    prep = prepare(load_config(overrides=fixture_overrides()), df=raw_frames[0])
    return prep, train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu")


@pytest.mark.slow
def test_explaining_a_transaction_returns_one_importance_per_feature(trained):
    prep, res = trained
    mode = prep.cfg["explain"]["mode"]
    row = int(np.flatnonzero(prep.graph["transaction"].test_mask.numpy())[0])
    out = explain_transaction(prep, res.model, res.heads, row, mode, seed=0, device="cpu")
    assert out["importance"].shape == (len(prep.transformer.feature_names),)
    assert np.isfinite(out["importance"]).all()


@pytest.mark.slow
def test_hetero_mode_reports_edges_pyg_fidelity_and_a_user_explanation(trained):
    prep, res = trained
    if prep.cfg["explain"]["mode"] != "hetero":
        pytest.skip("the spike chose occlusion mode")
    row = int(np.flatnonzero(prep.graph["transaction"].test_mask.numpy())[0])
    out = explain_transaction(prep, res.model, res.heads, row, "hetero", seed=0, device="cpu")
    edges = top_edges(out, 5, prep.entity_index)
    assert 0 < len(edges) <= 5 and all(len(e) == 4 and np.isfinite(e[3]) for e in edges)
    fid = pyg_fidelity(out)
    assert fid is None or (np.isfinite(fid["fidelity_plus"]) and np.isfinite(fid["fidelity_minus"]))
    user = int(np.flatnonzero(prep.graph["user"].test_mask.numpy())[0])
    assert len(explain_user(prep, res.model, res.heads, user, seed=0, device="cpu")) > 0


def test_occlusion_mode_always_works_on_the_trained_model(trained):
    prep, res = trained
    row = int(np.flatnonzero(prep.graph["transaction"].test_mask.numpy())[0])
    out = explain_transaction(prep, res.model, res.heads, row, "occlusion", seed=0, device="cpu")
    assert out["importance"].shape == (len(prep.transformer.feature_names),) and np.isfinite(out["logit"])
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_explain.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.explain'`.

- [ ] **Step 3: Implement** (`scripts/explain.py`)

```python
"""GNNExplainer (or an occlusion fallback) with stability and fidelity checks."""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pipeline.config import resolve_device, resolve_path
from scripts.train import load_checkpoint, make_loader, predict_scores


class FraudLogitWrapper(torch.nn.Module):
    def __init__(self, model, heads, num_nodes_dict):
        super().__init__()
        self.model, self.heads, self.num_nodes = model, heads, num_nodes_dict

    def forward(self, x_dict, edge_index_dict):
        h = self.model(x_dict, edge_index_dict, self.num_nodes)
        return self.heads.fraud_logits(h["transaction"])


def top_features(importance, names, k):
    order = np.argsort(-np.asarray(importance), kind="stable")[:k]
    return [(names[i], float(importance[i])) for i in order]


def topk_overlap(importances, k) -> float:
    sets = [set(np.argsort(-np.asarray(v), kind="stable")[:k].tolist()) for v in importances]
    pairs = list(itertools.combinations(sets, 2))
    if not pairs:
        return 1.0
    return float(np.mean([len(a & b) / len(a | b) for a, b in pairs]))


@torch.no_grad()
def occlusion_importance(wrapper, x_dict, edge_index_dict, row=0):
    wrapper.eval()
    base = wrapper(x_dict, edge_index_dict)[row]
    x = x_dict["transaction"]
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        x2 = x.clone()
        x2[row, j] = 0.0  # features are standardised on the train period, so 0 is the train mean
        out[j] = float((base - wrapper({**x_dict, "transaction": x2}, edge_index_dict)[row]).abs())
    return out


@torch.no_grad()
def deletion_fidelity(wrapper, x_dict, edge_index_dict, importance, k, row=0, seed=0):
    wrapper.eval()
    x = x_dict["transaction"]
    base = float(wrapper(x_dict, edge_index_dict)[row])

    def drop(cols):
        x2 = x.clone()
        x2[row, list(cols)] = 0.0
        return abs(base - float(wrapper({**x_dict, "transaction": x2}, edge_index_dict)[row]))

    top = np.argsort(-np.asarray(importance), kind="stable")[:k]
    rng = np.random.default_rng(seed)
    rand = [drop(rng.choice(x.shape[1], size=k, replace=False)) for _ in range(20)]
    return {"drop_top_k": drop(top), "drop_random_k": float(np.mean(rand))}


def _seed_batch(prepared, tx_row, device):
    return next(iter(make_loader(prepared, "transaction", np.array([tx_row]), 1, False))).to(device)


class AccountLogitWrapper(torch.nn.Module):
    def __init__(self, model, heads, num_nodes_dict):
        super().__init__()
        self.model, self.heads, self.num_nodes = model, heads, num_nodes_dict

    def forward(self, x_dict, edge_index_dict):
        return self.heads.account_logits(self.model(x_dict, edge_index_dict, self.num_nodes)["user"])


def _hetero_explainer(wrapper, epochs):
    from torch_geometric.explain import Explainer, GNNExplainer

    return Explainer(model=wrapper, algorithm=GNNExplainer(epochs=epochs), explanation_type="model",
                     node_mask_type="attributes", edge_mask_type="object",
                     model_config=dict(mode="binary_classification", task_level="node", return_type="raw"))


def explain_transaction(prepared, model, heads, tx_row, mode, seed, device):
    batch = _seed_batch(prepared, tx_row, device)
    wrapper = FraudLogitWrapper(model, heads, {nt: batch[nt].num_nodes for nt in batch.node_types}).to(device).eval()
    x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
    with torch.no_grad():
        logit = float(wrapper(x_dict, edge_index_dict)[0])
    common = {"logit": logit, "wrapper": wrapper, "x_dict": x_dict, "edge_index_dict": edge_index_dict, "batch": batch}
    if mode == "occlusion":
        return {"importance": occlusion_importance(wrapper, x_dict, edge_index_dict), **common}
    torch.manual_seed(seed)
    explainer = _hetero_explainer(wrapper, prepared.cfg["explain"]["epochs"])
    expl = explainer(x_dict, edge_index_dict, index=torch.tensor([0], device=device))
    importance = expl["transaction"].node_mask[0].detach().cpu().numpy()
    return {"importance": importance, "explainer": explainer, "explanation": expl, **common}


def top_edges(out, k, entity_index=None):
    """The k most influential sampled edges of a hetero explanation: [(edge_type, src, dst, importance)].

    Transactions are labelled by row; entity nodes by key when `entity_index` is given, else by node id."""
    expl, batch = out["explanation"], out["batch"]

    def label(node_type, node_id):
        node_id = int(node_id)
        return node_id if node_type == "transaction" or entity_index is None else entity_index[node_type][node_id]

    rows = []
    for et in expl.edge_types:
        store = expl[et]
        if "edge_mask" not in store:
            continue
        src_t, _, dst_t = et
        ei = batch[et].edge_index
        for j, value in enumerate(store.edge_mask.detach().cpu().tolist()):
            rows.append((et, label(src_t, batch[src_t].n_id[ei[0, j]]), label(dst_t, batch[dst_t].n_id[ei[1, j]]), float(value)))
    return sorted(rows, key=lambda r: -r[3])[:k]


def pyg_fidelity(out):
    """PyG's fidelity+/- and characterization score for one hetero explanation, or None if PyG cannot compute them."""
    try:
        from torch_geometric.explain.metric import characterization_score, fidelity

        plus, minus = fidelity(out["explainer"], out["explanation"])
        return {"fidelity_plus": float(plus), "fidelity_minus": float(minus),
                "characterization": float(characterization_score(plus, minus))}
    except Exception as exc:                              # unsupported for this model or graph: say so, do not hide it
        warnings.warn(f"PyG fidelity unavailable: {exc!r}")
        return None


def explain_user(prepared, model, heads, user_node, seed, device, k=5):
    """Most influential sampled edges behind one user's account-risk score (hetero mode; users have no features)."""
    batch = next(iter(make_loader(prepared, "user", np.array([user_node]), 1, False))).to(device)
    wrapper = AccountLogitWrapper(model, heads, {nt: batch[nt].num_nodes for nt in batch.node_types}).to(device).eval()
    torch.manual_seed(seed)
    explainer = _hetero_explainer(wrapper, prepared.cfg["explain"]["epochs"])
    expl = explainer(batch.x_dict, batch.edge_index_dict, index=torch.tensor([0], device=device))
    return top_edges({"explanation": expl, "batch": batch}, k, prepared.entity_index)


def explain_report(prepared, checkpoint, out_dir, device=None):
    cfg = prepared.cfg
    e = cfg["explain"]
    hetero = e["mode"] == "hetero"
    device = resolve_device(device or cfg["train"]["device"])
    model, heads, meta = load_checkpoint(checkpoint, prepared, device)
    names = prepared.transformer.feature_names
    te = np.flatnonzero(prepared.graph["transaction"].test_mask.numpy())
    p = predict_scores(prepared, model, heads, "fraud", te, device)
    y = prepared.df["isFraud"].to_numpy()[te]
    order = np.argsort(-p, kind="stable")
    half = e["n_explained"] // 2
    picks = ([int(te[i]) for i in order if y[i] == 1][:half]
             + [int(te[i]) for i in order if y[i] == 0][: e["n_explained"] - half])
    sample = np.random.default_rng(0).choice(te, size=min(e["fidelity_sample"], len(te)), replace=False)
    stability, rows, mean_imp = [], [], np.zeros(len(names))
    for row in picks:
        outs = [explain_transaction(prepared, model, heads, row, e["mode"], s, device) for s in range(e["stability_seeds"])]
        runs = [o["importance"] for o in outs]
        stability.append(topk_overlap(runs, e["top_k"]))
        mean_imp += np.mean(runs, axis=0)
        edges = top_edges(outs[0], 5, prepared.entity_index) if hetero else []
        rows.append((row, top_features(np.mean(runs, axis=0), names, e["top_k"]), edges))
    fid_top, fid_rand, pyg = [], [], []
    for row in sample:
        occ = explain_transaction(prepared, model, heads, int(row), "occlusion", 0, device)
        f = deletion_fidelity(occ["wrapper"], occ["x_dict"], occ["edge_index_dict"], occ["importance"], e["top_k"])
        fid_top.append(f["drop_top_k"])
        fid_rand.append(f["drop_random_k"])
        if hetero:
            r = pyg_fidelity(explain_transaction(prepared, model, heads, int(row), "hetero", 0, device))
            if r is not None:
                pyg.append(r)
    user_rows = []
    if hetero:
        index = prepared.entity_index["user"]
        for u in np.flatnonzero(prepared.graph["user"].test_mask.numpy())[:3]:
            user_rows.append((index[int(u)], explain_user(prepared, model, heads, int(u), 0, device)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top = np.argsort(-mean_imp)[: e["top_k"]]
    plt.figure(figsize=(7, 4))
    plt.barh([names[i] for i in top][::-1], mean_imp[top][::-1] / max(len(picks), 1))
    plt.xlabel("mean importance"), plt.title("Top features across the explained transactions")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=130)
    plt.close()
    mean_pyg = ({k: float(np.mean([r[k] for r in pyg])) for k in ("fidelity_plus", "fidelity_minus", "characterization")}
                if pyg else None)
    lines = ["# Explanations", "",
             f"Mode: `{e['mode']}`. Features are anonymised, so names are column names, not business meaning. "
             "Explanations describe the model, not causes.", "",
             f"- Stability (mean top-{e['top_k']} overlap across {e['stability_seeds']} runs): {np.mean(stability):.2f}",
             f"- Deletion fidelity (occlusion, {len(sample)} transactions): dropping the top-{e['top_k']} features changes the "
             f"logit by {np.mean(fid_top):.3f} on average, against {np.mean(fid_rand):.3f} for random features"]
    if mean_pyg:
        lines.append(f"- PyG fidelity+ {mean_pyg['fidelity_plus']:.3f}, fidelity- {mean_pyg['fidelity_minus']:.3f}, "
                     f"characterization {mean_pyg['characterization']:.3f} ({len(pyg)} transactions)")
    elif hetero:
        lines.append("- PyG fidelity metrics were not available for this model (see the warnings of the run)")
    lines.append("")
    for row, feats, edges in rows:
        lines.append(f"## Transaction row {row} (TransactionID {int(prepared.df['TransactionID'].iloc[row])})")
        lines += [f"- `{n}`: {v:.3f}" for n, v in feats]
        lines += [f"- edge `{et[1]}`: {s} -> {d} ({v:.3f})" for et, s, d, v in edges] + [""]
    for key, edges in user_rows:
        lines.append(f"## User {key} (account-risk head)")
        lines += [f"- edge `{et[1]}`: {s} -> {d} ({v:.3f})" for et, s, d, v in edges] + [""]
    (out_dir / "explanations.md").write_text("\n".join(lines), encoding="utf-8")
    return {"stability": float(np.mean(stability)), "fidelity_top": float(np.mean(fid_top)),
            "fidelity_random": float(np.mean(fid_rand)), "n_fidelity": int(len(sample)),
            "pyg_fidelity_plus": mean_pyg and mean_pyg["fidelity_plus"],
            "pyg_fidelity_minus": mean_pyg and mean_pyg["fidelity_minus"],
            "pyg_characterization": mean_pyg and mean_pyg["characterization"]}
```
Fidelity and explainer behaviour depend on the Task 2 spike: `explain.mode` is `occlusion` if the spike found no working hetero `GNNExplainer`, in which case the edge, PyG-fidelity and user parts are skipped and the report says so. The occlusion deletion check runs in both modes.

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_explain.py -q`
Expected: the toy and occlusion tests pass; the `slow` hetero test passes if the spike chose `hetero` (run it with `-m slow`), otherwise record why in the ledger.

```bash
git add scripts/explain.py tests/test_explain.py
git commit -m "feat: add GNNExplainer and occlusion explanations with stability and fidelity checks" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Notebook support and notebooks 01-04

**Files:**
- Create: `pipeline/notebook_support.py`, `scripts/build_notebooks.py`, `notebooks/01_data_and_graph.ipynb`, `notebooks/02_train_and_compare.ipynb`, `notebooks/03_drift_and_retraining.ipynb`, `notebooks/04_explainability.ipynb` (the last four are generated by the builder), `tests/test_notebook_support.py`, `tests/test_notebooks.py`

**Interfaces:**
- Consumes: Tasks 3-14 (`prepare`, `period_of`, result CSV schemas from Tasks 12-13, `explain_report` outputs from Task 14).
- Produces in `pipeline.notebook_support`:
  - `fixture_mode() -> bool` (env `URE_FIXTURE == "1"` or `URE_QUICK == "1"`), `out_dir() -> Path` (env `URE_OUT_DIR`, default `plots`), `reports_dir() -> Path` (env `URE_REPORTS_DIR`, default `reports`).
  - `get_prepared() -> Prepared` (fixture data with `fixture_overrides()` in fixture mode, the real config otherwise).
  - `load_results(name) -> DataFrame` (raises `FileNotFoundError` naming the script that creates the file).
  - `seed_summary(df, metrics, by="model") -> DataFrame` with `<metric>_mean`, `<metric>_min`, `<metric>_max` columns (a range over seeds, never a confidence interval).
  - `plot_pr_auc(fraud) -> Figure`, `plot_weekly(weekly, models=None) -> Figure`, `plot_calibration(cal, models=None) -> Figure`, `plot_drift(drift) -> Figure`.
- Produces in `scripts.build_notebooks`: `NOTEBOOKS: dict[str, list[tuple[str, str]]]` (cell kind `"md"` or `"code"`, source) and `build(out_dir="notebooks") -> list[Path]`.
- Notebooks use the kernel `unified_risk_engine` (registered in Task 1). They only read result files and call library code; they never train.

- [ ] **Step 1: Write the failing tests** (`tests/test_notebook_support.py`)

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pipeline import notebook_support as ns


def test_paths_follow_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("URE_OUT_DIR", str(tmp_path / "o"))
    monkeypatch.setenv("URE_REPORTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("URE_FIXTURE", "1")
    assert ns.out_dir() == tmp_path / "o" and ns.reports_dir() == tmp_path / "r" and ns.fixture_mode()
    monkeypatch.delenv("URE_FIXTURE")
    assert not ns.fixture_mode()
    monkeypatch.setenv("URE_QUICK", "1")
    assert ns.fixture_mode()


def test_load_results_names_the_script_that_creates_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv("URE_OUT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="scripts.evaluate"):
        ns.load_results("results_fraud.csv")
    with pytest.raises(FileNotFoundError, match="scripts.drift_study"):
        ns.load_results("drift_weekly.csv")


def test_seed_summary_reports_a_range_not_a_confidence_interval():
    df = pd.DataFrame({"model": ["a", "a", "a", "b"], "seed": [0, 1, 2, 0], "pr_auc": [0.4, 0.5, 0.6, 0.3]})
    s = ns.seed_summary(df, ["pr_auc"]).set_index("model")
    assert s.loc["a", "pr_auc_mean"] == pytest.approx(0.5)
    assert s.loc["a", "pr_auc_min"] == 0.4 and s.loc["a", "pr_auc_max"] == 0.6
    assert s.loc["b", "pr_auc_min"] == s.loc["b", "pr_auc_max"] == 0.3


def test_plots_draw_something_and_survive_nan_weeks():
    fraud = pd.DataFrame({"model": ["a", "a", "b"], "seed": [0, 1, 0], "pr_auc": [0.4, 0.5, 0.3],
                          "pr_auc_lo": [0.3, 0.4, 0.2], "pr_auc_hi": [0.5, 0.6, 0.4], "prevalence": [0.03] * 3})
    weekly = pd.DataFrame({"model": "a", "seed": 0, "week": [0, 1, 2], "pr_auc": [0.4, np.nan, 0.3]})
    cal = pd.DataFrame({"model": "a", "seed": 0, "bin_lo": [0, 0.5], "bin_hi": [0.5, 1], "mean_pred": [0.1, 0.7],
                        "frac_pos": [0.05, 0.6], "count": [90, 10]})
    drift = pd.DataFrame({"kind": "k", "schedule": ["static"] * 2 + ["retrain"] * 2, "week": [0, 1, 0, 1],
                          "pr_auc": [0.5, 0.4, 0.5, 0.5]})
    for fig in (ns.plot_pr_auc(fraud), ns.plot_weekly(weekly), ns.plot_calibration(cal), ns.plot_drift(drift)):
        assert len(fig.axes[0].lines) + len(fig.axes[0].patches) + len(fig.axes[0].collections) > 0
        plt.close(fig)
```

`tests/test_notebooks.py` (slow; it runs the whole fixture pipeline once and then executes every notebook):

```python
import nbformat
import pytest
from nbclient import NotebookClient

from pipeline.config import PROJECT_ROOT, load_config
from pipeline.prepare import prepare
from scripts.build_notebooks import NOTEBOOKS, build
from scripts.drift_study import drift_study, summarize_drift
from scripts.evaluate import evaluate_all
from scripts.explain import explain_report
from scripts.train import default_spec, train_variant
from tests.fixtures import fixture_overrides, make_fixture


def test_every_notebook_cell_kind_is_valid():
    assert len(NOTEBOOKS) == 4
    for cells in NOTEBOOKS.values():
        assert cells and {kind for kind, _ in cells} <= {"md", "code"}


def test_build_writes_valid_notebooks_with_the_project_kernel(tmp_path):
    for path in build(tmp_path):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert nb.metadata["kernelspec"]["name"] == "unified_risk_engine"


@pytest.mark.slow
def test_notebooks_execute_on_fixture_results(tmp_path, monkeypatch):
    out, reports = tmp_path / "plots", tmp_path / "reports"
    cfg = load_config(overrides=fixture_overrides())
    cfg["explain"]["mode"] = "occlusion"
    prep = prepare(cfg, df=make_fixture()[0])
    evaluate_all(prep, out, variants=("multitask", "fraud_only"), seeds=[0], device="cpu", log=lambda *_: None)
    drift = drift_study(prep, kinds=("history_tabular",), log=lambda *_: None)
    drift.to_csv(out / "drift_weekly.csv", index=False)
    summarize_drift(drift).to_csv(out / "drift_summary.csv", index=False)
    train_variant(prep, default_spec(prep, "multitask", 0), "cpu", tmp_path / "m.pt")
    explain_report(prep, tmp_path / "m.pt", reports, device="cpu")
    monkeypatch.setenv("URE_FIXTURE", "1")
    monkeypatch.setenv("URE_OUT_DIR", str(out))
    monkeypatch.setenv("URE_REPORTS_DIR", str(reports))
    for path in build(tmp_path / "nb"):
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(PROJECT_ROOT)}}).execute()
```
(`kernel_name="python3"` is the interpreter running pytest; the saved notebooks keep the project kernel.)

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_notebook_support.py tests/test_notebooks.py -q -m "not slow"`
Expected: `ModuleNotFoundError: No module named 'pipeline.notebook_support'`.

- [ ] **Step 3: Implement the helpers** (`pipeline/notebook_support.py`)

```python
"""Helpers shared by the notebooks: where results live, how to load them, and the plots."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import load_config, resolve_path

HOW_TO = {
    "results_fraud.csv": "python -m scripts.evaluate", "results_account.csv": "python -m scripts.evaluate",
    "results_weekly.csv": "python -m scripts.evaluate", "calibration.csv": "python -m scripts.evaluate",
    "drift_weekly.csv": "python -m scripts.drift_study", "drift_summary.csv": "python -m scripts.drift_study",
}


def fixture_mode() -> bool:
    return os.environ.get("URE_FIXTURE") == "1" or os.environ.get("URE_QUICK") == "1"   # URE_QUICK: the spec's name for the tiny smoke run


def out_dir() -> Path:
    return resolve_path(os.environ.get("URE_OUT_DIR", "plots"))


def reports_dir() -> Path:
    return resolve_path(os.environ.get("URE_REPORTS_DIR", "reports"))


def get_prepared():
    from pipeline.prepare import prepare

    if fixture_mode():
        from tests.fixtures import fixture_overrides, make_fixture

        return prepare(load_config(overrides=fixture_overrides()), df=make_fixture()[0])
    return prepare(load_config())


def load_results(name: str) -> pd.DataFrame:
    path = out_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with `{HOW_TO.get(name, 'the matching script')}`.")
    return pd.read_csv(path)


def seed_summary(df: pd.DataFrame, metrics: list[str], by: str = "model") -> pd.DataFrame:
    g = df.groupby(by)[metrics]
    return g.mean().add_suffix("_mean").join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max")).reset_index()


def plot_pr_auc(fraud: pd.DataFrame):
    s = seed_summary(fraud, ["pr_auc"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(s))
    ax.barh(y, s["pr_auc_mean"], xerr=[s["pr_auc_mean"] - s["pr_auc_min"], s["pr_auc_max"] - s["pr_auc_mean"]],
            color="#4c78a8")
    ax.axvline(fraud["prevalence"].mean(), color="grey", linestyle="--", label="prevalence (chance)")
    ax.set_yticks(y, s["model"])
    ax.set_xlabel("test PR-AUC (bar = mean over seeds, whisker = min to max)")
    ax.legend()
    fig.tight_layout()
    return fig


def _by_model_week(weekly, models):
    models = models or list(weekly["model"].unique())
    return weekly[weekly["model"].isin(models)].groupby(["model", "week"], as_index=False)["pr_auc"].mean(), models


def plot_weekly(weekly: pd.DataFrame, models=None):
    data, models = _by_model_week(weekly, models)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for m in models:
        d = data[data["model"] == m]
        ax.plot(d["week"], d["pr_auc"], marker="o", label=m)
    ax.set_xlabel("week of the test period"), ax.set_ylabel("PR-AUC (gaps = no positives that week)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(cal: pd.DataFrame, models=None):
    models = models or list(cal["model"].unique())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    for m in models:
        d = cal[cal["model"] == m].groupby("mean_pred", as_index=False)["frac_pos"].mean()
        ax.plot(d["mean_pred"], d["frac_pos"], marker="o", label=m)
    ax.set_xlabel("mean predicted probability"), ax.set_ylabel("observed fraud rate")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drift(drift: pd.DataFrame):
    kinds = list(drift["kind"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 3.5), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for schedule, g in drift[drift["kind"] == kind].groupby("schedule"):
            ax.plot(g["week"], g["pr_auc"], marker="o", label=schedule)
        ax.set_title(kind), ax.set_xlabel("week since validation start"), ax.set_ylabel("PR-AUC")
        ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Implement the notebook builder** (`scripts/build_notebooks.py`)

```python
"""Generate the notebooks from the cell lists below, so their source is reviewable as plain text."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

KERNEL = {"name": "unified_risk_engine", "display_name": "Python (unified_risk_engine)", "language": "python"}
SETUP = "%matplotlib inline\nfrom pipeline.notebook_support import *\nimport pandas as pd\nimport numpy as np\npd.set_option('display.width', 160)"

NOTEBOOKS = {
    "01_data_and_graph.ipynb": [
        ("md", "# 1. Data, splits and graph\n\nWhat the data looks like, how it is split in time, and the graph that is built from it. "
               "If this says *fixture*, the numbers describe generated data and are **not findings**."),
        ("code", SETUP + "\nprep = get_prepared()\ncfg, df = prep.cfg, prep.df\nprint('FIXTURE DATA - not real results' if fixture_mode() else 'IEEE-CIS')"),
        ("code", "days = (df['TransactionDT'].max() - df['TransactionDT'].min()) / 86400\n"
                 "pd.Series({'transactions': len(df), 'users': df['user'].nunique(), 'days': round(days, 1),\n"
                 "           'fraud rate': round(df['isFraud'].mean(), 4)})"),
        ("md", "## Time periods\n\nTrain, validation and test are consecutive in time, separated by an embargo so that labels "
               "that arrive late cannot leak across a boundary."),
        ("code", "from pipeline.splits import DAY, period_of\n"
                 "p = prep.periods\nperiod = pd.Series(period_of(df['TransactionDT'].to_numpy(), p), name='period')\n"
                 "table = df.groupby(period).agg(rows=('isFraud', 'size'), fraud_rate=('isFraud', 'mean'))\ntable"),
        ("code", "gaps = {'train->validation (days)': (p.val_start - p.t_train) / DAY, 'validation->test (days)': (p.test_start - p.val_end) / DAY}\n"
                 "assert all(abs(g - p.embargo / DAY) < 1e-6 for g in gaps.values()), gaps\ngaps"),
        ("md", "## Account-risk population\n\nA proxy label from fraud history up to the end of training, for users with enough history."),
        ("code", "acc = prep.account\npd.DataFrame({'users': acc.groupby('split').size(), 'eligible': acc[acc['eligible']].groupby('split').size(),\n"
                 "              'positives': acc[acc['eligible']].groupby('split')['label'].sum()})"),
        ("md", "## Threshold choice\n\nThe account label's fraud-rate threshold is chosen by how many positives it leaves, never by model "
               "performance. The configured value is shown against alternatives."),
        ("code", "rates = acc[acc['eligible']]['fraud_rate']\n"
                 "print('configured threshold:', cfg['account']['fraud_rate_threshold'])\n"
                 "pd.DataFrame({t: {'positives': int((rates >= t).sum()), 'prevalence': round(float((rates >= t).mean()), 4)}\n"
                 "              for t in (0.1, 0.2, 0.3, 0.5)}).T"),
        ("md", "## Leakage checks\n\nTrain-period fraud rows come only from train users and end by the end of the training period."),
        ("code", "tr = prep.graph['transaction'].train_mask.numpy()\n"
                 "assert (df.loc[tr, 'TransactionDT'] <= p.t_train).all()\n"
                 "assert (prep.user_split.reindex(df.loc[tr, 'user']).to_numpy() == 'train').all()\n"
                 "print('leakage checks passed')"),
        ("md", "## Graph\n\nNode and edge counts of the heterogeneous graph (transaction, user, email_domain, device)."),
        ("code", "g = prep.graph\ndisplay(pd.DataFrame({'nodes': {nt: g[nt].num_nodes for nt in g.node_types}}))\n"
                 "display(pd.DataFrame({'edges': {str(et): g[et].edge_index.shape[1] for et in g.edge_types}}))"),
    ],
    "02_train_and_compare.ipynb": [
        ("md", "# 2. Results\n\nAll models on the same test period. Differences smaller than the bootstrap interval or the "
               "range over seeds are not claims; three seeds give a range, not a confidence interval."),
        ("code", SETUP + "\nfraud = load_results('results_fraud.csv')\naccount = load_results('results_account.csv')\n"
                 "print('config hash:', fraud['config_hash'].unique())"),
        ("code", "metrics = ['pr_auc', 'roc_auc', 'recall_at_fpr', 'cost', 'brier', 'ece']\nseed_summary(fraud, metrics).round(3)"),
        ("code", "plot_pr_auc(fraud);"),
        ("md", "## Alert budgets\n\nRecall, precision and the share of fraud *value* caught when only the top 0.5%, 1% or 5% of "
               "transactions can be reviewed."),
        ("code", "cols = [c for c in fraud.columns if c.split('@')[0] in ('recall', 'precision', 'amount_recall') and '@' in c]\n"
                 "fraud.groupby('model')[cols].mean().round(3)"),
        ("md", "## Cost\n\nExpected cost uses an assumed review cost per false alert (see the config); compare each model with flagging nothing."),
        ("code", "fraud.groupby('model')[['cost', 'cost_flag_none', 'cost_flag_all']].mean().round(0)"),
        ("md", "## Account-risk head and ablations"),
        ("code", "seed_summary(account, ['pr_auc', 'roc_auc']).round(3)"),
        ("md", "## Over time and calibration"),
        ("code", "plot_weekly(load_results('results_weekly.csv'));"),
        ("code", "plot_calibration(load_results('calibration.csv'));"),
    ],
    "03_drift_and_retraining.ipynb": [
        ("md", "# 3. Drift and retraining\n\nA model trained once decays as fraud patterns change. The *retrain* schedule refits the "
               "model on all labels known at each cut-off (older than the embargo)."),
        ("code", SETUP + "\ndrift = load_results('drift_weekly.csv')\nsummary = load_results('drift_summary.csv')\nsummary.round(4)"),
        ("code", "plot_drift(drift);"),
        ("md", "A negative `slope_per_week` for *static* means the frozen model gets worse each week. If *retrain* is not better, "
               "then on this data retraining did not help; report that as it is."),
    ],
    "04_explainability.ipynb": [
        ("md", "# 4. Explainability\n\nWhich inputs drive the model's score for individual transactions, and how much to trust "
               "those explanations. Features are anonymised, so names are column names, not business meaning; explanations "
               "describe the model, not causes."),
        ("code", SETUP + "\nfrom IPython.display import Image, Markdown, display\nr = reports_dir()\n"
                 "display(Markdown((r / 'explanations.md').read_text(encoding='utf-8')))"),
        ("code", "display(Image(filename=str(r / 'feature_importance.png')))"),
    ],
}


def build(out_dir="notebooks") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = new_notebook(cells=[new_markdown_cell(s) if kind == "md" else new_code_cell(s) for kind, s in cells],
                          metadata={"kernelspec": KERNEL})
        nbformat.validate(nb)
        path = out_dir / name
        nbformat.write(nb, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
```

- [ ] **Step 5: Run to verify pass**

Run: `"$PY" -m pytest tests/test_notebook_support.py tests/test_notebooks.py -q -m "not slow"` then `"$PY" -m pytest tests/test_notebooks.py -q -m slow`
Expected: all pass; the slow test takes a few minutes. If a notebook cell fails, fix the cell source in the builder (never edit the generated `.ipynb`).

- [ ] **Step 6: Build the committed notebooks and commit**

```bash
"$PY" -m scripts.build_notebooks
git add pipeline/notebook_support.py scripts/build_notebooks.py notebooks tests/test_notebook_support.py tests/test_notebooks.py
git commit -m "feat: add notebook helpers and notebooks 01-04 generated from reviewable cell lists" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
The committed notebooks have no outputs (real outputs are produced in Task 18 once real results exist).

---

### Task 16: Model card, README, and CI

**Files:**
- Create: `scripts/make_model_card.py`, `scripts/make_ci_requirements.py`, `README.md` (rewrite), `docs/MODEL_CARD.md` (generated), `requirements-ci.txt` (generated), `.github/workflows/tests.yml`, `tests/test_model_card.py`, `tests/test_ci_config.py`
- Modify: `scripts/drift_study.py` (`main` saves `config_hash`), `scripts/explain.py` (add `main`), `tests/fixtures.py` (`fixture_overrides` sets `train.temporal` from pyg-lib availability)

**Interfaces:**
- Consumes: result CSVs of Tasks 12-13 (`config_hash` column), `explain_report` metrics.
- Produces:
  - `scripts.explain.main(argv=None)` (CLI `--config --checkpoint --out`) writing `reports/explain_metrics.json` = `{"stability", "fidelity_top", "fidelity_random", "config_hash", "mode"}` besides the Task 14 files.
  - `scripts.drift_study.main` adds a `config_hash` column to `drift_weekly.csv` and `drift_summary.csv`.
  - In `scripts.make_model_card`: `load_tables(out_dir, reports_dir) -> dict` (keys `fraud`, `account`, `drift_summary` and `explain` when present), `check_single_config(tables) -> str` (raises `ValueError` containing "mixed config hashes"), `results_markdown(fraud, account) -> str`, `build_model_card(out_dir, reports_dir, fixture=False) -> str`, `splice_readme(readme, block) -> str` (replaces the text between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`, raises `ValueError` if the markers are missing), `main(argv=None)` (CLI `--out plots --reports reports --card docs/MODEL_CARD.md --readme README.md --fixture`).
  - In `scripts.make_ci_requirements`: `ci_requirements(lines, extra=()) -> list[str]` and CLI that writes `requirements-ci.txt`.

- [ ] **Step 1: Write the failing tests** (`tests/test_model_card.py`)

```python
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
```

`tests/test_ci_config.py`:

```python
import yaml

from pipeline.config import PROJECT_ROOT
from scripts.make_ci_requirements import ci_requirements


def test_ci_requirements_drop_local_builds_and_gpu_only_packages():
    lines = ["torch==2.14.1+cu130", "pyg-lib==0.9.0+pt214cu130", "pandas==2.3.1", "jupyterlab==4.4.0",
             "ipykernel==6.30.0", "numpy==2.3.0", "# a comment"]
    out = ci_requirements(lines, extra=["torch==2.14.1"])
    assert "torch==2.14.1" in out and "pandas==2.3.1" in out and "numpy==2.3.0" in out
    assert not any("cu130" in l or "jupyterlab" in l or "ipykernel" in l for l in out)
    assert not any(l.startswith("#") for l in out)


def test_workflow_runs_the_cpu_suite_without_secrets():
    wf = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    text = (PROJECT_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    steps = " ".join(str(s.get("run", "")) for s in wf["jobs"]["test"]["steps"])
    assert 'pytest -m "not gpu and not slow"' in steps and "requirements-ci.txt" in steps
    assert "secrets." not in text and "KAGGLE" not in text
    assert "cu130" not in (PROJECT_ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_model_card.py tests/test_ci_config.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.make_model_card'`.

- [ ] **Step 3: Implement the model-card generator** (`scripts/make_model_card.py`)

```python
"""Generate docs/MODEL_CARD.md and the README results block from the result files. Numbers are never typed by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.config import resolve_path
from pipeline.notebook_support import seed_summary

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
FRAUD_COLS = ["pr_auc", "recall_at_fpr", "recall@0.01", "amount_recall@0.01", "cost", "brier", "ece"]


def load_tables(out_dir, reports_dir) -> dict:
    out_dir, reports_dir = Path(out_dir), Path(reports_dir)
    tables = {"fraud": pd.read_csv(out_dir / "results_fraud.csv"), "account": pd.read_csv(out_dir / "results_account.csv")}
    if (out_dir / "drift_summary.csv").exists():
        tables["drift_summary"] = pd.read_csv(out_dir / "drift_summary.csv")
    if (reports_dir / "explain_metrics.json").exists():
        tables["explain"] = json.loads((reports_dir / "explain_metrics.json").read_text(encoding="utf-8"))
    return tables


def check_single_config(tables: dict) -> str:
    hashes = set()
    for name, t in tables.items():
        hashes |= set(t["config_hash"].unique()) if isinstance(t, pd.DataFrame) else {t["config_hash"]}
    if len(hashes) != 1:
        raise ValueError(f"mixed config hashes {sorted(hashes)}: re-run all steps with one configuration")
    return hashes.pop()


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def results_markdown(fraud: pd.DataFrame, account: pd.DataFrame) -> str:
    f = seed_summary(fraud, ["pr_auc"])
    others = fraud.groupby("model")[[c for c in FRAUD_COLS if c != "pr_auc" and c in fraud.columns]].mean().reset_index()
    f = f.merge(others, on="model")
    f["pr_auc (mean, min-max over seeds)"] = [f"{m:.3f} ({lo:.3f}-{hi:.3f})" for m, lo, hi in
                                              zip(f["pr_auc_mean"], f["pr_auc_min"], f["pr_auc_max"])]
    f = f[["model", "pr_auc (mean, min-max over seeds)"] + [c for c in others.columns if c != "model"]]
    a = seed_summary(account, ["pr_auc"])[["model", "pr_auc_mean", "pr_auc_min", "pr_auc_max"]]
    return ("Transaction fraud, test period:\n\n" + _table(f) + "\n\nAccount risk (proxy label), test users:\n\n" + _table(a))


def build_model_card(out_dir, reports_dir, fixture=False) -> str:
    tables = load_tables(out_dir, reports_dir)
    cfg_hash = check_single_config(tables)
    fraud = tables["fraud"]
    parts = ["# Model card: unified risk engine\n"]
    if fixture:
        parts.append("> **FIXTURE RESULTS: generated data, not findings.** Nothing below describes real fraud.\n")
    parts.append(f"Configuration hash: `{cfg_hash}`. Seeds: {sorted(fraud['seed'].unique().tolist())}. "
                 f"Test transactions: {int(fraud['n_test'].iloc[0])}; fraud prevalence {fraud['prevalence'].iloc[0]:.3f}.\n")
    parts.append("## Intended use\n\nA research prototype that ranks transactions by fraud risk and users by an account-risk proxy, "
                 "to study whether a temporal graph model adds value over strong tabular baselines. Not for production decisions, "
                 "credit, or any decision about a person.\n")
    parts.append("## Data\n\nIEEE-CIS Fraud Detection (Kaggle; Vesta Corporation): card-not-present e-commerce transactions with "
                 "anonymised features, about 590k training transactions over about six months. Users are an approximation built "
                 "from card and address fields, so \"user\" is a proxy for a real account holder.\n")
    parts.append("## Leakage controls\n\n- Chronological split: train, validation, test, separated by an embargo so late labels cannot cross a boundary.\n"
                 "- Users are split into disjoint train, validation and test groups; the account label uses history up to the end of training only.\n"
                 "- Feature statistics are fitted on the train period only; history features are strictly causal.\n"
                 "- Neighbour sampling only reaches transactions at or before the seed transaction's time.\n"
                 "- Baselines get the same history features as the graph model.\n")
    parts.append("## Results\n\n" + results_markdown(fraud, tables["account"]) + "\n")
    parts.append("Cost uses an assumed review cost per false alert; compare it with `cost_flag_none`, the cost of flagging nothing. "
                 "Differences smaller than the bootstrap interval or the range over seeds are not evidence of a better model.\n")
    parts.append("## Drift\n\n" + (_table(tables["drift_summary"].drop(columns=["config_hash"]))
                                   if "drift_summary" in tables else "The drift study was not run.") + "\n")
    e = tables.get("explain")
    parts.append("## Explanations\n\n" + (
        f"Mode `{e['mode']}`. Stability (top-k overlap across runs): {e['stability']:.2f}. Deleting the top-k features moves the "
        f"logit by {e['fidelity_top']:.2f} on average, against {e['fidelity_random']:.2f} for random features"
        + (f"; PyG fidelity+ {e['pyg_fidelity_plus']:.2f}, fidelity- {e['pyg_fidelity_minus']:.2f}, characterization "
           f"{e['pyg_characterization']:.2f}" if e.get("pyg_fidelity_plus") is not None else "")
        + ". Features are anonymised; explanations describe the model and are not causal."
        if e else "Explanations have not been generated.") + "\n")
    parts.append("## Limitations\n\n- One dataset, one split in time; no claim about other portfolios or later periods.\n"
                 "- The account label is a proxy derived from fraud history, not a ground-truth account-takeover label.\n"
                 "- Anonymised features limit what an explanation can say.\n- Three seeds give a range, not a confidence interval.\n"
                 "- The review cost is an assumption.\n- Any explanation of *why* a model wins is a hypothesis unless an ablation tested it.\n")
    return "\n".join(parts)


def splice_readme(readme: str, block: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError(f"README is missing the {START} / {END} markers (RESULTS block)")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the model card and update the README results block.")
    p.add_argument("--out", default="plots")
    p.add_argument("--reports", default="reports")
    p.add_argument("--card", default="docs/MODEL_CARD.md")
    p.add_argument("--readme", default=None, help="README path to update; omit to skip")
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args(argv)
    card = build_model_card(resolve_path(a.out), resolve_path(a.reports), fixture=a.fixture)
    card_path = resolve_path(a.card)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    print("wrote", card_path)
    if a.readme:
        tables = load_tables(resolve_path(a.out), resolve_path(a.reports))
        banner = "> **FIXTURE RESULTS: not findings.**\n\n" if a.fixture else ""
        path = resolve_path(a.readme)
        path.write_text(splice_readme(path.read_text(encoding="utf-8"),
                                      banner + results_markdown(tables["fraud"], tables["account"])), encoding="utf-8")
        print("updated", path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the small edits the generator depends on**

In `scripts/drift_study.py`, replace everything in `main` from the line `out = resolve_path(args.out)` to the end of the function with:

```python
    from pipeline.config import config_hash

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_drift(table)
    table.assign(config_hash=config_hash(cfg)).to_csv(out / "drift_weekly.csv", index=False)
    summary.assign(config_hash=config_hash(cfg)).to_csv(out / "drift_summary.csv", index=False)
    print(summary.to_string(index=False))
```

Append to `scripts/explain.py`:

```python
def main(argv=None):
    import argparse
    import json

    from pipeline.config import config_hash, load_config
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Explain test transactions and check stability and fidelity.")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/multitask_seed0.pt")
    p.add_argument("--out", default="reports")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    metrics = explain_report(prepare(cfg), resolve_path(args.checkpoint), resolve_path(args.out))
    metrics.update(config_hash=config_hash(cfg), mode=cfg["explain"]["mode"])
    (resolve_path(args.out) / "explain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
```

In `tests/fixtures.py`, make `fixture_overrides` robust on machines without pyg-lib (CI) by adding `"temporal": importlib.util.find_spec("pyg_lib") is not None` inside the `"train"` dict and `import importlib.util` at the top. Add to `tests/test_config.py`:

```python
def test_fixture_overrides_disable_temporal_sampling_without_pyg_lib(monkeypatch):
    import importlib.util

    from tests.fixtures import fixture_overrides

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert fixture_overrides()["train"]["temporal"] is False
```

- [ ] **Step 5: Implement the CI requirements generator** (`scripts/make_ci_requirements.py`)

```python
"""Derive requirements-ci.txt (CPU-only, no notebooks) from the pinned requirements.txt."""
from __future__ import annotations

import re
from pathlib import Path

DROP = ("torch", "pyg-lib", "pyg_lib", "jupyterlab", "ipykernel", "nbconvert")  # torch and pyg-lib are re-added for CPU


def ci_requirements(lines, extra=()):
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[=<>~ ]", line, maxsplit=1)[0].lower()
        if name in DROP:
            continue
        out.append(re.sub(r"\+[A-Za-z0-9.]+", "", line))
    return out + list(extra)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    extra = ["--extra-index-url https://download.pytorch.org/whl/cpu", "torch==2.14.1",
             "--find-links https://data.pyg.org/whl/torch-2.14.0+cpu.html", "pyg-lib==0.9.0+pt214cpu"]
    body = ci_requirements(lines, extra=[])
    (root / "requirements-ci.txt").write_text("\n".join(extra + body) + "\n", encoding="utf-8")
    print((root / "requirements-ci.txt").read_text(encoding="utf-8"))
```
`torch-geometric` stays in the CI file (it is not in `DROP`).
The `extra` lines in `__main__` come from the Task 2 spike report: if it recorded different CPU wheel names for Linux (or no Linux CPU wheel for pyg-lib), use that finding here instead and record it in the ledger. Without a pyg-lib wheel, drop the `pyg-lib` line; the fixture then disables temporal sampling by itself (Step 4) and CI covers the non-temporal paths only, which the README states.

- [ ] **Step 6: Write the workflow** (`.github/workflows/tests.yml`)

```yaml
name: tests
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
          cache: pip
          cache-dependency-path: requirements-ci.txt
      - name: Install CPU dependencies
        run: pip install -r requirements-ci.txt
      - name: Test suite (no GPU, no slow tests)
        run: python -m pytest -m "not gpu and not slow" -q
```

- [ ] **Step 7: Write the README**

Rewrite `README.md` with a CI status badge under the title (`![tests](https://github.com/jackalaka21/unified-risk-engine/actions/workflows/tests.yml/badge.svg)`; the repository is private, so the badge only renders for people signed in with access) and these sections (prose, no invented numbers): **What this is** (one paragraph: temporal heterogeneous GNN for transaction fraud and account-risk proxy on IEEE-CIS, evaluated against history-aware tree baselines); **The question** (does the graph add anything over a strong tabular model with the same history features, and does the shared multi-task backbone help or hurt?); **Design in brief** (graph schema, leakage controls, embargo, user-level split, temporal sampling, metrics); **Results** containing exactly the two marker lines with a one-line sentence between them (`No results yet. Run python -m scripts.run_all, then python -m scripts.make_model_card --readme README.md.`); **How to run** (venv, the pinned install commands from Task 1, data location `data/` with the two Kaggle file names, `python -m scripts.run_all`, notebooks, tests); **Limitations**; **Layout**; **Scoring service and latency** (a short "added in phase 2" note that Task 22 fills in).

- [ ] **Step 8: Generate the CI file, run the tests, and commit**

```bash
"$PY" -m scripts.make_ci_requirements
"$PY" -m pytest tests/test_model_card.py tests/test_ci_config.py tests/test_config.py -q
git add scripts/make_model_card.py scripts/make_ci_requirements.py scripts/drift_study.py scripts/explain.py tests/fixtures.py tests/test_config.py tests/test_model_card.py tests/test_ci_config.py README.md requirements-ci.txt .github/workflows/tests.yml
git commit -m "feat: add the model card generator, README results block and CPU CI" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
Expected: all pass. `docs/MODEL_CARD.md` is committed in Task 18 from real results. Do not push; CI itself runs after the user pushes, so the first CI run is verified in Task 22 once the user agrees to a push.

---

### Task 17: End-to-end driver and the fixture run

**Files:**
- Create: `scripts/run_all.py`, `tests/test_run_all.py`

**Interfaces:**
- Consumes: Tasks 9-16.
- Produces in `scripts.run_all`: `run_all(cfg, out_dir, reports_dir, models_dir, df=None, variants=("multitask","fraud_only","account_only","no_graph"), seeds=None, drift_kinds=("history_tabular","gnn_fraud_only"), device=None, fixture=False, log=print) -> dict[str, Path]` (runs prepare, evaluation, drift study, one `multitask` checkpoint, explanations and the model card; every table carries the same config hash) and `main(argv=None)` with `--config --out --reports --models --fixture --seeds`. `--fixture` generates `tests.fixtures.make_fixture()` data and applies `fixture_overrides()`, writes under `outputs/fixture/` by default, and prints the not-findings banner; it never writes into `data/`.

- [ ] **Step 1: Write the failing test** (`tests/test_run_all.py`)

```python
import pandas as pd
import pytest

from pipeline.config import load_config
from scripts.run_all import run_all
from tests.fixtures import fixture_overrides, make_fixture


@pytest.mark.slow
def test_run_all_produces_every_artifact_with_one_config_hash(tmp_path):
    cfg = load_config(overrides=fixture_overrides())
    cfg["explain"]["mode"] = "occlusion"
    paths = run_all(cfg, tmp_path / "plots", tmp_path / "reports", tmp_path / "models", df=make_fixture()[0],
                    variants=("multitask", "fraud_only"), seeds=[0], drift_kinds=("history_tabular",),
                    device="cpu", fixture=True, log=lambda *_: None)
    for key in ("fraud", "account", "weekly", "calibration", "drift_weekly", "drift_summary", "explanations", "card"):
        assert paths[key].exists(), key
    hashes = {pd.read_csv(paths[k])["config_hash"].iloc[0] for k in ("fraud", "account", "drift_summary")}
    assert len(hashes) == 1
    assert "FIXTURE RESULTS" in paths["card"].read_text(encoding="utf-8")
    assert (tmp_path / "models" / "multitask_seed0.pt").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_run_all.py -q -m slow`
Expected: `ModuleNotFoundError: No module named 'scripts.run_all'`.

- [ ] **Step 3: Implement** (`scripts/run_all.py`)

```python
"""One command from raw data to evaluation tables, drift study, explanations and the model card."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.prepare import prepare
from scripts.drift_study import drift_study, summarize_drift
from scripts.evaluate import evaluate_all
from scripts.explain import explain_report
from scripts.make_model_card import build_model_card
from scripts.train import default_spec, train_variant


def run_all(cfg, out_dir, reports_dir, models_dir, df=None,
            variants=("multitask", "fraud_only", "account_only", "no_graph"), seeds=None,
            drift_kinds=("history_tabular", "gnn_fraud_only"), device=None, fixture=False, log=print):
    out_dir, reports_dir, models_dir = Path(out_dir), Path(reports_dir), Path(models_dir)
    device = resolve_device(device or cfg["train"]["device"])
    seeds = cfg["train"]["seeds"] if seeds is None else seeds
    if fixture:
        log("FIXTURE RUN: generated data, results are not findings")
    prepared = prepare(cfg, df=df)
    paths = evaluate_all(prepared, out_dir, variants=variants, seeds=seeds, device=device, log=log)
    h = config_hash(cfg)
    drift = drift_study(prepared, seed=cfg["drift"]["seed"], kinds=drift_kinds, device=device, log=log)
    paths["drift_weekly"], paths["drift_summary"] = out_dir / "drift_weekly.csv", out_dir / "drift_summary.csv"
    drift.assign(config_hash=h).to_csv(paths["drift_weekly"], index=False)
    summarize_drift(drift).assign(config_hash=h).to_csv(paths["drift_summary"], index=False)
    ckpt = models_dir / "multitask_seed0.pt"  # one extra run so the explanations have a checkpoint to load
    train_variant(prepared, default_spec(prepared, "multitask", seeds[0]), device, ckpt)
    metrics = explain_report(prepared, ckpt, reports_dir, device)
    metrics.update(config_hash=h, mode=cfg["explain"]["mode"])
    (reports_dir / "explain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["explanations"] = reports_dir / "explanations.md"
    paths["card"] = out_dir / "MODEL_CARD.md"
    paths["card"].write_text(build_model_card(out_dir, reports_dir, fixture=fixture), encoding="utf-8")
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the whole pipeline.")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--reports", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--fixture", action="store_true", help="use generated data; results are not findings")
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        cfg, df = load_config(a.config, overrides=fixture_overrides()), make_fixture()[0]
        base = "outputs/fixture"
    else:
        cfg, df, base = load_config(a.config), None, None
    out = resolve_path(a.out or (f"{base}/plots" if base else "plots"))
    reports = resolve_path(a.reports or (f"{base}/reports" if base else "reports"))
    models = resolve_path(a.models or (f"{base}/models" if base else "models"))
    paths = run_all(cfg, out, reports, models, df=df, seeds=a.seeds, fixture=a.fixture)
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
```
The model card is written next to the tables (`MODEL_CARD.md` in `out_dir`), never straight into `docs/`; Task 18 copies the real-data card into `docs/MODEL_CARD.md` on purpose, so a fixture run can never overwrite it.

- [ ] **Step 4: Run to verify pass, then do the fixture run**

Run: `"$PY" -m pytest tests/test_run_all.py -q -m slow` then `"$PY" -m scripts.run_all --fixture --seeds 0`
Expected: the test passes; the fixture run prints the banner and writes under `outputs/fixture/` (git-ignored). Skim `outputs/fixture/plots/results_fraud.csv`: every model has finite PR-AUC. Fixture numbers are never quoted anywhere.

- [ ] **Step 5: Run the whole fast suite and commit**

Run: `"$PY" -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

```bash
git add scripts/run_all.py tests/test_run_all.py
git commit -m "feat: add the end-to-end driver with a clearly labelled fixture mode" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 18: Real-data gate, GPU test and phase 1 review

**Files:**
- Create: `scripts/check_data.py`, `tests/test_check_data.py`, `tests/test_gpu.py`
- Generated and committed once real results exist: `plots/*.csv`, `plots/*.png`, `docs/MODEL_CARD.md`, executed `notebooks/*.ipynb`, README results block

**Interfaces:**
- Produces in `scripts.check_data`: `check_data(cfg) -> list[str]` (problems; empty list means usable) and a CLI that prints the problems and, when a file is missing, the Kaggle download command for the user to run themselves.

- [ ] **Step 1: Write the failing tests** (`tests/test_check_data.py`, `tests/test_gpu.py`)

```python
# tests/test_check_data.py
import pandas as pd

from pipeline.config import load_config
from scripts.check_data import check_data
from tests.fixtures import write_fixture


def cfg_for(tmp_path):
    return load_config(overrides={"data": {"dir": str(tmp_path)}})


def test_missing_files_give_the_download_instruction(tmp_path):
    problems = check_data(cfg_for(tmp_path))
    assert any("train_transaction.csv" in p for p in problems)
    assert any("kaggle competitions download -c ieee-fraud-detection" in p for p in problems)


def test_a_small_but_valid_file_is_reported_as_not_the_full_dataset(tmp_path):
    write_fixture(tmp_path, n_tx=500, n_users=30, days=10)
    problems = check_data(cfg_for(tmp_path))
    assert any("rows" in p for p in problems) and not any("missing columns" in p for p in problems)


def test_missing_columns_are_named(tmp_path):
    write_fixture(tmp_path, n_tx=500, n_users=30, days=10)
    path = tmp_path / "train_transaction.csv"
    pd.read_csv(path).drop(columns=["card1"]).to_csv(path, index=False)
    assert any("card1" in p for p in check_data(cfg_for(tmp_path)))
```
```python
# tests/test_gpu.py
import pytest
import torch

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.train import default_spec, predict_scores, train_variant
from tests.fixtures import fixture_overrides, make_fixture

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_one_epoch_on_cuda_matches_cpu_scores_closely():
    prep = prepare(load_config(overrides=fixture_overrides()), df=make_fixture()[0])
    spec = default_spec(prep, "fraud_only", 0)
    cpu = train_variant(prep, spec, "cpu")
    gpu = train_variant(prep, spec, "cuda")
    idx = spec.fraud_val_idx[:64]
    a = predict_scores(prep, cpu.model, cpu.heads, "fraud", idx, "cpu")
    b = predict_scores(prep, gpu.model, gpu.heads, "fraud", idx, "cuda")
    assert abs(a.mean() - b.mean()) < 0.1 and ((b >= 0) & (b <= 1)).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_check_data.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.check_data'`.

- [ ] **Step 3: Implement** (`scripts/check_data.py`)

```python
"""Check that the Kaggle files are in place before a real run. Never touches credentials."""
from __future__ import annotations

import pandas as pd

from pipeline.config import load_config, resolve_path
from pipeline.data_loading import REQUIRED_TX

EXPECTED_ROWS = 590_540
HOW_TO = ("Download the data yourself (it needs your own Kaggle login and acceptance of the competition rules): "
          "`kaggle competitions download -c ieee-fraud-detection`, unzip train_transaction.csv "
          "(and optionally train_identity.csv) into the data directory.")


def check_data(cfg) -> list[str]:
    folder = resolve_path(cfg["data"]["dir"])
    path = folder / cfg["data"]["transaction_file"]
    if not path.exists():
        return [f"{path} not found. {HOW_TO}"]
    problems = []
    head = pd.read_csv(path, nrows=5)
    missing = [c for c in REQUIRED_TX if c not in head.columns]
    if missing:
        problems.append(f"{path.name} is missing columns {missing}")
    rows = sum(1 for _ in open(path, "rb")) - 1
    if rows < EXPECTED_ROWS * 0.9:
        problems.append(f"{path.name} has {rows} rows; the full IEEE-CIS training file has about {EXPECTED_ROWS}. "
                        "Real results need the full file.")
    if not (folder / cfg["data"]["identity_file"]).exists():
        problems.append(f"{cfg['data']['identity_file']} not found: device edges will be absent (allowed, but the graph is smaller).")
    return problems


if __name__ == "__main__":
    out = check_data(load_config())
    print("\n".join(out) if out else "data looks complete")
```

- [ ] **Step 4: Run to verify pass and commit**

Run: `"$PY" -m pytest tests/test_check_data.py -q` and `"$PY" -m pytest tests/test_gpu.py -q -m gpu`
Expected: pass (the GPU test runs on this machine's RTX 3070 Ti and skips elsewhere).

```bash
git add scripts/check_data.py tests/test_check_data.py tests/test_gpu.py
git commit -m "feat: add the data check and a GPU consistency test" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Gate on the data (a human step)**

Run: `"$PY" -m scripts.check_data`
- If it reports the files missing or incomplete: STOP the real-data steps, tell the user exactly what to download and where, and continue with Phase 2 on the fixture (clearly labelled). Do not log in to Kaggle or ask for credentials.
- If it reports `data looks complete` (a missing identity file is a warning, not a stop): continue.

- [ ] **Step 6: Real run and sanity gates (only with the full data)**

```bash
"$PY" -m scripts.run_all
```
Then check, and record in the ledger, before reporting anything:
- Every model's test PR-AUC is finite. If any model's test PR-AUC is above 0.95, treat it as a suspected leak: stop, re-read the leakage tests in Tasks 4-7, and find the cause before continuing.
- The history-aware tabular baseline is at least as good as the naive tabular baseline (it has strictly more information). If not, suspect a bug in `history_features`.
- All result tables share one config hash (the card generator enforces it).
- GNN training finished within a reasonable time; if not, record the per-epoch time and the `max_steps_per_epoch` used.

- [ ] **Step 7: Generate the committed artifacts**

```bash
"$PY" -m scripts.make_model_card --out plots --reports reports --card docs/MODEL_CARD.md --readme README.md
"$PY" -m scripts.build_notebooks
"$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=unified_risk_engine --ExecutePreprocessor.timeout=900 notebooks/01_data_and_graph.ipynb notebooks/02_train_and_compare.ipynb notebooks/03_drift_and_retraining.ipynb notebooks/04_explainability.ipynb
```
Read `docs/MODEL_CARD.md` and the README results block once end to end and confirm that every sentence about results is either a generated number or an explicitly labelled hypothesis (Global Constraints). Remove any sentence that explains *why* a model won unless an ablation in the tables supports it.

- [ ] **Step 8: Whole-branch review of phase 1, then commit**

Run `"$PY" -m pytest -m "not gpu and not slow" -q` and `"$PY" -m pytest -m slow -q`; expected: all pass. Native method: dispatch one fresh reviewer on the most capable model with the spec, this plan and `git diff main...HEAD`; ask it to look for leakage (anything in train-time computation that touches validation/test rows or labels, the embargo, the user split), metric correctness, train/serve-relevant divergences, and statements in the README or model card that the tables do not support. Fix what it confirms.

```bash
git add plots/*.csv plots/*.png docs/MODEL_CARD.md README.md notebooks
git commit -m "docs: add real-data results, model card and executed notebooks" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
Skip the commit's results files when the data gate stopped at Step 5. Do not push; ask the user.

---

## Phase 2: scoring service and latency benchmark

Phase 2 starts after Task 18. It can run on the fixture while the real data is missing; every number it produces from the fixture is labelled and never reported as a finding.

### Task 19: Graph index, online scorers and the parity gate

**Files:**
- Create: `serving/__init__.py` (empty), `serving/graph_index.py`, `serving/scorer.py`, `scripts/spike_serving.py`, `tests/serving_helpers.py`, `tests/test_graph_index.py`, `tests/test_scorer_parity.py`, `tests/test_spike_serving.py`, `docs/serving-spike.md` (generated)

**Interfaces:**
- Consumes: `Prepared`, `HeteroGNN`, `TaskHeads`, `FeatureTransformer.transform`, `finalize`, `build_modules`, `warm_up`, `make_loader`, `predict_scores`, `history_features` column names.
- Produces in `serving.graph_index`:
  - `ENTITIES = ("user", "email_domain", "device")`, `EDGES` (entity -> `(edge into transaction, edge out of transaction)`, identical to `pipeline.graph_builder.EDGE_TYPES`), `DAY`, `PREFIX`.
  - `OutOfOrderError(ValueError)`.
  - `Subgraph` dataclass: `x` (`ndarray`, seed first), `tx_rows` (`list[int]`, global row per transaction node, seed = `-1`), `entity_keys` (`dict[entity, list[str]]`), `edges` (`dict[edge_type, list[(src_local, dst_local)]]`), property `num_nodes`.
  - `GraphIndex(n_features, fanouts=(15, 10))` with `__len__`, `last_time`, `add(time, amount, keys, x) -> row`, `lookup(entity, key, t, k) -> list[int]`, `subgraph(keys, t, x_new) -> Subgraph`, `history(keys, t) -> dict` (same names as `history_features` columns), `x_of(row)`, `classmethod from_prepared(prepared, upto_row=None)`.
  - `keys_from_frame(frame, i) -> (user, email, device)` with missing values as `None`.
- Produces in `serving.scorer`:
  - `raw_kinds_of(df) -> dict[col, "float"|"int"|"object"]`, `payload_of(df, i) -> dict` (JSON-safe, no label, no derived keys), `build_frame(payloads, kinds) -> DataFrame` (casts, finalises, adds `_pos`), `collate(subs, device="cpu") -> (x_dict, edge_index_dict, num_nodes, seed_positions)`.
  - `OnlineScorer(model, heads, transformer, index, raw_kinds, device="cpu", version="dev", late="reject", info=None)` with `score(payload) -> float`, `score_batch(payloads, timings=None) -> list[float]` (stateful: each scored transaction is appended to the index).
  - `TabularOnlineScorer(model, transformer, index, raw_kinds, hist_columns=None, version="dev", late="reject", info=None)` with the same two methods (`hist_columns=None` is the naive tabular model).
  - `timings` keys: `feature_transform`, `neighbour_lookup`, `model_forward`, `index_update` (seconds, summed over the call).
- Rules the serving path must reproduce (spec section 4 and 9b): neighbours are transactions at or before the request time (earlier-or-equal), the `"last"` strategy keeps the most recent `fanouts[1]` of them, the new transaction itself is always a neighbour of its own entities, entities with no key (no email or device) produce no node and no edge.

- [ ] **Step 1: Write the failing index tests** (`tests/test_graph_index.py`)

```python
import numpy as np
import pytest

from pipeline.graph_builder import EDGE_TYPES
from serving.graph_index import EDGES, GraphIndex, OutOfOrderError

F = 3
Z = np.zeros(F, np.float32)


def idx_with(rows, k2=3):
    ix = GraphIndex(F, fanouts=(15, k2))
    for t, a, u, e, d in rows:
        ix.add(t, a, (u, e, d), np.full(F, float(t), np.float32))
    return ix


def test_edge_table_matches_the_graph_builder():
    assert {et for pair in EDGES.values() for et in pair} == set(EDGE_TYPES)


def test_lookup_returns_the_most_recent_k_up_to_t():
    ix = idx_with([(10, 1, "u", None, None), (20, 1, "u", None, None), (30, 1, "u", None, None), (40, 1, "u", None, None)])
    assert ix.lookup("user", "u", 40, 3) == [1, 2, 3]       # at or before t, most recent three
    assert ix.lookup("user", "u", 25, 3) == [0, 1]
    assert ix.lookup("user", "u", 5, 3) == []
    assert ix.lookup("user", "nobody", 99, 3) == []


def test_subgraph_adds_the_new_transaction_as_the_most_recent_neighbour():
    ix = idx_with([(10, 1, "u", "e", None), (20, 1, "u", None, None)])
    sg = ix.subgraph(("u", "e", None), 30, Z)
    assert sg.tx_rows == [-1, 0, 1] and sg.x.shape == (3, F) and sg.x[1, 0] == 10 and sg.x[2, 0] == 20
    assert sg.entity_keys == {"user": ["u"], "email_domain": ["e"], "device": []}
    assert sg.edges[("user", "owns", "transaction")] == [(0, 0)]
    assert sg.edges[("transaction", "rev_owns", "user")] == [(1, 0), (2, 0), (0, 0)]
    assert sg.edges[("email_domain", "rev_uses_email", "transaction")] == [(0, 0)]
    assert sg.edges[("transaction", "uses_email", "email_domain")] == [(1, 0), (0, 0)]
    assert not sg.edges.get(("device", "rev_uses_device", "transaction"))


def test_only_k_minus_one_old_neighbours_are_kept():
    ix = idx_with([(t, 1, "u", None, None) for t in range(10, 100, 10)])      # nine rows, fan-out 3
    sg = ix.subgraph(("u", None, None), 100, Z)
    assert sg.tx_rows == [-1, 7, 8]                                          # two old ones and the new one


def test_unseen_and_missing_keys():
    ix = idx_with([(10, 1, "u", "e", "d")])
    sg = ix.subgraph(("new_user", None, np.nan), 20, Z)
    assert sg.entity_keys == {"user": ["new_user"], "email_domain": [], "device": []}
    assert sg.tx_rows == [-1]
    assert sg.edges[("user", "owns", "transaction")] == [(0, 0)] and sg.edges[("transaction", "rev_owns", "user")] == [(0, 0)]
    assert not sg.edges.get(("transaction", "uses_email", "email_domain"))
    assert not sg.edges.get(("transaction", "uses_device", "device"))


def test_out_of_order_arrivals_are_rejected_but_equal_times_are_fine():
    ix = idx_with([(10, 1, "u", None, None)])
    with pytest.raises(OutOfOrderError):
        ix.add(5, 1, ("u", None, None), Z)
    ix.add(10, 1, ("u", None, None), Z)
    assert len(ix) == 2


def test_history_by_hand_with_strictly_earlier_semantics():
    ix = idx_with([(0, 10.0, "u", "e", "A"), (100, 20.0, "u", "e", "A")])
    h = ix.history(("u", "e", "A"), 200)
    assert h["user_n_prior"] == 2 and h["user_n_1d"] == 2 and h["user_amt_mean"] == 15.0
    assert h["user_amt_std"] == pytest.approx(50 ** 0.5) and h["user_secs_since_prev"] == 100
    assert h["user_distinct_devices"] == 1 and h["user_distinct_emails"] == 1
    tie = ix.history(("u", "e", "A"), 100)                                    # the row at t=100 does not count itself
    assert tie["user_n_prior"] == 1 and np.isnan(tie["user_amt_std"]) and tie["user_secs_since_prev"] == 100


def test_history_for_unseen_and_missing_entities():
    ix = idx_with([(0, 10.0, "u", "e", "A")])
    h = ix.history(("zzz", None, None), 50)
    assert h["user_n_prior"] == 0 and np.isnan(h["user_amt_mean"]) and h["user_distinct_devices"] == 0
    assert np.isnan(h["email_n_prior"]) and np.isnan(h["device_n_prior"])
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_graph_index.py -q`
Expected: `ModuleNotFoundError: No module named 'serving'`.

- [ ] **Step 3: Implement the index** (`serving/graph_index.py`)

```python
"""Append-only, time-ordered neighbour index that rebuilds the training-time sampled neighbourhood online.

It reproduces PyG temporal neighbour sampling with the "last" strategy on a two-hop graph: the seed transaction
connects to its user, email domain and device; each of those connects to its `fanouts[1]` most recent
transactions at or before the seed time. The new transaction is always the most recent of its own entities.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd

ENTITIES = ("user", "email_domain", "device")
DAY = 86400
# entity -> (edge into the transaction, edge out of the transaction); must equal pipeline.graph_builder.EDGE_TYPES
EDGES = {
    "user": (("user", "owns", "transaction"), ("transaction", "rev_owns", "user")),
    "email_domain": (("email_domain", "rev_uses_email", "transaction"), ("transaction", "uses_email", "email_domain")),
    "device": (("device", "rev_uses_device", "transaction"), ("transaction", "uses_device", "device")),
}
PREFIX = {"user": "user", "email_domain": "email", "device": "device"}   # history_features column prefixes
BASE_NAMES = ["n_prior", "n_1d", "n_7d", "amt_mean", "amt_std", "secs_since_prev"]


class OutOfOrderError(ValueError):
    """A request is older than a transaction already in the index."""


def _missing(key) -> bool:
    return key is None or (not isinstance(key, str) and bool(pd.isna(key)))


def _clean(keys):
    return tuple(None if _missing(k) else k for k in keys)


def keys_from_frame(frame, i):
    return _clean((frame["user"].iat[i], frame["email"].iat[i], frame["device"].iat[i]))


class _Members:
    """Everything one entity has done, in time order, with prefix sums so history queries are O(log n)."""

    __slots__ = ("rows", "times", "cs", "cs2", "dev_prefix", "mail_prefix")

    def __init__(self):
        self.rows, self.times = [], []
        self.cs, self.cs2 = [0.0], [0.0]
        self.dev_prefix, self.mail_prefix = [0], [0]       # distinct devices / emails over the first j members (users)


_EMPTY = _Members()                                        # shared, never mutated: an entity seen for the first time


@dataclass
class Subgraph:
    x: np.ndarray
    tx_rows: list
    entity_keys: dict
    edges: dict

    @property
    def num_nodes(self) -> dict:
        return {"transaction": len(self.tx_rows), **{e: len(self.entity_keys[e]) for e in ENTITIES}}


class GraphIndex:
    def __init__(self, n_features: int, fanouts=(15, 10)):
        if len(fanouts) != 2:
            raise ValueError("the index reproduces a two-hop sampler: train.num_neighbors needs exactly two entries")
        self.n_features, self.k2 = n_features, int(fanouts[1])
        self.members = {e: {} for e in ENTITIES}
        self._seen = {}                                    # user key -> (device keys, email keys)
        self.times, self.amounts, self.row_keys = [], [], []
        self._x = np.zeros((1024, n_features), np.float32)
        self.last_time = -np.inf

    def __len__(self) -> int:
        return len(self.times)

    def x_of(self, row: int) -> np.ndarray:
        return self._x[row]

    def add(self, time, amount, keys, x) -> int:
        if time < self.last_time:
            raise OutOfOrderError(f"transaction time {time} is older than the newest indexed time {self.last_time}")
        keys = _clean(keys)
        row = len(self.times)
        if row >= len(self._x):
            self._x = np.concatenate([self._x, np.zeros_like(self._x)])
        self._x[row] = x
        for e, key in zip(ENTITIES, keys):
            if key is None:
                continue
            m = self.members[e].setdefault(key, _Members())
            m.rows.append(row)
            m.times.append(time)
            m.cs.append(m.cs[-1] + amount)
            m.cs2.append(m.cs2[-1] + amount * amount)
            if e == "user":
                devices, emails = self._seen.setdefault(key, (set(), set()))
                if keys[2] is not None:
                    devices.add(keys[2])
                if keys[1] is not None:
                    emails.add(keys[1])
                m.dev_prefix.append(len(devices))
                m.mail_prefix.append(len(emails))
        self.times.append(time)
        self.amounts.append(amount)
        self.row_keys.append(keys)
        self.last_time = time
        return row

    def lookup(self, entity: str, key, t, k: int) -> list:
        m = self.members[entity].get(key)
        if m is None:
            return []
        hi = bisect_right(m.times, t)
        return m.rows[max(0, hi - k):hi]

    def subgraph(self, keys, t, x_new) -> Subgraph:
        keys = _clean(keys)
        tx_rows, tx_x, local = [-1], [np.asarray(x_new, np.float32)], {}
        entity_keys = {e: [] for e in ENTITIES}
        edges: dict = {}
        for e, key in zip(ENTITIES, keys):
            if key is None:
                continue
            into, out = EDGES[e]
            entity_keys[e].append(key)
            edges.setdefault(into, []).append((0, 0))
            for row in self.lookup(e, key, t, self.k2 - 1):
                if row not in local:
                    local[row] = len(tx_rows)
                    tx_rows.append(row)
                    tx_x.append(self._x[row])
                edges.setdefault(out, []).append((local[row], 0))
            edges.setdefault(out, []).append((0, 0))        # the new transaction is the most recent neighbour
        return Subgraph(np.stack(tx_x), tx_rows, entity_keys, edges)

    def history(self, keys, t) -> dict:
        """The tabular baseline's history features for a transaction arriving at time `t` (strictly earlier rows only)."""
        out = {}
        for e, key in zip(ENTITIES, _clean(keys)):
            pre = PREFIX[e]
            names = BASE_NAMES + (["distinct_devices", "distinct_emails"] if e == "user" else [])
            if key is None:
                out.update({f"{pre}_{n}": np.nan for n in names})
                continue
            m = self.members[e].get(key, _EMPTY)
            prior = bisect_left(m.times, t)
            mean = m.cs[prior] / prior if prior > 0 else np.nan
            var = (m.cs2[prior] - prior * mean ** 2) / (prior - 1) if prior > 1 else np.nan
            out[f"{pre}_n_prior"] = float(prior)
            out[f"{pre}_n_1d"] = float(prior - bisect_left(m.times, t - DAY))
            out[f"{pre}_n_7d"] = float(prior - bisect_left(m.times, t - 7 * DAY))
            out[f"{pre}_amt_mean"] = mean
            out[f"{pre}_amt_std"] = float(np.sqrt(max(var, 0.0))) if prior > 1 else np.nan
            out[f"{pre}_secs_since_prev"] = float(t - m.times[prior - 1]) if prior > 0 else np.nan
            if e == "user":
                out[f"{pre}_distinct_devices"] = float(m.dev_prefix[prior])
                out[f"{pre}_distinct_emails"] = float(m.mail_prefix[prior])
        return out

    @classmethod
    def from_prepared(cls, prepared, upto_row=None) -> "GraphIndex":
        df = prepared.df
        n = len(df) if upto_row is None else int(upto_row)
        index = cls(prepared.x.shape[1], prepared.cfg["train"]["num_neighbors"])
        t, a = df["TransactionDT"].to_numpy(np.int64), df["TransactionAmt"].to_numpy(float)
        keys = list(zip(df["user"], df["email"], df["device"]))
        for i in range(n):
            index.add(int(t[i]), float(a[i]), keys[i], prepared.x[i])
        return index
```
Note on the interface: `history` returns floats (counts included) so the vector aligns with `history_features`, whose columns are floats.

- [ ] **Step 4: Run to verify pass**

Run: `"$PY" -m pytest tests/test_graph_index.py -q`
Expected: all pass.

- [ ] **Step 5: Write the shared helpers and the failing scorer tests**

`tests/serving_helpers.py`:

```python
import importlib.util

import torch

from pipeline.config import load_config
from pipeline.prepare import prepare
from scripts.train import build_modules, warm_up
from serving.graph_index import GraphIndex
from serving.scorer import OnlineScorer, raw_kinds_of
from tests.fixtures import fixture_overrides, make_fixture

HAS_TEMPORAL = importlib.util.find_spec("pyg_lib") is not None


def last_rule_prepared():
    overrides = fixture_overrides()
    overrides["train"]["temporal_strategy"] = "last"
    return prepare(load_config(overrides=overrides), df=make_fixture()[0])


def make_scorer(prep, upto, seed=0, **kwargs):
    """A scorer with a seeded, untrained model: parity does not need training, only identical weights."""
    torch.manual_seed(seed)
    model, heads = build_modules(prep, "cpu")
    warm_up(model, heads, prep, "cpu")
    index = GraphIndex.from_prepared(prep, upto)
    return OnlineScorer(model, heads, prep.transformer, index, raw_kinds_of(prep.df), **kwargs)
```

`tests/test_scorer_parity.py`:

```python
import numpy as np
import pandas as pd
import pytest

from scripts.train import make_loader, predict_scores
from serving.graph_index import GraphIndex, OutOfOrderError, keys_from_frame
from serving.scorer import TabularOnlineScorer, payload_of, raw_kinds_of
from tests.serving_helpers import HAS_TEMPORAL, last_rule_prepared, make_scorer

needs_temporal = pytest.mark.skipif(not HAS_TEMPORAL, reason="temporal sampling needs pyg-lib")


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


def label_offline(prep, nt, node_id):
    return int(node_id) if nt == "transaction" else prep.entity_index[nt][int(node_id)]


@needs_temporal
def test_index_subgraph_equals_the_neighbor_loader_neighbourhood(prep):
    df, p = prep.df, len(prep.df) // 2
    rows = np.arange(p, p + 40)
    batch = next(iter(make_loader(prep, "transaction", rows, len(rows), False)))
    offline = {i: set() for i in range(len(rows))}
    for et in batch.edge_types:
        s, _, d = et
        ei = batch[et].edge_index
        for seed, a, b in zip(batch[d].batch[ei[1]].tolist(), batch[s].n_id[ei[0]].tolist(), batch[d].n_id[ei[1]].tolist()):
            offline[seed].add((et, label_offline(prep, s, a), label_offline(prep, d, b)))
    index = GraphIndex.from_prepared(prep, p)
    n_checked = 0
    for j, row in enumerate(rows):
        keys = (df["user"].iat[row], df["email"].iat[row], df["device"].iat[row])
        sg = index.subgraph(keys, int(df["TransactionDT"].iat[row]), prep.x[row])
        online = set()
        for et, pairs in sg.edges.items():
            for a, b in pairs:
                lab = lambda nt, i: row if nt == "transaction" and sg.tx_rows[i] == -1 else (
                    sg.tx_rows[i] if nt == "transaction" else sg.entity_keys[nt][i])
                online.add((et, lab(et[0], a), lab(et[2], b)))
        assert online == offline[j], f"row {row}: only online {online - offline[j]}, only offline {offline[j] - online}"
        n_checked += len(online)
        index.add(int(df["TransactionDT"].iat[row]), float(df["TransactionAmt"].iat[row]), keys, prep.x[row])
    assert n_checked > 100


@needs_temporal
def test_online_scores_equal_offline_scores_under_the_last_rule(prep):
    p = len(prep.df) // 2
    rows = np.arange(p, p + 60)
    scorer = make_scorer(prep, p)
    online = np.array(scorer.score_batch([payload_of(prep.df, int(i)) for i in rows]))
    offline = predict_scores(prep, scorer.model, scorer.heads, "fraud", rows, "cpu")
    np.testing.assert_allclose(online, offline, atol=1e-4)


def test_batch_scoring_equals_one_by_one(prep):
    p = len(prep.df) // 2
    payloads = [payload_of(prep.df, i) for i in range(p, p + 20)]
    batch = make_scorer(prep, p).score_batch(payloads)
    single = make_scorer(prep, p)
    assert np.allclose(batch, [single.score(x) for x in payloads], atol=1e-5)


def test_payload_order_does_not_change_which_score_goes_with_which_request(prep):
    p = len(prep.df) // 2
    payloads = [payload_of(prep.df, i) for i in range(p, p + 10)]
    forward = make_scorer(prep, p).score_batch(payloads)
    shuffled = make_scorer(prep, p).score_batch(payloads[::-1])
    assert np.allclose(forward, shuffled[::-1], atol=1e-5)


def test_payload_number_types_do_not_change_the_entity_keys(prep):
    scorer = make_scorer(prep, 100)
    pay = payload_of(prep.df, 150)
    variants = [pay, {**pay, "card1": float(pay["card1"])}, {**pay, "addr1": None if pay["addr1"] is None else int(pay["addr1"])}]
    users = [scorer._frame([v])["user"].iat[0] for v in variants]
    assert len(set(users)) == 1


def test_a_card_never_seen_with_no_email_or_device_scores_cleanly(prep):
    scorer = make_scorer(prep, len(prep.df) // 2)
    pay = {"TransactionDT": int(prep.df["TransactionDT"].max()) + 10, "TransactionAmt": 25.0, "ProductCD": "W", "card1": 999999}
    assert 0.0 <= scorer.score(pay) <= 1.0


def test_old_requests_are_rejected_without_changing_the_index(prep):
    p = len(prep.df) // 2
    scorer = make_scorer(prep, p)
    n = len(scorer.index)
    with pytest.raises(OutOfOrderError):
        scorer.score(payload_of(prep.df, 0))
    with pytest.raises(OutOfOrderError):
        scorer.score_batch([payload_of(prep.df, p + 5), payload_of(prep.df, 0)])
    assert len(scorer.index) == n


def test_late_requests_can_be_clamped_to_the_newest_time(prep):
    scorer = make_scorer(prep, len(prep.df) // 2, late="clamp")
    before = scorer.index.last_time
    assert 0.0 <= scorer.score(payload_of(prep.df, 0)) <= 1.0
    assert scorer.index.last_time == before


def test_missing_required_fields_are_named(prep):
    scorer = make_scorer(prep, 100)
    with pytest.raises(ValueError, match="TransactionDT"):
        scorer.score({"TransactionAmt": 5.0})
    with pytest.raises(ValueError, match="numbers"):
        scorer.score({"TransactionDT": "soon", "TransactionAmt": 5.0})


def test_tabular_online_history_matches_the_offline_history_features(prep):
    p = len(prep.df) // 2
    index = GraphIndex.from_prepared(prep, p)
    df = prep.df
    for j in range(p, p + 80):
        keys = (df["user"].iat[j], df["email"].iat[j], df["device"].iat[j])
        got = pd.Series(index.history(keys, int(df["TransactionDT"].iat[j])))[prep.hist.columns]
        np.testing.assert_allclose(got.to_numpy(float), prep.hist.iloc[j].to_numpy(float), rtol=1e-6, equal_nan=True)
        index.add(int(df["TransactionDT"].iat[j]), float(df["TransactionAmt"].iat[j]), keys, prep.x[j])


def test_tabular_scorer_matches_the_offline_baseline_predictions(prep):
    from scripts.baseline import fit_hgb

    p = len(prep.df) // 2
    X = np.hstack([prep.x, prep.hist.to_numpy(np.float32)])
    model = fit_hgb(X[:p], prep.df["isFraud"].to_numpy()[:p], seed=0, max_iter=20)
    scorer = TabularOnlineScorer(model, prep.transformer, GraphIndex.from_prepared(prep, p), raw_kinds_of(prep.df),
                                 hist_columns=list(prep.hist.columns))
    rows = list(range(p, p + 40))
    online = scorer.score_batch([payload_of(prep.df, i) for i in rows])
    np.testing.assert_allclose(online, model.predict_proba(X[rows])[:, 1], atol=1e-5)
```

`tests/test_spike_serving.py`:

```python
from scripts.spike_serving import time_index, time_stock_loader
from tests.serving_helpers import HAS_TEMPORAL, last_rule_prepared

import numpy as np
import pytest


@pytest.mark.skipif(not HAS_TEMPORAL, reason="the stock loader path needs pyg-lib")
def test_both_timers_return_one_measurement_per_request():
    prep = last_rule_prepared()
    rows = np.arange(len(prep.df) // 2, len(prep.df) // 2 + 5)
    assert len(time_stock_loader(prep, rows)) == 5 and len(time_index(prep, rows)) == 5
```

- [ ] **Step 6: Run to verify failure**

Run: `"$PY" -m pytest tests/test_scorer_parity.py tests/test_spike_serving.py -q`
Expected: `ModuleNotFoundError: No module named 'serving.scorer'`.

- [ ] **Step 7: Implement the scorers** (`serving/scorer.py`)

```python
"""Online scorers. A request becomes features and a sampled neighbourhood exactly as in training, then the model runs."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from pipeline.data_loading import finalize
from serving.graph_index import EDGES, ENTITIES, OutOfOrderError, keys_from_frame

REQUIRED = ("TransactionDT", "TransactionAmt")
DERIVED = ("user", "email", "device")
LABEL = "isFraud"
ALL_EDGE_TYPES = [et for pair in EDGES.values() for et in pair]


def raw_kinds_of(df: pd.DataFrame) -> dict:
    kinds = {}
    for c in df.columns:
        if c in DERIVED:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            kinds[c] = "float"
        elif pd.api.types.is_integer_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            kinds[c] = "int"
        else:
            kinds[c] = "object"
    return kinds


def _py(v):
    if isinstance(v, (list, tuple, dict)):
        return v
    if v is None or pd.isna(v):
        return None
    return v.item() if isinstance(v, np.generic) else v


def payload_of(df: pd.DataFrame, i: int) -> dict:
    """A request body for row `i`: raw fields only (no label, no derived keys), JSON-safe."""
    row = df.iloc[i]
    return {c: _py(row[c]) for c in df.columns if c not in DERIVED and c != LABEL}


def build_frame(payloads, kinds) -> pd.DataFrame:
    for p in payloads:
        missing = [c for c in REQUIRED if p.get(c) is None]
        if missing:
            raise ValueError(f"missing required field(s): {missing}")
    frame = pd.DataFrame({c: [p.get(c) for p in payloads] for c in kinds})
    for c, kind in kinds.items():
        if kind == "object":
            frame[c] = frame[c].astype(object)
        else:
            s = pd.to_numeric(frame[c], errors="coerce")
            frame[c] = s.astype("int64") if kind == "int" and not s.isna().any() else s.astype("float64")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("TransactionDT and TransactionAmt must be numbers")
    if LABEL in frame:
        frame[LABEL] = 0                                   # unknown at scoring time and never used by the features
    frame["_pos"] = np.arange(len(frame))
    return finalize(frame)


def collate(subs, device="cpu"):
    """Merge per-request subgraphs into one disjoint batch (what the training-time loader produces)."""
    offsets = {nt: 0 for nt in ("transaction",) + ENTITIES}
    xs, seeds = [], []
    src = {et: [] for et in ALL_EDGE_TYPES}
    dst = {et: [] for et in ALL_EDGE_TYPES}
    for s in subs:
        seeds.append(offsets["transaction"])
        xs.append(s.x)
        for et, pairs in s.edges.items():
            for a, b in pairs:
                src[et].append(a + offsets[et[0]])
                dst[et].append(b + offsets[et[2]])
        for nt, n in s.num_nodes.items():
            offsets[nt] += n
    edge_index = {et: torch.tensor([src[et], dst[et]], dtype=torch.long, device=device) for et in ALL_EDGE_TYPES}
    x_dict = {"transaction": torch.from_numpy(np.concatenate(xs)).to(device)}
    return x_dict, edge_index, dict(offsets), seeds


class _BaseScorer:
    def __init__(self, transformer, index, raw_kinds, version, late, info):
        if late not in ("reject", "clamp"):
            raise ValueError("late must be 'reject' or 'clamp'")
        self.transformer, self.index, self.raw_kinds = transformer, index, raw_kinds
        self.version, self.late, self.info = version, late, info or {}

    def _frame(self, payloads):
        return build_frame(payloads, self.raw_kinds)

    def _times(self, frame):
        times = frame["TransactionDT"].to_numpy(np.int64)
        if times.min() < self.index.last_time:
            if self.late == "reject":
                raise OutOfOrderError(f"request time {int(times.min())} is older than the newest indexed time "
                                      f"{int(self.index.last_time)}")
            times = np.maximum(times, int(self.index.last_time))   # late events are treated as arriving now
        return times

    def score(self, payload) -> float:
        return self.score_batch([payload])[0]


def _lap(timings, name, t0):
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0


class OnlineScorer(_BaseScorer):
    def __init__(self, model, heads, transformer, index, raw_kinds, device="cpu", version="dev", late="reject", info=None):
        super().__init__(transformer, index, raw_kinds, version, late, info)
        self.model, self.heads, self.device = model.to(device).eval(), heads.to(device).eval(), device

    def score_batch(self, payloads, timings=None):
        if not payloads:
            return []
        tm = timings if timings is not None else {}
        t0 = time.perf_counter()
        frame = self._frame(payloads)
        x = self.transformer.transform(frame)
        _lap(tm, "feature_transform", t0)
        times, amounts = self._times(frame), frame["TransactionAmt"].to_numpy(float)
        subs = []
        for i in range(len(frame)):
            keys = keys_from_frame(frame, i)
            t0 = time.perf_counter()
            subs.append(self.index.subgraph(keys, int(times[i]), x[i]))
            _lap(tm, "neighbour_lookup", t0)
            t0 = time.perf_counter()
            self.index.add(int(times[i]), float(amounts[i]), keys, x[i])
            _lap(tm, "index_update", t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            x_dict, edge_index, num_nodes, seeds = collate(subs, self.device)
            h = self.model(x_dict, edge_index, num_nodes)
            prob = torch.sigmoid(self.heads.fraud_logits(h["transaction"][torch.tensor(seeds, device=self.device)]))
            prob = prob.cpu().numpy()                      # also synchronises a GPU, so the lap is honest
        _lap(tm, "model_forward", t0)
        out = np.empty(len(frame))
        out[frame["_pos"].to_numpy()] = prob
        return out.tolist()


class TabularOnlineScorer(_BaseScorer):
    """The tabular baseline under the same serving protocol; `hist_columns=None` is the naive model."""

    def __init__(self, model, transformer, index, raw_kinds, hist_columns=None, version="dev", late="reject", info=None):
        super().__init__(transformer, index, raw_kinds, version, late, info)
        self.model, self.hist_columns = model, hist_columns

    def score_batch(self, payloads, timings=None):
        if not payloads:
            return []
        tm = timings if timings is not None else {}
        t0 = time.perf_counter()
        frame = self._frame(payloads)
        x = self.transformer.transform(frame)
        _lap(tm, "feature_transform", t0)
        times, amounts = self._times(frame), frame["TransactionAmt"].to_numpy(float)
        rows = []
        for i in range(len(frame)):
            keys = keys_from_frame(frame, i)
            t0 = time.perf_counter()
            if self.hist_columns is not None:
                h = self.index.history(keys, int(times[i]))
                rows.append(np.hstack([x[i], np.array([h[c] for c in self.hist_columns], dtype=np.float32)]))
            else:
                rows.append(x[i])
            _lap(tm, "neighbour_lookup", t0)
            t0 = time.perf_counter()
            self.index.add(int(times[i]), float(amounts[i]), keys, x[i])
            _lap(tm, "index_update", t0)
        t0 = time.perf_counter()
        prob = self.model.predict_proba(np.stack(rows))[:, 1]
        _lap(tm, "model_forward", t0)
        out = np.empty(len(frame))
        out[frame["_pos"].to_numpy()] = prob
        return out.tolist()
```
For the tabular scorer the `neighbour_lookup` lap is the history-feature computation; the benchmark labels it accordingly.

- [ ] **Step 8: Run to verify pass**

Run: `"$PY" -m pytest tests/test_scorer_parity.py -q`
Expected: all pass. If `test_index_subgraph_equals_the_neighbor_loader_neighbourhood` fails, that is the spike's job: the printed message shows the edges only one side has. Fix `GraphIndex.subgraph` to match the observed PyG behaviour (check, in this order: whether the seed itself is returned as its own neighbour, how `"last"` orders ties, whether hop-two neighbours of the seed's entities include the seed's other entity edges), record what differed in the ledger, and never loosen the test. If the scores differ while the edge sets match, the cause is in `collate` or `OnlineScorer` (check empty edge types and node counts first).

- [ ] **Step 9: Write the serving spike** (`scripts/spike_serving.py`)

```python
"""Serving spike: what does the stock NeighborLoader cost per single request, against the custom index?"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pipeline.config import PROJECT_ROOT
from scripts.train import make_loader
from serving.graph_index import GraphIndex


def time_stock_loader(prep, rows) -> list:
    """Milliseconds per request when each request builds a batch-size-1 loader (what the stock path needs online)."""
    out = []
    for r in rows:
        t0 = time.perf_counter()
        next(iter(make_loader(prep, "transaction", np.array([int(r)]), 1, False)))
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def time_index(prep, rows) -> list:
    df = prep.df
    index = GraphIndex.from_prepared(prep, int(rows[0]))
    out = []
    for r in rows:
        r = int(r)
        keys = (df["user"].iat[r], df["email"].iat[r], df["device"].iat[r])
        t0 = time.perf_counter()
        index.subgraph(keys, int(df["TransactionDT"].iat[r]), prep.x[r])
        out.append((time.perf_counter() - t0) * 1e3)
        index.add(int(df["TransactionDT"].iat[r]), float(df["TransactionAmt"].iat[r]), keys, prep.x[r])
    return out


def main(argv=None):
    from pipeline.config import load_config
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Compare per-request neighbourhood cost: stock loader vs custom index.")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        cfg = load_config(overrides={**fixture_overrides(), "train": {**fixture_overrides()["train"], "temporal_strategy": "last"}})
        prep = prepare(cfg, df=make_fixture()[0])
    else:
        prep = prepare(load_config(overrides={"train": {"temporal_strategy": "last"}}))
    rows = np.arange(int(len(prep.df) * 0.85), int(len(prep.df) * 0.85) + a.n)
    stock, custom = np.array(time_stock_loader(prep, rows)), np.array(time_index(prep, rows))
    text = ("# Serving spike\n\n" + ("> Fixture data: indicative only.\n\n" if a.fixture else "")
            + "Per-request cost of building the sampled neighbourhood, milliseconds, one machine, CPU:\n\n"
            "| method | p50 | p95 |\n|---|---|---|\n"
            f"| stock `NeighborLoader`, batch size 1 (loader built per request) | {np.percentile(stock, 50):.2f} | {np.percentile(stock, 95):.2f} |\n"
            f"| custom time-sorted index | {np.percentile(custom, 50):.3f} | {np.percentile(custom, 95):.3f} |\n\n"
            "Decision: serve with the custom index regardless of the gap. The serving image then needs no sampler "
            "dependency (pyg-lib), the index is append-only so new transactions become neighbours immediately, and the "
            "parity tests pin it to the stock sampler's `\"last\"` rule.\n")
    path = PROJECT_ROOT / "docs" / "serving-spike.md"
    path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run the spike tests, run the spike, and commit**

Run: `"$PY" -m pytest tests/test_spike_serving.py tests/test_graph_index.py tests/test_scorer_parity.py -q` then `"$PY" -m scripts.spike_serving --fixture`
Expected: tests pass; the spike prints a table and writes `docs/serving-spike.md` (labelled fixture). The real-data spike runs in Task 22.

```bash
git add serving scripts/spike_serving.py tests/serving_helpers.py tests/test_graph_index.py tests/test_scorer_parity.py tests/test_spike_serving.py docs/serving-spike.md
git commit -m "feat: add the temporal graph index, online scorers and the train/serve parity gate" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 20: Artifact export, scoring API and container

**Files:**
- Create: `serving/bundle.py`, `serving/app.py`, `scripts/export_artifacts.py`, `scripts/smoke_service.py`, `Dockerfile`, `.dockerignore`, `requirements-serving.txt` (generated), `tests/test_export.py`, `tests/test_api.py`, `tests/test_docker_files.py`
- Modify: `scripts/make_ci_requirements.py` (add `serving_requirements` and write `requirements-serving.txt`), `tests/test_ci_config.py`, `requirements.txt`, `requirements-ci.txt` (regenerated)

**Interfaces:**
- Consumes: Task 19, `load_checkpoint`, `EDGE_TYPES`.
- Produces:
  - `serving.bundle`: `save_bundle(out_dir, model_cfg, model, heads, transformer, index, raw_kinds, hist_columns, meta, samples=None)`, `load_bundle(out_dir, device="cpu", late="reject") -> OnlineScorer`. Files: `weights.pt`, `bundle.pkl`, `meta.json`, `sample_requests.json`. The pickle is only for artifacts you built yourself; never load a bundle from an untrusted source.
  - `scripts.export_artifacts.export_artifacts(prepared, checkpoint, out_dir, history_until="test_start", device="cpu", fixture=False, n_samples=50) -> dict[str, Path]` and CLI `--config --checkpoint --out artifacts --history-until test_start|val_end|all --fixture`. `--fixture` trains a one-epoch `multitask` model on the fixture itself so CI needs no checkpoint. Refuses a checkpoint whose variant has no fraud head (`account_only`).
  - `serving.app.create_app(scorer=None, artifacts=None) -> FastAPI` and module-level `app`. Endpoints: `GET /health` (`status`, `indexed_transactions`, `model_version`), `GET /version` (the bundle's `meta.json` content), `POST /score` body = a transaction object (required `TransactionDT` int >= 0 and `TransactionAmt` float >= 0, any other raw column optional) -> `{"fraud_probability", "model_version"}`, `POST /score_batch` body `{"transactions": [1..256 items]}` -> `{"fraud_probabilities", "model_version"}`. Errors: 422 malformed or invalid, 409 request older than the index, 503 model not loaded. One lock serialises scoring because the index is mutable; concurrency does not scale scoring and the README says so.
  - `scripts.smoke_service.smoke(client, payloads) -> dict` (works with `fastapi.testclient.TestClient` and `httpx.Client`) and CLI `--url --samples --wait`.

- [ ] **Step 1: Install and pin the serving dependencies**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" -m pip install fastapi uvicorn httpx
```
Re-run the pinning script from Task 1 Step 3 with `"fastapi", "uvicorn", "httpx", "pydantic", "starlette"` appended to `names`, then `"$PY" -m scripts.make_ci_requirements` (updated in Step 5) to regenerate `requirements-ci.txt` and `requirements-serving.txt`.

- [ ] **Step 2: Write the failing tests**

`tests/test_export.py`:

```python
import json

import numpy as np
import pytest
import torch

from scripts.export_artifacts import export_artifacts
from scripts.train import default_spec, load_checkpoint, train_variant
from serving.bundle import load_bundle
from serving.graph_index import GraphIndex
from serving.scorer import OnlineScorer, raw_kinds_of
from tests.serving_helpers import last_rule_prepared


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


@pytest.fixture(scope="module")
def exported(prep, tmp_path_factory):
    d = tmp_path_factory.mktemp("export")
    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", d / "m.pt")
    train_variant(prep, default_spec(prep, "account_only", 0), "cpu", d / "acct.pt")
    return d, export_artifacts(prep, d / "m.pt", d / "artifacts", fixture=True)


def test_export_writes_every_file_and_meta(exported):
    d, paths = exported
    for key in ("weights", "bundle", "meta", "samples"):
        assert paths[key].exists(), key
    meta = json.loads(paths["meta"].read_text())
    assert meta["fixture"] is True and meta["variant"] == "fraud_only" and meta["config_hash"] and meta["version"]


def test_loaded_bundle_scores_like_the_in_memory_model(prep, exported):
    d, paths = exported
    samples = json.loads(paths["samples"].read_text())
    assert len(samples) > 5
    loaded = load_bundle(d / "artifacts")
    model, heads, _ = load_checkpoint(d / "m.pt", prep, "cpu")
    upto = int((prep.df["TransactionDT"] < prep.periods.test_start).sum())
    direct = OnlineScorer(model, heads, prep.transformer, GraphIndex.from_prepared(prep, upto), raw_kinds_of(prep.df))
    assert np.allclose(loaded.score_batch(samples), direct.score_batch(samples), atol=1e-5)


def test_bundle_weights_load_without_pickle_execution(exported):
    d, paths = exported
    state = torch.load(paths["weights"], map_location="cpu", weights_only=True)
    assert set(state) == {"model", "heads"}


def test_a_checkpoint_without_a_fraud_head_is_refused(prep, exported):
    d, _ = exported
    with pytest.raises(ValueError, match="fraud"):
        export_artifacts(prep, d / "acct.pt", d / "bad", fixture=True)
```

`tests/test_api.py`:

```python
import threading

import pytest
from fastapi.testclient import TestClient

from serving.app import create_app
from serving.scorer import payload_of
from tests.serving_helpers import last_rule_prepared, make_scorer


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


@pytest.fixture()
def client_and_payloads(prep):
    p = len(prep.df) // 2
    scorer = make_scorer(prep, p, version="test-v1")
    return TestClient(create_app(scorer)), [payload_of(prep.df, i) for i in range(p, p + 40)]


def test_health_and_version(client_and_payloads):
    client, _ = client_and_payloads
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/version").json()["model_version"] == "test-v1"


def test_score_returns_a_probability_and_grows_the_index(client_and_payloads):
    client, payloads = client_and_payloads
    n0 = client.get("/health").json()["indexed_transactions"]
    r = client.post("/score", json=payloads[0])
    assert r.status_code == 200 and 0.0 <= r.json()["fraud_probability"] <= 1.0 and r.json()["model_version"] == "test-v1"
    assert client.get("/health").json()["indexed_transactions"] == n0 + 1


def test_batch_returns_one_probability_per_transaction(client_and_payloads):
    client, payloads = client_and_payloads
    r = client.post("/score_batch", json={"transactions": payloads[:8]})
    assert r.status_code == 200 and len(r.json()["fraud_probabilities"]) == 8


@pytest.mark.parametrize("body", [
    {"TransactionAmt": 5.0},                                  # missing time
    {"TransactionDT": 100},                                    # missing amount
    {"TransactionDT": 100, "TransactionAmt": -1.0},            # negative amount
    {"TransactionDT": "soon", "TransactionAmt": 5.0},          # wrong type
    {},
])
def test_malformed_transactions_get_422(client_and_payloads, body):
    client, _ = client_and_payloads
    assert client.post("/score", json=body).status_code == 422


@pytest.mark.parametrize("body", [{"transactions": []}, {"transactions": [{"TransactionDT": 1, "TransactionAmt": 1.0}] * 257}, {}])
def test_bad_batches_get_422(client_and_payloads, body):
    client, _ = client_and_payloads
    assert client.post("/score_batch", json=body).status_code == 422


def test_unknown_extra_fields_are_ignored(client_and_payloads):
    client, payloads = client_and_payloads
    assert client.post("/score", json={**payloads[0], "not_a_column": 1}).status_code == 200


def test_old_requests_get_409_and_do_not_change_the_index(client_and_payloads, prep):
    client, payloads = client_and_payloads
    n0 = client.get("/health").json()["indexed_transactions"]
    r = client.post("/score", json=payload_of(prep.df, 0))
    assert r.status_code == 409 and "older" in r.json()["detail"]
    assert client.get("/health").json()["indexed_transactions"] == n0


def test_service_without_a_model_returns_503():
    client = TestClient(create_app(scorer=None))              # no lifespan: nothing is loaded
    assert client.get("/health").status_code == 503
    assert client.post("/score", json={"TransactionDT": 1, "TransactionAmt": 1.0}).status_code == 503


def test_concurrent_requests_are_serialised_safely(client_and_payloads):
    client, payloads = client_and_payloads
    n0 = client.get("/health").json()["indexed_transactions"]
    same_time = [{**payloads[-1]} for _ in range(16)]         # equal times never count as out of order
    codes = []
    threads = [threading.Thread(target=lambda b=b: codes.append(client.post("/score", json=b).status_code)) for b in same_time]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes == [200] * 16 and client.get("/health").json()["indexed_transactions"] == n0 + 16
```

`tests/test_docker_files.py`:

```python
from pipeline.config import PROJECT_ROOT


def read(name):
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_is_cpu_only_and_ships_no_data():
    d = read("Dockerfile")
    assert d.startswith("FROM python:3.14-slim") and "uvicorn" in d and "serving.app:app" in d
    assert "COPY data" not in d and "cu130" not in d and "whl/cpu" in d


def test_dockerignore_keeps_data_and_models_out_of_the_image():
    ignore = read(".dockerignore").split()
    assert {"data", "models/*.pt", "outputs", "notebooks", "tests", ".git"} <= set(ignore)   # models/ is also a package


def test_serving_requirements_are_minimal():
    req = read("requirements-serving.txt")
    for needed in ("fastapi", "uvicorn", "torch-geometric", "numpy", "pandas"):
        assert needed in req
    for banned in ("pytest", "jupyter", "ipykernel", "scikit-learn", "matplotlib", "cu130"):
        assert banned not in req
```

Extend `tests/test_ci_config.py`:

```python
from scripts.make_ci_requirements import serving_requirements


def test_serving_requirements_keep_only_what_the_service_imports():
    lines = ["numpy==2.3.0", "pandas==2.3.1", "pytest==8.4.0", "scikit-learn==1.7.0", "fastapi==0.115.0",
             "uvicorn==0.34.0", "torch-geometric==2.8.1", "PyYAML==6.0.2", "matplotlib==3.10.0"]
    out = serving_requirements(lines)
    assert set(out) == {"numpy==2.3.0", "pandas==2.3.1", "fastapi==0.115.0", "uvicorn==0.34.0",
                        "torch-geometric==2.8.1", "PyYAML==6.0.2"}
```

`tests/test_smoke_service.py`:

```python
import json

import pytest
from fastapi.testclient import TestClient

from scripts.smoke_service import smoke
from serving.app import create_app
from serving.scorer import payload_of
from tests.serving_helpers import last_rule_prepared, make_scorer


def test_smoke_passes_against_a_healthy_service_and_fails_against_a_broken_one():
    prep = last_rule_prepared()
    p = len(prep.df) // 2
    payloads = [payload_of(prep.df, i) for i in range(p, p + 15)]
    result = smoke(TestClient(create_app(make_scorer(prep, p))), payloads)
    assert result["scored"] >= 14 and result["malformed_rejected"] and result["late_rejected"]
    with pytest.raises(AssertionError):
        smoke(TestClient(create_app(scorer=None)), payloads)
```

- [ ] **Step 3: Run to verify failure**

Run: `"$PY" -m pytest tests/test_export.py tests/test_api.py tests/test_docker_files.py tests/test_smoke_service.py -q`
Expected: import errors for `serving.app`, `serving.bundle`, `scripts.export_artifacts`, `scripts.smoke_service`.

- [ ] **Step 4: Implement the bundle, exporter, API and smoke script**

`serving/bundle.py`:

```python
"""Save and load everything the scoring service needs. The pickle is for artifacts you built yourself only."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from models.gnn_backbone import HeteroGNN
from models.multi_task_heads import TaskHeads
from serving.graph_index import GraphIndex
from serving.scorer import OnlineScorer, collate

WEIGHTS, BUNDLE, META, SAMPLES = "weights.pt", "bundle.pkl", "meta.json", "sample_requests.json"


def save_bundle(out_dir, model_cfg, model, heads, transformer, index, raw_kinds, hist_columns, meta, samples=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, out / WEIGHTS)
    with open(out / BUNDLE, "wb") as f:
        pickle.dump({"model_cfg": model_cfg, "transformer": transformer, "index": index, "raw_kinds": raw_kinds,
                     "hist_columns": list(hist_columns)}, f)
    (out / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if samples is not None:
        (out / SAMPLES).write_text(json.dumps(samples), encoding="utf-8")
    return {"weights": out / WEIGHTS, "bundle": out / BUNDLE, "meta": out / META, "samples": out / SAMPLES}


def _warm(model, heads, n_features):
    """Lazy PyG layers need one forward pass over every edge type before weights can be loaded."""
    index = GraphIndex(n_features, (2, 2))
    index.add(0, 1.0, ("u", "e", "d"), np.zeros(n_features, np.float32))
    sg = index.subgraph(("u", "e", "d"), 1, np.zeros(n_features, np.float32))
    with torch.no_grad():
        model(*collate([sg])[:3])


def load_bundle(out_dir, device="cpu", late="reject") -> OnlineScorer:
    out = Path(out_dir)
    with open(out / BUNDLE, "rb") as f:
        b = pickle.load(f)
    cfg = b["model_cfg"]
    model, heads = HeteroGNN(**cfg), TaskHeads(cfg["hidden"], cfg["dropout"])
    _warm(model, heads, b["index"].n_features)
    weights = torch.load(out / WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(weights["model"])
    heads.load_state_dict(weights["heads"])
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    return OnlineScorer(model, heads, b["transformer"], b["index"], b["raw_kinds"], device, meta["version"], late, meta)
```

`scripts/export_artifacts.py`:

```python
"""Turn a trained checkpoint and the history before the test period into a self-contained scoring bundle."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np

from pipeline.config import PROJECT_ROOT, config_hash, load_config, resolve_path
from scripts.train import load_checkpoint
from serving.bundle import save_bundle
from serving.graph_index import GraphIndex
from serving.scorer import payload_of, raw_kinds_of


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def export_artifacts(prepared, checkpoint, out_dir, history_until="test_start", device="cpu", fixture=False, n_samples=50):
    model, heads, ckpt = load_checkpoint(checkpoint, prepared, device)
    if ckpt["variant"] not in ("multitask", "fraud_only"):
        raise ValueError(f"checkpoint variant {ckpt['variant']!r} has no fraud head to serve")
    cut = {"test_start": prepared.periods.test_start, "val_end": prepared.periods.val_end, "all": np.inf}[history_until]
    upto = int((prepared.df["TransactionDT"] < cut).sum())           # rows are sorted by time
    index = GraphIndex.from_prepared(prepared, upto)
    m = prepared.cfg["model"]
    model_cfg = dict(tx_in=prepared.x.shape[1], edge_types=list(prepared.graph.edge_types), hidden=m["hidden"],
                     heads=m["heads"], layers=m["layers"], dropout=m["dropout"], conv=m["conv"])
    chash = config_hash(prepared.cfg)
    meta = {"version": f"{ckpt['variant']}-seed{ckpt['seed']}-{chash}", "variant": ckpt["variant"], "seed": ckpt["seed"],
            "config_hash": chash, "git_commit": _git_commit(), "indexed_transactions": upto,
            "history_until": history_until, "fixture": bool(fixture),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    samples = [payload_of(prepared.df, i) for i in range(upto, min(upto + n_samples, len(prepared.df)))]
    return save_bundle(out_dir, model_cfg, model, heads, prepared.transformer, index, raw_kinds_of(prepared.df),
                       prepared.hist.columns, meta, samples)


def main(argv=None):
    from pipeline.prepare import prepare
    from scripts.train import default_spec, train_variant

    p = argparse.ArgumentParser(description="Export the scoring bundle.")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/multitask_seed0.pt")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--history-until", default="test_start", choices=["test_start", "val_end", "all"])
    p.add_argument("--fixture", action="store_true", help="train a one-epoch model on generated data (CI and demos)")
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        prepared = prepare(load_config(a.config, overrides=fixture_overrides()), df=make_fixture()[0])
        ckpt = resolve_path(a.out) / "fixture_checkpoint.pt"
        train_variant(prepared, default_spec(prepared, "multitask", 0), "cpu", ckpt)
    else:
        prepared, ckpt = prepare(load_config(a.config)), resolve_path(a.checkpoint)
    paths = export_artifacts(prepared, ckpt, resolve_path(a.out), a.history_until, fixture=a.fixture)
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
```
The fixture checkpoint is written inside the output folder and stays out of the image because the Dockerfile copies only the bundle files (Step 5).

`serving/app.py`:

```python
"""FastAPI scoring service. One lock serialises scoring: the neighbour index is mutable and order-sensitive."""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from serving.graph_index import OutOfOrderError


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")                 # any other raw column is optional and passed through
    TransactionDT: int = Field(ge=0, description="seconds on the dataset's clock; must not go backwards")
    TransactionAmt: float = Field(ge=0)


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=256)


def create_app(scorer=None, artifacts=None) -> FastAPI:
    state = {"scorer": scorer}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if state["scorer"] is None:
            from serving.bundle import load_bundle

            state["scorer"] = load_bundle(artifacts or os.environ.get("URE_ARTIFACTS", "artifacts"))
        yield

    app = FastAPI(title="Unified risk engine scorer", lifespan=lifespan)

    def current():
        if state["scorer"] is None:
            raise HTTPException(503, "model not loaded")
        return state["scorer"]

    def run(payloads):
        s = current()
        try:
            with lock:
                return s.score_batch(payloads), s.version
        except OutOfOrderError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/health")
    def health():
        s = current()
        return {"status": "ok", "indexed_transactions": len(s.index), "model_version": s.version}

    @app.get("/version")
    def version():
        s = current()
        return {"model_version": s.version, **s.info}

    @app.post("/score")
    def score(txn: Transaction):
        probs, v = run([txn.model_dump()])
        return {"fraud_probability": probs[0], "model_version": v}

    @app.post("/score_batch")
    def score_batch(batch: Batch):
        probs, v = run([t.model_dump() for t in batch.transactions])
        return {"fraud_probabilities": probs, "model_version": v}

    return app


app = create_app()
```

`scripts/smoke_service.py`:

```python
"""Smoke test for a running scoring service (works with httpx.Client and FastAPI's TestClient)."""
from __future__ import annotations

import argparse
import json
import time


def smoke(client, payloads) -> dict:
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "ok", f"health: {health.status_code}"
    assert client.get("/version").json()["model_version"]
    scored = 0
    for body in payloads[:5]:
        r = client.post("/score", json=body)
        assert r.status_code == 200, f"/score returned {r.status_code}: {r.text}"
        assert 0.0 <= r.json()["fraud_probability"] <= 1.0
        scored += 1
    r = client.post("/score_batch", json={"transactions": payloads[5:15]})
    assert r.status_code == 200 and len(r.json()["fraud_probabilities"]) == len(payloads[5:15])
    scored += len(payloads[5:15])
    malformed = client.post("/score", json={"TransactionAmt": 1.0}).status_code == 422
    late = client.post("/score", json={**payloads[0], "TransactionDT": 0}).status_code == 409
    assert malformed, "malformed request was not rejected with 422"
    assert late, "request older than the index was not rejected with 409"
    return {"scored": scored, "malformed_rejected": malformed, "late_rejected": late}


def main(argv=None):
    import httpx

    p = argparse.ArgumentParser(description="Smoke-test a running scoring service.")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--samples", default="artifacts/sample_requests.json")
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /health")
    a = p.parse_args(argv)
    payloads = json.load(open(a.samples, encoding="utf-8"))
    with httpx.Client(base_url=a.url, timeout=30) as client:
        deadline = time.time() + a.wait
        while True:
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.time() > deadline:
                raise SystemExit(f"service at {a.url} did not become healthy within {a.wait}s")
            time.sleep(1)
        print(smoke(client, payloads))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Container files and the requirements generator**

`Dockerfile`:

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY pipeline ./pipeline
COPY models ./models
COPY serving ./serving
COPY configs ./configs
COPY artifacts/weights.pt artifacts/bundle.pkl artifacts/meta.json ./artifacts/
ENV URE_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore` (one entry per line):

```
.git
.github
data
models/*.pt
outputs
plots
reports
notebooks
tests
docs
scripts
```

Append to `scripts/make_ci_requirements.py` (before the `if __name__` block) and extend its `__main__` to write the serving file:

```python
SERVING_KEEP = {"numpy", "pandas", "pyyaml", "torch-geometric", "fastapi", "uvicorn"}   # pydantic comes with fastapi


def serving_requirements(lines):
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.split(r"[=<>~ ]", line, maxsplit=1)[0].lower() in SERVING_KEEP:
            out.append(re.sub(r"\+[A-Za-z0-9.]+", "", line))
    return out
```
and in `__main__`: `(root / "requirements-serving.txt").write_text("\n".join(serving_requirements(lines)) + "\n", encoding="utf-8")`. The serving image does not need pyg-lib or scikit-learn: the service never samples with PyG and it loads only the GNN. If `FeatureTransformer` or `finalize` import something outside these packages, the container smoke test in Task 21 fails loudly; add the package to `SERVING_KEEP` then.

- [ ] **Step 6: Run to verify pass**

Run: `"$PY" -m scripts.make_ci_requirements` then `"$PY" -m pytest tests/test_export.py tests/test_api.py tests/test_docker_files.py tests/test_smoke_service.py tests/test_ci_config.py -q`
Expected: all pass.

- [ ] **Step 7: Build and smoke-test the container locally if Docker is available**

```bash
docker --version
```
If Docker is installed:

```bash
"$PY" -m scripts.export_artifacts --fixture --out artifacts
docker build -t ure-scorer .
docker run -d --name ure-scorer -p 8000:8000 ure-scorer
"$PY" -m scripts.smoke_service --url http://localhost:8000 --samples artifacts/sample_requests.json
docker rm -f ure-scorer
```
Expected: the smoke script prints `{'scored': 15, 'malformed_rejected': True, 'late_rejected': True}`. If Docker is not installed, say so in the ledger; the CI container job (Task 21) is then the first real check.

- [ ] **Step 8: Commit**

```bash
git add serving/bundle.py serving/app.py scripts/export_artifacts.py scripts/smoke_service.py scripts/make_ci_requirements.py Dockerfile .dockerignore requirements.txt requirements-ci.txt requirements-serving.txt tests/test_export.py tests/test_api.py tests/test_docker_files.py tests/test_smoke_service.py tests/test_ci_config.py
git commit -m "feat: add the artifact export, FastAPI scoring service and CPU container" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 21: Latency benchmark, notebook 05 and the container CI job

**Files:**
- Create: `scripts/benchmark_latency.py`, `tests/test_benchmark.py`, `plots/latency.csv` and `plots/latency_meta.json` (generated; committed once real)
- Modify: `pipeline/notebook_support.py` (add `plot_latency`, `plot_breakdown`), `scripts/build_notebooks.py` (add notebook 05), `tests/test_notebook_support.py`, `tests/test_notebooks.py`, `.github/workflows/tests.yml`, `tests/test_ci_config.py`

**Interfaces:**
- Consumes: Tasks 19-20, `fit_hgb`, `load_checkpoint`, `time_stock_loader`.
- Produces in `scripts.benchmark_latency`:
  - `COMPONENTS = ("feature_transform", "neighbour_lookup", "model_forward", "index_update")`, `ParityError`.
  - `percentiles(ms) -> {"n","mean_ms","p50_ms","p95_ms","p99_ms","max_ms"}`.
  - `check_parity(prepared, make_scorer, n=200, atol=1e-4) -> {"checked", "max_abs_diff", "n_rows", "reason"}`: replays rows right after the index's history through a fresh scorer and compares with the offline `"last"`-rule scores (rows whose timestamp is shared with another transaction are skipped); raises `ParityError` above `atol`; returns `checked: False` with a reason when temporal sampling is unavailable.
  - `bench_inprocess(make_scorer, payloads, batch_size, warmup_batches=3) -> dict` and `bench_http(make_scorer, payloads, batch_size, concurrency, warmup_batches=3) -> dict` (a real uvicorn server on a free local port, one `httpx.Client` per client thread; non-200 responses abort the run).
  - `run_benchmark(prepared, checkpoint, out_dir, devices=None, batch_sizes=(1, 8, 32, 128), n_requests=2000, budget_ms=100.0, skip_parity=False, include_http=True, include_tabular=True, concurrent_clients=8, fixture=False, log=print) -> dict[str, Path]` writing `latency.csv` and `latency_meta.json`. Rows: `model` (`gnn`, `history_tabular`, `stock_neighbor_loader`), `device`, `mode` (`inprocess`, `http`, `lookup_only`), `batch_size`, `concurrency`, percentile columns, `throughput_rps`, `<component>_ms` per call, `p99_within_budget`.
  - A benchmark never reports before parity passes; with `skip_parity=True` the meta says `parity.checked == false`.
- Late policy: concurrent HTTP runs use `late="clamp"` (requests from different clients can arrive slightly out of order), recorded as `late_policy` in the rows' `notes`.

- [ ] **Step 1: Write the failing tests** (`tests/test_benchmark.py`)

```python
import json

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_latency import COMPONENTS, ParityError, check_parity, percentiles, run_benchmark
from scripts.train import default_spec, train_variant
from serving.scorer import OnlineScorer, payload_of
from tests.serving_helpers import HAS_TEMPORAL, last_rule_prepared, make_scorer


@pytest.fixture(scope="module")
def prep():
    return last_rule_prepared()


@pytest.fixture(scope="module")
def ckpt(prep, tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "m.pt"
    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", path)
    return path


def test_percentiles_by_hand():
    p = percentiles(list(range(1, 101)))
    assert p["n"] == 100 and p["p50_ms"] == pytest.approx(50.5) and p["p99_ms"] == pytest.approx(99.01)
    assert p["max_ms"] == 100 and p["p50_ms"] <= p["p95_ms"] <= p["p99_ms"] <= p["max_ms"]


def test_replay_payloads_survive_strict_json(prep):
    for i in range(0, len(prep.df), 500):
        json.dumps(payload_of(prep.df, i), allow_nan=False)


def test_benchmark_report_has_every_field_and_never_asserts_on_timing(prep, ckpt, tmp_path):
    paths = run_benchmark(prep, ckpt, tmp_path, devices=["cpu"], batch_sizes=(1, 4), n_requests=40, include_http=False,
                          fixture=True, log=lambda *_: None)
    df = pd.read_csv(paths["csv"])
    meta = json.loads(paths["meta"].read_text())
    assert {"gnn", "history_tabular"} <= set(df["model"])
    assert {"model", "device", "mode", "batch_size", "concurrency", "n", "p50_ms", "p95_ms", "p99_ms", "throughput_rps",
            "p99_within_budget"} <= set(df.columns)
    assert all(f"{c}_ms" in df.columns for c in COMPONENTS)
    inproc = df[df["mode"] == "inprocess"]
    assert (inproc["p50_ms"] <= inproc["p95_ms"]).all() and (inproc["p95_ms"] <= inproc["p99_ms"]).all()
    assert set(inproc["batch_size"]) == {1, 4}
    assert meta["budget_ms"] == 100.0 and meta["fixture"] is True and "hardware" in meta and "index_pickle_mb" in meta
    assert meta["parity"]["checked"] is HAS_TEMPORAL


@pytest.mark.skipif(not HAS_TEMPORAL, reason="parity needs temporal sampling")
def test_parity_passes_for_the_real_scorer_and_fails_for_a_skewed_one(prep):
    p = len(prep.df) // 2
    ok = check_parity(prep, lambda: make_scorer(prep, p), n=30)
    assert ok["checked"] and ok["max_abs_diff"] < 1e-4 and ok["n_rows"] > 0

    class Skewed(OnlineScorer):
        def score_batch(self, payloads, timings=None):
            return [min(1.0, s + 0.2) for s in super().score_batch(payloads, timings)]

    def skewed():
        s = make_scorer(prep, p)
        return Skewed(s.model, s.heads, s.transformer, s.index, s.raw_kinds)

    with pytest.raises(ParityError):
        check_parity(prep, skewed, n=30)


def test_parity_is_reported_as_unchecked_without_temporal_sampling(prep):
    if HAS_TEMPORAL:
        pytest.skip("temporal sampling is available here")
    assert check_parity(prep, lambda: make_scorer(prep, 100), n=10)["checked"] is False


@pytest.mark.slow
def test_http_benchmark_runs_end_to_end(prep, ckpt, tmp_path):
    paths = run_benchmark(prep, ckpt, tmp_path, devices=["cpu"], batch_sizes=(1,), n_requests=30, include_http=True,
                          concurrent_clients=2, include_tabular=False, skip_parity=True, fixture=True, log=lambda *_: None)
    df = pd.read_csv(paths["csv"])
    http = df[df["mode"] == "http"]
    assert set(http["concurrency"]) == {1, 2} and http["throughput_rps"].gt(0).all()
```

Extend `tests/test_notebook_support.py` (plots for the new notebook):

```python
def test_latency_plots_draw_something():
    df = pd.DataFrame({"model": ["gnn", "gnn", "history_tabular"], "device": "cpu", "mode": "inprocess",
                       "batch_size": [1, 8, 1], "concurrency": 1, "p50_ms": [3.0, 9.0, 1.0], "p95_ms": [4.0, 11.0, 2.0],
                       "p99_ms": [5.0, 12.0, 2.5], "feature_transform_ms": 0.5, "neighbour_lookup_ms": 0.1,
                       "model_forward_ms": 2.0, "index_update_ms": 0.05})
    for fig in (ns.plot_latency(df, budget_ms=100.0), ns.plot_breakdown(df)):
        assert len(fig.axes[0].patches) + len(fig.axes[0].lines) > 0
        plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_benchmark.py tests/test_notebook_support.py -q -m "not slow"`
Expected: `ModuleNotFoundError: No module named 'scripts.benchmark_latency'` and missing `plot_latency`.

- [ ] **Step 3: Implement the benchmark** (`scripts/benchmark_latency.py`)

```python
"""Latency and throughput of the online scorers: in process and over HTTP, with a component breakdown."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import platform
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.config import config_hash, resolve_path
from scripts.baseline import fit_hgb
from scripts.spike_serving import time_stock_loader
from scripts.train import load_checkpoint, predict_scores
from serving.graph_index import GraphIndex
from serving.scorer import OnlineScorer, TabularOnlineScorer, payload_of, raw_kinds_of

COMPONENTS = ("feature_transform", "neighbour_lookup", "model_forward", "index_update")


class ParityError(RuntimeError):
    """The online scorer disagrees with the offline batch scores: no latency number may be reported."""


def percentiles(ms) -> dict:
    a = np.asarray(ms, dtype=float)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


def _batches(payloads, size, warmup):
    batches = [payloads[i:i + size] for i in range(0, len(payloads), size)]
    batches = [b for b in batches if len(b) == size] or batches
    if len(batches) <= warmup:                                   # too little data to warm up and measure: measure everything
        return [], batches
    return batches[:warmup], batches[warmup:]


def check_parity(prepared, make_scorer, n=200, atol=1e-4) -> dict:
    cfg = copy.deepcopy(prepared.cfg)
    if not cfg["train"]["temporal"]:
        return {"checked": False, "max_abs_diff": None, "n_rows": 0, "reason": "temporal sampling is disabled in the config"}
    try:
        import pyg_lib  # noqa: F401
    except ImportError:
        return {"checked": False, "max_abs_diff": None, "n_rows": 0, "reason": "pyg-lib is not installed"}
    cfg["train"]["temporal_strategy"] = "last"                   # the rule the online index implements
    offline_prep = copy.copy(prepared)
    offline_prep.cfg = cfg
    scorer = make_scorer()
    start = len(scorer.index)
    rows = np.arange(start, min(start + 3 * n, len(prepared.df)))
    times = prepared.df["TransactionDT"].to_numpy()
    unique = pd.Series(times).map(pd.Series(times).value_counts()).to_numpy() == 1
    online = np.array(scorer.score_batch([payload_of(prepared.df, int(i)) for i in rows]))
    offline = predict_scores(offline_prep, scorer.model, scorer.heads, "fraud", rows, scorer.device)
    keep = unique[rows]
    diff = float(np.abs(online[keep][:n] - offline[keep][:n]).max()) if keep.any() else 0.0
    if diff > atol:
        raise ParityError(f"online and offline scores differ by {diff:.2e} (> {atol}); fix train/serve skew before benchmarking")
    return {"checked": True, "max_abs_diff": diff, "n_rows": int(min(keep.sum(), n)), "reason": ""}


def bench_inprocess(make_scorer, payloads, batch_size, warmup_batches=3) -> dict:
    scorer = make_scorer()
    warm, measured = _batches(payloads, batch_size, warmup_batches)
    for b in warm:
        scorer.score_batch(b)
    lat, comp, n_req = [], dict.fromkeys(COMPONENTS, 0.0), 0
    start = time.perf_counter()
    for b in measured:
        tm: dict = {}
        t0 = time.perf_counter()
        scorer.score_batch(b, timings=tm)
        lat.append((time.perf_counter() - t0) * 1e3)
        n_req += len(b)
        for c in COMPONENTS:
            comp[c] += tm.get(c, 0.0)
    wall = time.perf_counter() - start
    return {**percentiles(lat), "n_requests": n_req, "throughput_rps": n_req / wall,
            **{f"{c}_ms": comp[c] * 1e3 / len(lat) for c in COMPONENTS}}


def start_server(app, host="127.0.0.1", timeout=30.0):
    import uvicorn

    with socket.socket() as s:
        s.bind((host, 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("benchmark server did not start")
        time.sleep(0.05)
    return server, thread, f"http://{host}:{port}"


def bench_http(make_scorer, payloads, batch_size, concurrency, warmup_batches=3) -> dict:
    import httpx

    from serving.app import create_app

    scorer = make_scorer()
    server, thread, url = start_server(create_app(scorer))
    try:
        warm, measured = _batches(payloads, batch_size, warmup_batches)

        def send(client, b):
            if batch_size == 1:
                r = client.post("/score", json=b[0])
            else:
                r = client.post("/score_batch", json={"transactions": b})
            if r.status_code != 200:
                raise RuntimeError(f"benchmark request failed: {r.status_code} {r.text}")

        with httpx.Client(base_url=url, timeout=60) as client:
            for b in warm:
                send(client, b)

        def worker(chunk):
            lat = []
            with httpx.Client(base_url=url, timeout=60) as c:
                for b in chunk:
                    t0 = time.perf_counter()
                    send(c, b)
                    lat.append((time.perf_counter() - t0) * 1e3)
            return lat

        chunks = [measured[i::concurrency] for i in range(concurrency)]
        start = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            lat = [x for part in pool.map(worker, chunks) for x in part]
        wall = time.perf_counter() - start
        n_req = sum(len(b) for b in measured)
        return {**percentiles(lat), "n_requests": n_req, "throughput_rps": n_req / wall}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def hardware_info() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def run_benchmark(prepared, checkpoint, out_dir, devices=None, batch_sizes=(1, 8, 32, 128), n_requests=2000,
                  budget_ms=100.0, skip_parity=False, include_http=True, include_tabular=True, concurrent_clients=8,
                  fixture=False, log=print) -> dict:
    devices = devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, heads, ckpt = load_checkpoint(checkpoint, prepared, "cpu")
    df = prepared.df
    upto = int((df["TransactionDT"] < prepared.periods.test_start).sum())
    base_index = GraphIndex.from_prepared(prepared, upto)
    kinds = raw_kinds_of(df)
    payloads = [payload_of(df, i) for i in range(upto, min(upto + n_requests, len(df)))]

    def gnn_factory(device, late="reject"):
        return lambda: OnlineScorer(copy.deepcopy(model), copy.deepcopy(heads), prepared.transformer,
                                    copy.deepcopy(base_index), kinds, device=device, late=late)

    parity = {"checked": False, "max_abs_diff": None, "n_rows": 0, "reason": "skipped by request"}
    if not skip_parity:
        parity = check_parity(prepared, gnn_factory("cpu"))           # raises ParityError: nothing is written
        log(f"parity: {parity}")

    rows = []

    def add(model_name, device, mode, batch, conc, result, notes=""):
        rows.append({"model": model_name, "device": device, "mode": mode, "batch_size": batch, "concurrency": conc,
                     **result, "p99_within_budget": result.get("p99_ms", np.nan) <= budget_ms, "notes": notes})

    for device in devices:
        for b in batch_sizes:
            log(f"gnn {device} in-process batch={b}")
            add("gnn", device, "inprocess", b, 1, bench_inprocess(gnn_factory(device), payloads, b))
    if include_http:
        for b in sorted({1} | ({32} & set(batch_sizes))):
            log(f"gnn cpu http batch={b}")
            add("gnn", "cpu", "http", b, 1, bench_http(gnn_factory("cpu"), payloads, b, 1))
        log(f"gnn cpu http batch=1 clients={concurrent_clients}")
        add("gnn", "cpu", "http", 1, concurrent_clients,
            bench_http(gnn_factory("cpu", late="clamp"), payloads, 1, concurrent_clients),
            notes="late events clamped to the newest indexed time; scoring is serialised by one lock")
    if include_tabular:
        X = np.hstack([prepared.x, prepared.hist.to_numpy(np.float32)])
        train_rows = np.flatnonzero(prepared.graph["transaction"].train_mask.numpy())
        hgb = fit_hgb(X[train_rows], df["isFraud"].to_numpy()[train_rows], seed=0)

        def tab_factory(late="reject"):
            return lambda: TabularOnlineScorer(hgb, prepared.transformer, copy.deepcopy(base_index), kinds,
                                               hist_columns=list(prepared.hist.columns), late=late)

        for b in batch_sizes:
            log(f"history_tabular cpu in-process batch={b}")
            add("history_tabular", "cpu", "inprocess", b, 1, bench_inprocess(tab_factory(), payloads, b),
                notes="neighbour_lookup_ms is the history-feature computation")
        if include_http:
            add("history_tabular", "cpu", "http", 1, 1, bench_http(tab_factory(), payloads, 1, 1))
            add("history_tabular", "cpu", "http", 1, concurrent_clients,
                bench_http(tab_factory("clamp"), payloads, 1, concurrent_clients), notes="late events clamped")
    if parity["checked"]:
        sample = np.arange(upto, min(upto + 100, len(df)))
        add("stock_neighbor_loader", "cpu", "lookup_only", 1, 1, percentiles(time_stock_loader(prepared, sample)),
            notes="neighbourhood sampling only, loader built per request; no features, model or HTTP")
    table = pd.DataFrame(rows)
    for c in COMPONENTS:
        if f"{c}_ms" not in table:
            table[f"{c}_ms"] = np.nan
    meta = {"hardware": hardware_info(), "budget_ms": budget_ms, "budget_note": "illustrative assumption, not a requirement",
            "config_hash": config_hash(prepared.cfg), "fixture": bool(fixture), "parity": parity, "variant": ckpt["variant"],
            "indexed_transactions": upto, "index_pickle_mb": len(pickle.dumps(base_index)) / 1e6,
            "n_requests": len(payloads), "notes": "one machine replaying historical data; not a production service-level claim"}
    paths = {"csv": out_dir / "latency.csv", "meta": out_dir / "latency_meta.json"}
    table.to_csv(paths["csv"], index=False)
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def main(argv=None):
    from pipeline.config import load_config
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Benchmark online scoring latency.")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/fraud_only_seed0.pt")
    p.add_argument("--out", default="plots")
    p.add_argument("--n-requests", type=int, default=2000)
    p.add_argument("--budget-ms", type=float, default=100.0)
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args(argv)
    if a.fixture:
        from tests.fixtures import fixture_overrides, make_fixture

        cfg = load_config(a.config, overrides=fixture_overrides())
        prepared = prepare(cfg, df=make_fixture()[0])
    else:
        prepared = prepare(load_config(a.config))
    paths = run_benchmark(prepared, resolve_path(a.checkpoint), resolve_path(a.out), n_requests=a.n_requests,
                          budget_ms=a.budget_ms, fixture=a.fixture)
    print(pd.read_csv(paths["csv"])[["model", "device", "mode", "batch_size", "concurrency", "p50_ms", "p95_ms",
                                      "p99_ms", "throughput_rps"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
```
`check_parity` uses `scorer.device`; `OnlineScorer` stores `device`. The offline `predict_scores` is called with the copied `Prepared` so the shared prepared object's config is never mutated. The default checkpoint is a `fraud_only` run; point `--checkpoint` at a `multitask` one if that is the model to serve.

- [ ] **Step 4: Add the notebook plots and notebook 05**

Append to `pipeline/notebook_support.py`:

```python
def plot_latency(df: pd.DataFrame, budget_ms: float = 100.0):
    d = df[df["mode"].isin(["inprocess", "http"])].copy()
    d["label"] = (d["model"] + " " + d["device"] + " " + d["mode"] + " b" + d["batch_size"].astype(str)
                  + " c" + d["concurrency"].astype(str))
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.barh(y, d["p99_ms"], color="#e45756", label="p99")
    ax.barh(y, d["p95_ms"], color="#f58518", label="p95")
    ax.barh(y, d["p50_ms"], color="#4c78a8", label="p50")
    ax.axvline(budget_ms, color="grey", linestyle="--", label=f"illustrative budget {budget_ms:g} ms")
    ax.set_yticks(y, d["label"])
    ax.set_xlabel("milliseconds per call (log scale)"), ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_breakdown(df: pd.DataFrame):
    comps = ["feature_transform_ms", "neighbour_lookup_ms", "model_forward_ms", "index_update_ms"]
    d = df[(df["mode"] == "inprocess") & (df["batch_size"] == 1)].dropna(subset=comps)
    d = d.assign(label=d["model"] + " " + d["device"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    left = np.zeros(len(d))
    for c in comps:
        ax.barh(d["label"], d[c], left=left, label=c.replace("_ms", ""))
        left += d[c].to_numpy()
    ax.set_xlabel("milliseconds per request (batch size 1)")
    ax.legend()
    fig.tight_layout()
    return fig
```

In `scripts/build_notebooks.py` add (and bump the notebook count asserted in `tests/test_notebooks.py::test_every_notebook_cell_kind_is_valid` from 4 to 5):

```python
    "05_serving_latency.ipynb": [
        ("md", "# 5. Serving and latency\n\nThe model scored one transaction at a time by a service that keeps a time-sorted "
               "neighbour index. Numbers come from one machine replaying historical data; they are not a production claim, and "
               "the 100 ms budget is an illustrative assumption."),
        ("code", SETUP + "\nimport json\nlat = load_results('latency.csv')\nmeta = json.loads((out_dir() / 'latency_meta.json').read_text())\n"
                 "print('FIXTURE RESULTS - not findings' if meta['fixture'] else 'real data')\nprint('hardware:', meta['hardware'])\n"
                 "print('parity with the offline scores:', meta['parity'])"),
        ("code", "cols = ['model', 'device', 'mode', 'batch_size', 'concurrency', 'p50_ms', 'p95_ms', 'p99_ms', 'throughput_rps', 'p99_within_budget']\n"
                 "lat[cols].round(2)"),
        ("code", "plot_latency(lat, meta['budget_ms']);"),
        ("md", "## Where the time goes (batch size 1)"),
        ("code", "plot_breakdown(lat);"),
        ("md", "## Stock sampler versus the custom index\n\nThe `stock_neighbor_loader` row is the cost of sampling the neighbourhood "
               "alone with PyG's loader built per request; compare it with the `neighbour_lookup` share above."),
        ("code", "lat[lat['mode'] == 'lookup_only'][['model', 'p50_ms', 'p95_ms', 'p99_ms']].round(2)"),
        ("md", "Index size and the concurrency caveat: the metadata printed above lists the indexed transactions and the serialised "
               "index size. Scoring is serialised by one lock, so extra clients do not add throughput; the multi-client row shows "
               "the queueing cost."),
    ],
```

In `tests/test_notebooks.py::test_notebooks_execute_on_fixture_results`, before executing, also create the latency files:

```python
    from scripts.benchmark_latency import run_benchmark

    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", tmp_path / "f.pt")
    run_benchmark(prep, tmp_path / "f.pt", out, devices=["cpu"], batch_sizes=(1, 4), n_requests=40, include_http=False,
                  skip_parity=True, fixture=True, log=lambda *_: None)
```

- [ ] **Step 5: Add the container job to CI** (append to `.github/workflows/tests.yml`)

```yaml
  container:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
          cache: pip
          cache-dependency-path: requirements-ci.txt
      - name: Install CPU dependencies
        run: pip install -r requirements-ci.txt
      - name: Export fixture artifacts
        run: python -m scripts.export_artifacts --fixture --out artifacts
      - name: Build image
        run: docker build -t ure-scorer .
      - name: Run the container and smoke-test it
        run: |
          docker run -d --name scorer -p 8000:8000 ure-scorer
          python -m scripts.smoke_service --url http://localhost:8000 --samples artifacts/sample_requests.json --wait 90
      - name: Container logs on failure
        if: failure()
        run: docker logs scorer
```
Extend `tests/test_ci_config.py::test_workflow_runs_the_cpu_suite_without_secrets` (or add a sibling test):

```python
def test_workflow_builds_and_smoke_tests_the_container():
    wf = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    steps = " ".join(str(s.get("run", "")) for s in wf["jobs"]["container"]["steps"])
    assert "docker build" in steps and "scripts.smoke_service" in steps and "export_artifacts --fixture" in steps
    assert wf["jobs"]["container"]["needs"] == "test"
```

- [ ] **Step 6: Run to verify pass**

Run: `"$PY" -m pytest tests/test_benchmark.py tests/test_notebook_support.py tests/test_ci_config.py -q -m "not slow"` then `"$PY" -m pytest tests/test_benchmark.py tests/test_notebooks.py -q -m slow`
Expected: all pass. The benchmark tests assert structure, the percentile ordering that holds by construction, and parity; they never assert on a measured time.

- [ ] **Step 7: Fixture run and commit**

```bash
"$PY" -m scripts.run_all --fixture --seeds 0
"$PY" -m scripts.benchmark_latency --fixture --checkpoint outputs/fixture/models/multitask_seed0.pt --out outputs/fixture/plots --n-requests 400
"$PY" -m scripts.build_notebooks
git add scripts/benchmark_latency.py pipeline/notebook_support.py scripts/build_notebooks.py notebooks .github/workflows/tests.yml tests/test_benchmark.py tests/test_notebook_support.py tests/test_notebooks.py tests/test_ci_config.py
git commit -m "feat: add the latency benchmark with a parity gate, notebook 05 and the container CI job" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
Expected: the fixture benchmark prints a table; nothing from it is quoted as a result. The fixture checkpoint is a `multitask` one, which the benchmark serves through its fraud head.

---

### Task 22: Final docs, real-data rerun and branch review

**Files:**
- Create: `scripts/sampling_rule.py`, `tests/test_sampling_rule.py`
- Modify: `scripts/make_model_card.py` (serving section, sampling-rule table and README latency block), `tests/test_model_card.py`, `README.md`
- Generated and committed when real results exist: `docs/serving-spike.md`, `plots/latency.csv`, `plots/latency_meta.json`, `docs/MODEL_CARD.md`, README blocks, executed `notebooks/05_serving_latency.ipynb`

**Interfaces:**
- Produces in `scripts.make_model_card`: `latency_markdown(df, meta) -> str`; `build_model_card` adds a `## Serving and latency` section when `latency.csv` and `latency_meta.json` exist in `out_dir` (otherwise "The latency benchmark was not run."), and the latency metadata's `config_hash` joins the single-hash check; `splice_block(text, name, block)` (generalises `splice_readme`, which stays as the `RESULTS` wrapper); the CLI also fills `<!-- LATENCY:START -->` / `<!-- LATENCY:END -->` in the README when `--readme` is given and the markers exist.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_model_card.py`)

```python
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
```
Add one more test next to these: a latency file whose metadata says `"fixture": true` must make `build_model_card(out, reports, fixture=False)` raise `ValueError` matching "fixture".

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_model_card.py -q`
Expected: `ImportError` for `latency_markdown`.

- [ ] **Step 3: Implement**

In `scripts/make_model_card.py`:

```python
def splice_block(text: str, name: str, block: str) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    if start not in text or end not in text:
        raise ValueError(f"missing the {start} / {end} markers ({name} block)")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def splice_readme(readme: str, block: str) -> str:
    return splice_block(readme, "RESULTS", block)


def latency_markdown(df: pd.DataFrame, meta: dict) -> str:
    cols = ["model", "device", "mode", "batch_size", "concurrency", "p50_ms", "p95_ms", "p99_ms", "throughput_rps"]
    d = df[df["mode"].isin(["inprocess", "http"])][cols]
    p = meta["parity"]
    parity = (f"Online and offline scores agree (max absolute difference {p['max_abs_diff']:.1e} over {p['n_rows']} transactions)."
              if p["checked"] else f"Parity with the offline scores was not checked ({p['reason']}).")
    return (_table(d.assign(**{c: d[c].astype(float).round(2) for c in cols[5:]})) + f"\n\n{parity} "
            f"Latencies are per call, milliseconds, on {meta['hardware'].get('gpu') or 'a CPU-only machine'}; the {meta['budget_ms']:g} ms "
            "budget is illustrative, not a requirement. One machine replaying historical data is not a production claim.")
```
In `load_tables` add: `if (out_dir / "latency.csv").exists(): tables["latency"] = pd.read_csv(...)` and `tables["latency_meta"] = json.loads(...)`; in `check_single_config` the meta dict already works through the non-DataFrame branch (it has `config_hash`). In `build_model_card` add before `## Limitations`:

```python
    lat = tables.get("latency")
    parts.append("## Serving and latency\n\n" + (latency_markdown(lat, tables["latency_meta"]) if lat is not None
                                                 else "The latency benchmark was not run.") + "\n")
```
Fixture latency files carry `"fixture": true`; when the card is built with `fixture=False` and the latency meta says fixture, raise `ValueError("latency results are fixture results")`. Add that check and a test for it. In `main`, after the RESULTS splice:

```python
        if "latency" in tables and START.replace("RESULTS", "LATENCY") in path.read_text(encoding="utf-8"):
            path.write_text(splice_block(path.read_text(encoding="utf-8"), "LATENCY",
                                         latency_markdown(tables["latency"], tables["latency_meta"])), encoding="utf-8")
```

README: add a "Scoring service and latency" section (replace the Task 16 placeholder note) containing the two `LATENCY` marker lines with the sentence `No benchmark yet. Run python -m scripts.benchmark_latency.` between them, plus: how to export (`python -m scripts.export_artifacts`), run (`uvicorn serving.app:app`), the three curl examples (`/health`, `/score`, `/score_batch`), the 409 rule (a request older than the newest indexed transaction is rejected, or clamped in the benchmark), the single-lock caveat, the `"last"` versus `"uniform"` sampling note (the headline results use `"uniform"`, serving uses `"last"`, the model card reports both), the Docker commands, and the statement that CI builds and smoke-tests the image.

- [ ] **Step 4: Run to verify pass, then the full suites**

Run: `"$PY" -m pytest -m "not gpu and not slow" -q` then `"$PY" -m pytest -m slow -q`
Expected: all pass.

- [ ] **Step 5: Report the headline model under both sampling rules**

The spec requires the model card to show test PR-AUC under `"uniform"` (the training rule) and `"last"` (the serving rule). Write the failing test (`tests/test_sampling_rule.py`):

```python
import pandas as pd

from scripts.sampling_rule import sampling_rule_table
from scripts.train import default_spec, train_variant
from tests.serving_helpers import last_rule_prepared


def test_table_has_both_rules_with_finite_scores_and_leaves_the_config_alone(tmp_path):
    prep = last_rule_prepared()
    train_variant(prep, default_spec(prep, "fraud_only", 0), "cpu", tmp_path / "m.pt")
    before = prep.cfg["train"]["temporal_strategy"]
    table = sampling_rule_table(prep, tmp_path / "m.pt")
    assert list(table["rule"]) == ["uniform", "last"] and table["pr_auc"].notna().all()
    assert table["config_hash"].nunique() == 1 and prep.cfg["train"]["temporal_strategy"] == before
```
Run it (expect `ModuleNotFoundError: No module named 'scripts.sampling_rule'`), then implement `scripts/sampling_rule.py`:

```python
"""Test PR-AUC of one checkpoint under the training neighbour rule and the serving neighbour rule."""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from pipeline.config import config_hash, load_config, resolve_device, resolve_path
from pipeline.metrics import pr_auc
from scripts.train import load_checkpoint, predict_scores


def sampling_rule_table(prepared, checkpoint, device="cpu") -> pd.DataFrame:
    model, heads, meta = load_checkpoint(checkpoint, prepared, device)
    test = np.flatnonzero(prepared.graph["transaction"].test_mask.numpy())
    y = prepared.df["isFraud"].to_numpy()[test]
    rows = []
    for rule in ("uniform", "last"):
        variant_prep = copy.copy(prepared)
        variant_prep.cfg = copy.deepcopy(prepared.cfg)
        variant_prep.cfg["train"]["temporal_strategy"] = rule
        scores = predict_scores(variant_prep, model, heads, "fraud", test, device, variant=meta["variant"])
        rows.append({"rule": rule, "pr_auc": pr_auc(y, scores), "variant": meta["variant"], "seed": meta["seed"],
                     "temporal": bool(prepared.cfg["train"]["temporal"]), "config_hash": config_hash(prepared.cfg)})
    return pd.DataFrame(rows)


def main(argv=None):
    from pipeline.prepare import prepare

    p = argparse.ArgumentParser(description="Compare the uniform and last neighbour rules for one checkpoint.")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/multitask_seed0.pt")
    p.add_argument("--out", default="plots")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    table = sampling_rule_table(prepare(cfg), resolve_path(a.checkpoint), resolve_device(cfg["train"]["device"]))
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results_sampling_rule.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
```
Note that the `"uniform"` rule samples at random, so its PR-AUC varies slightly from run to run; the table is a single draw and the card says so. In `scripts/make_model_card.py`, load `results_sampling_rule.csv` when present (key `sampling_rule`, included in `check_single_config` through its `config_hash` column) and add a two-row table under `## Serving and latency` titled "Test PR-AUC of the served model under each neighbour rule". Add a card test that the table's two numbers appear in the card. Without the real data, run this step on the fixture (`python -m scripts.sampling_rule` needs real data, so call `sampling_rule_table` from the test only) and record that the real run is pending.

- [ ] **Step 6: Real-data benchmark and artifacts (only with the real data and checkpoint)**

```bash
"$PY" -m scripts.spike_serving
"$PY" -m scripts.benchmark_latency --checkpoint models/multitask_seed0.pt --out plots
"$PY" -m scripts.make_model_card --out plots --reports reports --card docs/MODEL_CARD.md --readme README.md
"$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=unified_risk_engine --ExecutePreprocessor.timeout=1800 notebooks/05_serving_latency.ipynb
```
If the benchmark raises `ParityError`, stop: that is train/serve skew. Fix the index or scorer (Task 19, Step 8) and rerun; never report latency from a scorer that fails parity. Read the generated README and card sections once and make sure each statement is generated or labelled as a hypothesis. Without the real data, skip this step and say so in the final message.

- [ ] **Step 7: Fresh-clone verification**

```bash
git clone . "$TMP/ure-verify"
cd "$TMP/ure-verify"
"$PY" -m pytest -m "not gpu and not slow" -q
"$PY" -m scripts.run_all --fixture --seeds 0
"$PY" -m scripts.export_artifacts --fixture --out artifacts
```
Expected: the tests pass and the fixture run completes, using the shared venv `PY` (the clone has no data and no models, which proves nothing depends on untracked files). Delete the temporary clone afterwards with the user's cleanup command if the safety check blocks removal, never by working around it.

- [ ] **Step 8: Whole-branch review, merge, and handoff**

Dispatch one fresh reviewer on the most capable model with the spec, this plan, and `git diff main...HEAD`. Ask it to check: (a) train/serve skew (anything the online path computes differently from `prepare`, the loader, or `history_features`), (b) leakage in the serving replay (the benchmark's index must contain only transactions before the test period plus the replayed ones), (c) API contract and error handling (422/409/503, thread safety), (d) CI files for secrets and for steps that cannot run on a clean runner, (e) claims in the README and model card that no table supports. Apply the fixes it confirms and commit them.

Then merge locally:

```bash
git checkout main
git merge --ff-only feature/risk-engine-core
git log --oneline | head -5
```
Do NOT push. Tell the user: what is done, what ran only on the fixture (and so is not a finding), which Kaggle files are still needed if the data gate stopped early, that the CI workflow has never run on GitHub and will do so on the first push, and ask whether to push (`GH_TOKEN="$(gh auth token --user jackalaka21)" git push` per the account rule). After an approved push, watch the first CI run with `GH_TOKEN="$(gh auth token --user jackalaka21)" gh run watch --repo jackalaka21/unified-risk-engine` and fix any failure it shows (a CPU-wheel or Python 3.14 issue is the most likely one) before calling CI verified.
