# Spike report: PyG capabilities the design depends on

Run on 2026-10-16 with `scripts/spike_pyg.py` in the project's own venv: Python 3.14.7, torch 2.14.1+cu130, torch-geometric 2.8.1, pyg-lib 0.9.0+pt214cu130, RTX 3070 Ti (8 GB). Synthetic hetero graph (transactions, users, email domains).

| # | Check | Result | Evidence | Decision |
|---|---|---|---|---|
| 0 | `pyg-lib` built for torch 2.14.0 under torch 2.14.1 | PASS | `torch 2.14.1+cu130, pyg_lib 0.9.0+pt214cu130, device cuda`; temporal sampling runs | Target stack stands; no fallback needed |
| 1 | Temporal hetero sampling, `time` on every node type, no neighbour later than its seed | PASS | `violations=0 batch_vector=True`; per-seed edges as designed (1 user per seed, about 10 recent transactions per entity) | **Option B stands: `train.temporal: true`** |
| 2 | Equal-timestamp rule | `earlier_or_equal` | a neighbour with the same timestamp as its seed is sampled (`[0, 1]`) | README states it; the online index uses `time <= t` |
| 3 | Entity nodes without `time` | sampling worked | no error | Design still gives every entity node a first-seen `time` |
| 4 | `"last"` strategy deterministic | PASS | identical sampled ids on two runs | Parity gate in phase 2 is feasible |
| 5 | Speed and memory (200k transactions, `[15, 10]`, GPU) | batch 256: 0.13 s/batch, 0.09 GiB, ~3.4 min/epoch; batch 1024: 0.10 s, 0.29 GiB, ~0.7 min; batch 2048: 0.19 s, 0.55 GiB, ~0.6 min | rule: largest batch with peak <= 5 GiB and epoch <= 5 min | `train.batch_fraud: 2048`, `train.batch_account: 1024`, `num_neighbors: [15, 10]` unchanged |
| 6 | Hetero `Explainer`/`GNNExplainer` | PASS | `HeteroExplanation`, transaction `node_mask (300, 8)`, one `edge_mask` per edge type | **`explain.mode: hetero`** |
| 6b | PyG `fidelity` on a hetero explanation | FAIL | `AttributeError: 'HeteroExplanation' has no attribute '_model_args'` | Task 14's `pyg_fidelity` returns `None` with a warning; the occlusion deletion fidelity is the reported check, and the report says PyG fidelity was unavailable |
| 7 | CPU-only Linux wheels for CI (Python 3.14) | PASS | `pyg_lib-0.9.0+pt214cpu-cp310-abi3-manylinux_2_28_x86_64.whl` and `torch-2.14.1+cpu-cp314-cp314-manylinux_2_28_x86_64.whl` both download | CI uses Python 3.14 with the CPU wheels; temporal tests run in CI |

Other facts recorded for later tasks: pandas is 3.0.6 (default string dtype), numpy 2.5.4, scikit-learn 1.9.1; importing torch under Python 3.14 prints a harmless `torch.jit.script` FutureWarning.

Limits: the speed numbers use a toy backbone and synthetic data; the real model and data are slower per batch. The epoch estimates are for planning only.
