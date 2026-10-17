# Screening of ideas to raise PR-AUC (validation only)

Script: `experiments/screen_improvements.py` (exploratory, not part of the tested pipeline). Seed 0 only. The test period was **not** scored.
The validation period is split in time into an earlier half A and a later half B. Anything chosen (tuned parameters, a blend weight, a feature
variant) is chosen on A and reported on B. Each idea is compared with `graph_labels_tabular` at default settings using a paired bootstrap on B.
The screen ran in two parts because the first run crashed in idea F (a bug in the script's identity code, since fixed); the baseline row
is identical in both parts. Ideas A, B, D and E come from part 1; F and C from part 2.

| idea | PR-AUC on A | PR-AUC on B | B minus baseline | paired 95% CI of that difference | note |
|---|---|---|---|---|---|
| baseline: graph_labels_tabular (defaults) | 0.573 | 0.480 | +0.000 | [+0.000, +0.000] |  |
| A. tuned trees (best of 20 random trials on A) | 0.595 | 0.502 | +0.022 | [+0.012, +0.035] | learning_rate=0.065, leaves=63, min_leaf=100, l2=10.0, max_features=1.0 |
| B. bag of 5 trees (max_features 0.7) | 0.593 | 0.507 | +0.027 | [+0.018, +0.037] |  |
| D. refit on train + validation A (up to B start minus embargo) | n/a | 0.518 | +0.038 | [+0.026, +0.051] | A is in the training data, so no A score |
| E1 + rates of card1, addr1, R_emaildomain | 0.570 | 0.476 | -0.003 | [-0.013, +0.008] |  |
| E2 + 30-day windowed rates | 0.582 | 0.507 | +0.028 | [+0.016, +0.039] |  |
| E3 + both | 0.582 | 0.503 | +0.023 | [+0.012, +0.034] | chosen on A |
| F. user identity from card1+addr1+D1 start day (replaces user) | 0.670 | 0.586 | +0.106 | [+0.079, +0.137] | 217850 users instead of 39974 |
| F+. same identity added as extra columns | 0.677 | 0.611 | +0.131 | [+0.103, +0.165] |  |
| C. rank blend, GNN weight 0.05 | 0.572 | 0.476 | -0.003 | [-0.010, +0.004] | chosen on A |
| C. rank blend, GNN weight 0.1 | 0.564 | 0.468 | -0.011 | [-0.021, -0.001] |  |
| C. rank blend, GNN weight 0.2 | 0.548 | 0.455 | -0.025 | [-0.038, -0.012] |  |
| C. rank blend, GNN weight 0.3 | 0.535 | 0.444 | -0.036 | [-0.051, -0.022] |  |
| C. rank blend, GNN weight 0.5 | 0.510 | 0.423 | -0.057 | [-0.074, -0.039] |  |

GNN alone: A 0.411, B 0.335 (early-stopped on A only, epoch 6)

## Reading it

- **Differences are from one seed and one half of the validation period**, so treat sizes as indicative. Bootstrap intervals only cover resampling of rows.
- **D1 identity (F, F+)** is by far the largest effect. It is a feature available at scoring time and nests inside the old user key, so users are still
  never split across train and validation. A gain this large still needs a leakage review before it is adopted; see the notes in the project README once adopted.
- **Refit on train + validation (D)**, **windowed rates (E2)**, **bagging (B)** and **tuning (A)** each give a smaller, clearly positive effect.
- **E3 was chosen on A over E2** only by a tie at three decimals; E2 alone is simpler and scored higher on B, which is a fact about B, not a basis for choosing.
- **Nothing in the GNN blend (C) helps:** every weight is at or below the baseline, and larger weights hurt.
- **Untested together:** these effects may overlap. The next step is to adopt the winners in the pipeline, with tests, and run the test period once.

<details><summary>Tuning trials (score on A)</summary>

- tuning trial 0: A=0.592 {'learning_rate': 0.06499076070944743, 'max_leaf_nodes': 127, 'min_samples_leaf': 200, 'l2_regularization': 0.0, 'max_features': 0.5, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 1: A=0.581 {'learning_rate': 0.17769673281094006, 'max_leaf_nodes': 15, 'min_samples_leaf': 50, 'l2_regularization': 10.0, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 2: A=0.561 {'learning_rate': 0.13450319151782653, 'max_leaf_nodes': 31, 'min_samples_leaf': 50, 'l2_regularization': 1.0, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 3: A=0.594 {'learning_rate': 0.02131027768758377, 'max_leaf_nodes': 127, 'min_samples_leaf': 200, 'l2_regularization': 10.0, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 4: A=0.593 {'learning_rate': 0.0427328356139911, 'max_leaf_nodes': 31, 'min_samples_leaf': 200, 'l2_regularization': 0.0, 'max_features': 0.5, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 5: A=0.584 {'learning_rate': 0.056823487556710434, 'max_leaf_nodes': 127, 'min_samples_leaf': 20, 'l2_regularization': 0.1, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 6: A=0.593 {'learning_rate': 0.031951057466740936, 'max_leaf_nodes': 63, 'min_samples_leaf': 50, 'l2_regularization': 0.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 7: A=0.580 {'learning_rate': 0.03814509963851027, 'max_leaf_nodes': 31, 'min_samples_leaf': 50, 'l2_regularization': 0.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 8: A=0.554 {'learning_rate': 0.18309951305979266, 'max_leaf_nodes': 15, 'min_samples_leaf': 100, 'l2_regularization': 0.1, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 9: A=0.590 {'learning_rate': 0.03783739246681771, 'max_leaf_nodes': 63, 'min_samples_leaf': 20, 'l2_regularization': 0.1, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 10: A=0.585 {'learning_rate': 0.06562942226909711, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'l2_regularization': 0.1, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 11: A=0.582 {'learning_rate': 0.1195950239185465, 'max_leaf_nodes': 31, 'min_samples_leaf': 100, 'l2_regularization': 10.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 12: A=0.591 {'learning_rate': 0.021909015990150455, 'max_leaf_nodes': 63, 'min_samples_leaf': 100, 'l2_regularization': 10.0, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 13: A=0.571 {'learning_rate': 0.023087641746916857, 'max_leaf_nodes': 31, 'min_samples_leaf': 100, 'l2_regularization': 10.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 14: A=0.578 {'learning_rate': 0.07833773558588009, 'max_leaf_nodes': 127, 'min_samples_leaf': 50, 'l2_regularization': 0.1, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 15: A=0.595 {'learning_rate': 0.06464365103975811, 'max_leaf_nodes': 63, 'min_samples_leaf': 100, 'l2_regularization': 10.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 16: A=0.591 {'learning_rate': 0.02811590268405999, 'max_leaf_nodes': 63, 'min_samples_leaf': 200, 'l2_regularization': 0.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 17: A=0.593 {'learning_rate': 0.12249741869917302, 'max_leaf_nodes': 127, 'min_samples_leaf': 20, 'l2_regularization': 1.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 18: A=0.571 {'learning_rate': 0.03107090633199901, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'l2_regularization': 0.0, 'max_features': 1.0, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}
- tuning trial 19: A=0.572 {'learning_rate': 0.14531601906275285, 'max_leaf_nodes': 127, 'min_samples_leaf': 200, 'l2_regularization': 0.1, 'max_features': 0.8, 'max_iter': 600, 'early_stopping': True, 'n_iter_no_change': 20, 'random_state': 0}

</details>
