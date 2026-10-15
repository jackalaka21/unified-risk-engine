# Unified Credit & Fraud Adaptive Risk Engine

**Status: design phase, no code yet.**

A heterogeneous graph neural network (Hetero-GAT) over the IEEE-CIS fraud data, with a shared backbone and two heads: short-term fraud per transaction and a long-term "default" proxy per user. It will use Focal Loss for the class imbalance and GNNExplainer for human-readable explanations, and be compared against a gradient-boosting baseline.

Honest scope note: IEEE-CIS contains no real credit-default labels, so "default" will be a proxy derived from fraud history. The project is joint risk modelling on a shared graph, not validated credit scoring.

Companion project: [deep-feedback-control](https://github.com/jackalaka21/deep-feedback-control) (separate repo).

## Data

The dataset is not included. Download `train_transaction.csv` (and optionally `train_identity.csv`) from the Kaggle competition `ieee-fraud-detection` (a Kaggle account and acceptance of the competition rules are required) and place the files in `data/`.
