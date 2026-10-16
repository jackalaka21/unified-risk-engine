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
