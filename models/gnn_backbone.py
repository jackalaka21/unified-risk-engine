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
