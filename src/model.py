"""R-GCN model for drug-disease link prediction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

MP_EDGE_TYPES = [
    ("gene", "associated_with", "disease"),
    ("disease", "rev_associated_with", "gene"),
    ("drug", "targets", "gene"),
    ("gene", "rev_targets", "drug"),
    ("gene", "interacts_with", "gene"),
]

DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 2


class HeteroRGCN(nn.Module):
    """Two-layer heterogeneous R-GCN with cosine-similarity decoder."""

    def __init__(
        self,
        num_diseases: int,
        num_genes: int,
        num_drugs: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ):
        super().__init__()
        self.disease_emb = nn.Embedding(num_diseases, hidden_dim)
        self.gene_emb = nn.Embedding(num_genes, hidden_dim)
        self.drug_emb = nn.Embedding(num_drugs, hidden_dim)
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {et: SAGEConv((-1, -1), hidden_dim) for et in MP_EDGE_TYPES},
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(0.3)

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        x_dict = {
            "disease": self.disease_emb(torch.arange(data["disease"].num_nodes, device=device)),
            "gene": self.gene_emb(torch.arange(data["gene"].num_nodes, device=device)),
            "drug": self.drug_emb(torch.arange(data["drug"].num_nodes, device=device)),
        }
        edge_index_dict = {et: data[et].edge_index.to(device) for et in MP_EDGE_TYPES}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}
        return x_dict

    def decode(
        self,
        x_dict: dict[str, torch.Tensor],
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        drugs = F.normalize(x_dict["drug"][drug_idx], p=2, dim=-1)
        diseases = F.normalize(x_dict["disease"][disease_idx], p=2, dim=-1)
        return (drugs * diseases).sum(dim=-1)

    def forward(
        self,
        data: HeteroData,
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(data), drug_idx, disease_idx)


def load_checkpoint(
    model: HeteroRGCN,
    checkpoint_path: Path,
    device: torch.device,
) -> HeteroRGCN:
    """Load weights, trimming embeddings if node counts drifted since training."""
    state = torch.load(checkpoint_path, map_location=device)
    for key in ("disease_emb.weight", "gene_emb.weight", "drug_emb.weight"):
        emb_name = key.split(".")[0]
        ckpt_size = state[key].shape[0]
        model_size = getattr(model, emb_name).weight.shape[0]
        if ckpt_size != model_size:
            state[key] = state[key][:model_size]
    model.load_state_dict(state)
    return model
