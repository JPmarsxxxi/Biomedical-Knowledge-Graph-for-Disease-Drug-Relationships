"""Link-prediction evaluation metrics and inference helpers."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.model import HeteroRGCN

HIT_KS = (1, 3, 10)


def corrupt_tails(tail_idx: torch.Tensor, num_tails: int) -> torch.Tensor:
    return torch.randint(0, num_tails, tail_idx.shape, device=tail_idx.device)


def evaluate(
    model: HeteroRGCN,
    data: HeteroData,
    edge_index: torch.Tensor,
    num_tails: int,
    device: torch.device,
    eval_batch: int = 256,
) -> tuple[float, dict[int, float]]:
    """
    Rank the true disease for each drug against all candidates.

    Returns mean reciprocal rank (MRR) and Hits@k for k in {1, 3, 10}.
    """
    model.eval()
    all_diseases = torch.arange(num_tails, device=device)
    ranks: list[torch.Tensor] = []

    with torch.no_grad():
        x_dict = model.encode(data)
        drug_idx = edge_index[0]
        disease_idx = edge_index[1]

        for start in range(0, len(drug_idx), eval_batch):
            end = min(start + eval_batch, len(drug_idx))
            drugs = drug_idx[start:end]
            true_diseases = disease_idx[start:end]
            batch_size = len(drugs)

            drugs_expanded = drugs.unsqueeze(1).expand(batch_size, num_tails).reshape(-1)
            all_expanded = all_diseases.unsqueeze(0).expand(batch_size, num_tails).reshape(-1)
            scores = model.decode(x_dict, drugs_expanded, all_expanded).reshape(batch_size, num_tails)

            true_scores = scores[torch.arange(batch_size, device=device), true_diseases]
            rank = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.append(rank.cpu())

    ranks_tensor = torch.cat(ranks).float()
    mrr = (1.0 / ranks_tensor).mean().item()
    hits = {k: (ranks_tensor <= k).float().mean().item() for k in HIT_KS}
    return mrr, hits


def predict_top_diseases(
    model: HeteroRGCN,
    data: HeteroData,
    drug_idx: int,
    device: torch.device,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Score every disease for one drug; return top-k (disease_idx, score) pairs."""
    model.eval()
    num_diseases = data["disease"].num_nodes

    with torch.no_grad():
        x_dict = model.encode(data)
        drug_tensor = torch.full((num_diseases,), drug_idx, device=device, dtype=torch.long)
        disease_tensor = torch.arange(num_diseases, device=device)
        scores = model.decode(x_dict, drug_tensor, disease_tensor)

    top_scores, top_indices = torch.topk(scores, k=min(top_k, num_diseases))
    return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]
