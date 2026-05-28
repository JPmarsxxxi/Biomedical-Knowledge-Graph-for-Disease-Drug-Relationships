"""
Final test-set evaluation. Loads best checkpoint, never touches train/val.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

DATA_DIR = Path("data")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM  = 64

# --- Load CSVs ---
df_disease_gene = pd.read_csv(DATA_DIR / "disease_gene.csv")
df_drug_gene    = pd.read_csv(DATA_DIR / "drug_gene.csv")
df_drug_disease = pd.read_csv(DATA_DIR / "drug_disease.csv")
df_gene_gene    = pd.read_csv(DATA_DIR / "gene_gene.csv")

# --- Rebuild node index maps ---
disease_ids = sorted(set(df_disease_gene["disease_id"].dropna()) | set(df_drug_disease["disease_id"].dropna()))
gene_ids    = sorted(set(df_disease_gene["gene_symbol"].dropna()) |
                     set(df_drug_gene["gene_symbol"].dropna())    |
                     set(df_gene_gene["gene_a"].dropna())         |
                     set(df_gene_gene["gene_b"].dropna()))
drug_ids    = sorted(set(df_drug_gene["drug_id"].dropna()) | set(df_drug_disease["drug_id"].dropna()))

disease_to_idx = {d: i for i, d in enumerate(disease_ids)}
gene_to_idx    = {g: i for i, g in enumerate(gene_ids)}
drug_to_idx    = {d: i for i, d in enumerate(drug_ids)}

print(f"Diseases: {len(disease_ids)}  Genes: {len(gene_ids)}  Drugs: {len(drug_ids)}")

def df_to_edge_index(df, src_col, dst_col, src_map, dst_map):
    src  = df[src_col].map(src_map)
    dst  = df[dst_col].map(dst_map)
    mask = src.notna() & dst.notna()
    return torch.stack([
        torch.tensor(src[mask].values.astype(int), dtype=torch.long),
        torch.tensor(dst[mask].values.astype(int), dtype=torch.long),
    ])

data = HeteroData()
data["disease"].num_nodes = len(disease_ids)
data["gene"].num_nodes    = len(gene_ids)
data["drug"].num_nodes    = len(drug_ids)

data["gene",    "associated_with", "disease"].edge_index = df_to_edge_index(df_disease_gene, "gene_symbol", "disease_id", gene_to_idx, disease_to_idx)
data["drug",    "targets",         "gene"].edge_index    = df_to_edge_index(df_drug_gene,    "drug_id",     "gene_symbol", drug_to_idx, gene_to_idx)
data["gene",    "interacts_with",  "gene"].edge_index    = df_to_edge_index(df_gene_gene,    "gene_a",      "gene_b",      gene_to_idx, gene_to_idx)

treats_ei  = df_to_edge_index(df_drug_disease, "drug_id", "disease_id", drug_to_idx, disease_to_idx)
num_treats = treats_ei.shape[1]
torch.manual_seed(42)
perm       = torch.randperm(num_treats)
n_train    = int(0.70 * num_treats)
n_val      = int(0.15 * num_treats)

data["drug", "treats", "disease"].edge_index      = treats_ei[:, perm[:n_train]]
data["drug", "treats", "disease"].edge_index_val  = treats_ei[:, perm[n_train:n_train + n_val]]
data["drug", "treats", "disease"].edge_index_test = treats_ei[:, perm[n_train + n_val:]]

data["disease", "rev_associated_with", "gene"].edge_index = data["gene", "associated_with", "disease"].edge_index.flip(0)
data["gene",    "rev_targets",         "drug"].edge_index = data["drug", "targets",         "gene"].edge_index.flip(0)

MP_EDGE_TYPES = [
    ("gene",    "associated_with",     "disease"),
    ("disease", "rev_associated_with", "gene"),
    ("drug",    "targets",             "gene"),
    ("gene",    "rev_targets",         "drug"),
    ("gene",    "interacts_with",      "gene"),
]

class HeteroRGCN(nn.Module):
    def __init__(self, num_diseases, num_genes, num_drugs, hidden_dim=64, num_layers=2):
        super().__init__()
        self.disease_emb = nn.Embedding(num_diseases, hidden_dim)
        self.gene_emb    = nn.Embedding(num_genes,    hidden_dim)
        self.drug_emb    = nn.Embedding(num_drugs,    hidden_dim)
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in MP_EDGE_TYPES}, aggr="sum")
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.3)

    def encode(self, data):
        device = next(self.parameters()).device
        x_dict = {
            "disease": self.disease_emb(torch.arange(data["disease"].num_nodes, device=device)),
            "gene":    self.gene_emb(   torch.arange(data["gene"].num_nodes,    device=device)),
            "drug":    self.drug_emb(   torch.arange(data["drug"].num_nodes,    device=device)),
        }
        eid = {et: data[et].edge_index.to(device) for et in MP_EDGE_TYPES}
        for conv in self.convs:
            x_dict = conv(x_dict, eid)
            x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}
        return x_dict

    def decode(self, x_dict, drug_idx, disease_idx):
        d = F.normalize(x_dict["drug"][drug_idx],       p=2, dim=-1)
        t = F.normalize(x_dict["disease"][disease_idx], p=2, dim=-1)
        return (d * t).sum(dim=-1)


def evaluate(model, edge_index, num_tails, eval_batch=256):
    model.eval()
    all_dis  = torch.arange(num_tails, device=DEVICE)
    ranks    = []
    with torch.no_grad():
        x_dict   = model.encode(data)
        drug_idx = edge_index[0]
        dis_idx  = edge_index[1]
        for start in range(0, len(drug_idx), eval_batch):
            end     = min(start + eval_batch, len(drug_idx))
            d_batch = drug_idx[start:end]
            t_batch = dis_idx[start:end]
            B       = len(d_batch)
            d_exp   = d_batch.unsqueeze(1).expand(B, num_tails).reshape(-1)
            a_exp   = all_dis.unsqueeze(0).expand(B, num_tails).reshape(-1)
            scores  = model.decode(x_dict, d_exp, a_exp).reshape(B, num_tails)
            true_s  = scores[torch.arange(B, device=DEVICE), t_batch]
            rank    = (scores > true_s.unsqueeze(1)).sum(dim=1) + 1
            ranks.append(rank.cpu())
    ranks = torch.cat(ranks).float()
    mrr   = (1.0 / ranks).mean().item()
    hits  = {k: (ranks <= k).float().mean().item() for k in [1, 3, 10]}
    return mrr, hits


rgcn = HeteroRGCN(len(disease_ids), len(gene_ids), len(drug_ids)).to(DEVICE)
ckpt = DATA_DIR / "rgcn_best.pt"
state = torch.load(ckpt, map_location=DEVICE)
for key in ["disease_emb.weight", "gene_emb.weight", "drug_emb.weight"]:
    ckpt_size  = state[key].shape[0]
    model_size = getattr(rgcn, key.split(".")[0]).weight.shape[0]
    if ckpt_size != model_size:
        state[key] = state[key][:model_size]
rgcn.load_state_dict(state)
print(f"Loaded {ckpt}\n")

test_ei      = data["drug", "treats", "disease"].edge_index_test.to(DEVICE)
num_diseases = data["disease"].num_nodes
print(f"Test edges: {test_ei.shape[1]}")

mrr, hits = evaluate(rgcn, test_ei, num_diseases)
print()
print("=== TEST SET RESULTS ===")
print(f"MRR:  {mrr:.4f}")
print(f"H@1:  {hits[1]:.4f}")
print(f"H@3:  {hits[3]:.4f}")
print(f"H@10: {hits[10]:.4f}")
