"""
Continue R-GCN training from checkpoint without re-downloading data.
Loads saved CSVs, rebuilds graph, loads best checkpoint, trains for N more epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

DATA_DIR  = Path("data")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM   = 64
EPOCHS    = 1000
BATCH     = 512
NEG_K     = 5
EVAL_EVERY = 50
GRAD_CLIP  = 1.0
LR        = 0.0003

print(f"Device: {DEVICE}")

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

# --- Rebuild HeteroData ---
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

treats_ei    = df_to_edge_index(df_drug_disease, "drug_id", "disease_id", drug_to_idx, disease_to_idx)
num_treats   = treats_ei.shape[1]
torch.manual_seed(42)
perm         = torch.randperm(num_treats)
n_train      = int(0.70 * num_treats)
n_val        = int(0.15 * num_treats)

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

# --- Model ---
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
        d = F.normalize(x_dict["drug"][drug_idx],    p=2, dim=-1)
        t = F.normalize(x_dict["disease"][disease_idx], p=2, dim=-1)
        return (d * t).sum(dim=-1)

    def forward(self, data, drug_idx, disease_idx):
        return self.decode(self.encode(data), drug_idx, disease_idx)


def corrupt_tails(tail_idx, num_tails):
    return torch.randint(0, num_tails, tail_idx.shape, device=tail_idx.device)


def evaluate(model, edge_index, num_tails, eval_batch=256):
    model.eval()
    all_dis = torch.arange(num_tails, device=DEVICE)
    ranks   = []
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


# --- Load checkpoint and train ---
rgcn = HeteroRGCN(len(disease_ids), len(gene_ids), len(drug_ids)).to(DEVICE)
ckpt = DATA_DIR / "rgcn_best.pt"
if ckpt.exists():
    state = torch.load(ckpt, map_location=DEVICE)
    # trim any embedding that grew by 1 due to a historical NaN in the gene set
    for key in ["disease_emb.weight", "gene_emb.weight", "drug_emb.weight"]:
        ckpt_size   = state[key].shape[0]
        model_size  = getattr(rgcn, key.split(".")[0]).weight.shape[0]
        if ckpt_size != model_size:
            print(f"  Trimming {key}: {ckpt_size} -> {model_size}")
            state[key] = state[key][:model_size]
    rgcn.load_state_dict(state)
    print(f"Loaded checkpoint from {ckpt}")

optimizer = torch.optim.Adam(rgcn.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

train_ei     = data["drug", "treats", "disease"].edge_index.to(DEVICE)
val_ei       = data["drug", "treats", "disease"].edge_index_val.to(DEVICE)
num_diseases = data["disease"].num_nodes

best_mrr, best_epoch = 0.0, 0

for epoch in range(1, EPOCHS + 1):
    rgcn.train()
    perm     = torch.randperm(train_ei.shape[1], device=DEVICE)
    shuffled = train_ei[:, perm]
    total_loss, n_batches = 0.0, 0

    for start in range(0, shuffled.shape[1], BATCH):
        end      = min(start + BATCH, shuffled.shape[1])
        head     = shuffled[0, start:end]
        tail     = shuffled[1, start:end]
        B        = len(head)
        head_rep = head.repeat(NEG_K)
        neg_tail = corrupt_tails(tail.repeat(NEG_K), num_diseases)

        x_dict     = rgcn.encode(data)
        pos_scores = rgcn.decode(x_dict, head, tail)
        neg_scores = rgcn.decode(x_dict, head_rep, neg_tail)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(B, device=DEVICE), torch.zeros(B * NEG_K, device=DEVICE)])
        loss   = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rgcn.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    avg_loss = total_loss / n_batches

    if epoch % EVAL_EVERY == 0:
        mrr, hits = evaluate(rgcn, val_ei, num_diseases)
        marker = " *" if mrr > best_mrr else ""
        if mrr > best_mrr:
            best_mrr, best_epoch = mrr, epoch
            torch.save(rgcn.state_dict(), DATA_DIR / "rgcn_best.pt")
        print(f"Epoch {epoch:4d} | loss: {avg_loss:.4f} | "
              f"MRR: {mrr:.4f} | H@1: {hits[1]:.4f} | H@3: {hits[3]:.4f} | H@10: {hits[10]:.4f}{marker}")
    else:
        print(f"Epoch {epoch:4d} | loss: {avg_loss:.4f}", flush=True)

print()
print(f"Best val MRR: {best_mrr:.4f} at epoch {best_epoch}")
