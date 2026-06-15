"""
Continue R-GCN training from checkpoint without re-downloading data.
Loads saved CSVs, rebuilds graph, loads best checkpoint, trains for N more epochs.
"""

import torch
import torch.nn as nn
from pathlib import Path

from src.data import DEFAULT_DATA_DIR, load_graph
from src.evaluate import corrupt_tails, evaluate
from src.model import HeteroRGCN, load_checkpoint

DATA_DIR = Path(DEFAULT_DATA_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH = 512
NEG_K = 5
EVAL_EVERY = 50
GRAD_CLIP = 1.0
LR = 0.0003

print(f"Device: {DEVICE}")

bundle = load_graph(DATA_DIR)
data = bundle.data

print(
    f"Diseases: {len(bundle.disease_ids)}  "
    f"Genes: {len(bundle.gene_ids)}  "
    f"Drugs: {len(bundle.drug_ids)}"
)

rgcn = HeteroRGCN(len(bundle.disease_ids), len(bundle.gene_ids), len(bundle.drug_ids)).to(DEVICE)
ckpt = DATA_DIR / "rgcn_best.pt"
if ckpt.exists():
    load_checkpoint(rgcn, ckpt, DEVICE)
    print(f"Loaded checkpoint from {ckpt}")

optimizer = torch.optim.Adam(rgcn.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

train_ei = data["drug", "treats", "disease"].edge_index.to(DEVICE)
val_ei = data["drug", "treats", "disease"].edge_index_val.to(DEVICE)
num_diseases = data["disease"].num_nodes

best_mrr, best_epoch = 0.0, 0

for epoch in range(1, EPOCHS + 1):
    rgcn.train()
    perm = torch.randperm(train_ei.shape[1], device=DEVICE)
    shuffled = train_ei[:, perm]
    total_loss, n_batches = 0.0, 0

    for start in range(0, shuffled.shape[1], BATCH):
        end = min(start + BATCH, shuffled.shape[1])
        head = shuffled[0, start:end]
        tail = shuffled[1, start:end]
        batch_size = len(head)
        head_rep = head.repeat(NEG_K)
        neg_tail = corrupt_tails(tail.repeat(NEG_K), num_diseases)

        x_dict = rgcn.encode(data)
        pos_scores = rgcn.decode(x_dict, head, tail)
        neg_scores = rgcn.decode(x_dict, head_rep, neg_tail)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat(
            [torch.ones(batch_size, device=DEVICE), torch.zeros(batch_size * NEG_K, device=DEVICE)]
        )
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rgcn.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches

    if epoch % EVAL_EVERY == 0:
        mrr, hits = evaluate(rgcn, data, val_ei, num_diseases, DEVICE)
        marker = " *" if mrr > best_mrr else ""
        if mrr > best_mrr:
            best_mrr, best_epoch = mrr, epoch
            torch.save(rgcn.state_dict(), DATA_DIR / "rgcn_best.pt")
        print(
            f"Epoch {epoch:4d} | loss: {avg_loss:.4f} | "
            f"MRR: {mrr:.4f} | H@1: {hits[1]:.4f} | H@3: {hits[3]:.4f} | H@10: {hits[10]:.4f}{marker}"
        )
    else:
        print(f"Epoch {epoch:4d} | loss: {avg_loss:.4f}", flush=True)

print()
print(f"Best val MRR: {best_mrr:.4f} at epoch {best_epoch}")
