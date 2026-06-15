"""Final test-set evaluation. Loads best checkpoint, never touches train/val."""

import torch
from pathlib import Path

from src.data import DEFAULT_DATA_DIR, load_graph
from src.evaluate import evaluate
from src.model import HeteroRGCN, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(DEFAULT_DATA_DIR)

bundle = load_graph(DATA_DIR)
data = bundle.data

print(
    f"Diseases: {len(bundle.disease_ids)}  "
    f"Genes: {len(bundle.gene_ids)}  "
    f"Drugs: {len(bundle.drug_ids)}"
)

rgcn = HeteroRGCN(len(bundle.disease_ids), len(bundle.gene_ids), len(bundle.drug_ids)).to(DEVICE)
ckpt = DATA_DIR / "rgcn_best.pt"
load_checkpoint(rgcn, ckpt, DEVICE)
print(f"Loaded {ckpt}\n")

test_ei = data["drug", "treats", "disease"].edge_index_test.to(DEVICE)
num_diseases = data["disease"].num_nodes
print(f"Test edges: {test_ei.shape[1]}")

mrr, hits = evaluate(rgcn, data, test_ei, num_diseases, DEVICE)
print()
print("=== TEST SET RESULTS ===")
print(f"MRR:  {mrr:.4f}")
print(f"H@1:  {hits[1]:.4f}")
print(f"H@3:  {hits[3]:.4f}")
print(f"H@10: {hits[10]:.4f}")
