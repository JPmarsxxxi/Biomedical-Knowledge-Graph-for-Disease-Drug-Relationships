# Biomedical Knowledge Graph for Disease-Drug Relationship Prediction

A heterogeneous knowledge graph built from public biomedical databases, with a relational GCN trained to predict missing drug-disease links via link prediction.

## Overview

Drugs, diseases, and genes are represented as nodes in a multi-relational graph. A two-layer R-GCN learns embeddings by passing messages along biological pathways (gene-disease associations, drug-target interactions, protein-protein interactions), then scores candidate drug-disease pairs via cosine similarity. The task is to recover held-out drug-disease edges from the ChEMBL indication database.

## Data Sources

| Source | Relation | Edges |
|--------|----------|-------|
| [Open Targets](https://platform.opentargets.org/) | gene → associated_with → disease | 1,000 |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | drug → targets → gene | 2,164 |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | drug → treats → disease | 23,103 |
| [STRING](https://string-db.org/) | gene ↔ interacts_with ↔ gene | 7,401 |

**Graph:** 8,115 nodes (1,984 diseases, 1,094 genes, 5,036 drugs) — 33,668 edges total

## Models

### TransE (baseline)
Translation-based embedding. Learns `h + r ≈ t` for each triple. Scored with L1 distance, trained with margin ranking loss.

### R-GCN (main model)
Two-layer heterogeneous graph convolutional network using SAGEConv per relation type. Embeddings are aggregated across all non-target edge types (gene-disease, drug-gene, PPI) — the `treats` edges are deliberately excluded from message passing to prevent degree bias. Drug-disease pairs are scored with cosine similarity.

## Results

Evaluated on 3,466 held-out drug-disease pairs (15% test split). For each pair, the correct disease is ranked against all 1,984 candidates.

| Model | MRR | H@1 | H@3 | H@10 |
|-------|-----|-----|-----|------|
| TransE (50 epochs) | 0.0373 | 0.0099 | 0.0248 | 0.0883 |
| **R-GCN (~3,850 epochs)** | **0.3292** | **0.2519** | **0.3070** | **0.4209** |

R-GCN achieves **8.8× higher MRR** than TransE. The correct disease ranks in the top 10 for 42% of test pairs out of 1,984 candidates. Val and test MRR are within 0.003 of each other, indicating no overfitting.

## Subgraph Visualisation

`data/biokg_subgraph.html` — interactive Pyvis graph showing the top 15 R-GCN predictions for Rheumatoid Arthritis. Nodes are colour-coded: red = disease, green = known training drug, blue = held-out test drug, orange = novel hypothesis, gold = shared gene target.

Open the file in any browser to explore.

## Reproducing

```bash
pip install torch torch_geometric networkx pandas numpy requests tqdm pyvis

# Run the notebook top-to-bottom (re-fetches data from APIs)
jupyter notebook biokg.ipynb

# Or skip the API calls — CSVs are included, train from scratch:
python continue_training.py

# Evaluate the saved checkpoint on the test set:
python eval_test.py
```

**Note:** `data/rgcn_best.pt` is the trained checkpoint (~3,850 epochs). Delete it before running `continue_training.py` if you want to train from scratch.

## Project Structure

```
biokg.ipynb              — end-to-end notebook (data fetch → model → results → viz)
continue_training.py     — standalone training script (loads CSVs, continues from checkpoint)
eval_test.py             — test set evaluation
data/
  disease_gene.csv       — Open Targets gene-disease associations
  drug_gene.csv          — ChEMBL drug-target mechanisms
  drug_disease.csv       — ChEMBL drug indications (phase ≥ 3)
  gene_gene.csv          — STRING protein-protein interactions
  rgcn_best.pt           — best R-GCN checkpoint (val MRR 0.3317)
  biokg_subgraph.html    — interactive subgraph visualisation
```
