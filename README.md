# BioKG — Drug-Disease Link Prediction

## The problem

Given a drug, which diseases might it treat?

Drug repurposing usually means manually searching literature and databases. This project automates that: it learns from a biomedical knowledge graph (genes, drugs, diseases, and how they connect) and ranks likely drug–disease pairs.

## The result

A trained R-GCN scores every disease for a given drug. On held-out test data, the correct disease lands in the **top 10 predictions 42% of the time** (out of 1,984 candidates). MRR: **0.329**.

Live demo — send a drug name, get back ranked diseases:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"drug_name": "ADALIMUMAB"}'
```

Example response:

```json
{
  "drug_id": "CHEMBL1201580",
  "drug_name": "ADALIMUMAB",
  "predictions": [
    {"rank": 1, "disease_id": "EFO_0000685", "disease_name": "rheumatoid arthritis", "score": 0.9922},
    {"rank": 2, "disease_id": "EFO_0000305", "disease_name": "breast carcinoma", "score": 0.9084}
  ]
}
```

Interactive docs: http://localhost:8000/docs

---

## Quick start

### Docker (recommended)

```bash
docker build -t biokg .
docker run -p 8000:8000 biokg
```

### Local

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Accepts drug names (`ADALIMUMAB`) or ChEMBL IDs (`CHEMBL1201580`).

---

## How it works

Drugs, diseases, and genes are nodes in a heterogeneous graph. A two-layer R-GCN learns embeddings by passing messages along biological pathways (gene–disease associations, drug–target interactions, protein–protein interactions), then scores drug–disease pairs with cosine similarity. The `treats` edges are held out during message passing to avoid degree bias.

### Data sources

| Source | Relation | Edges |
|--------|----------|-------|
| [Open Targets](https://platform.opentargets.org/) | gene → associated_with → disease | 1,000 |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | drug → targets → gene | 2,164 |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | drug → treats → disease | 23,103 |
| [STRING](https://string-db.org/) | gene ↔ interacts_with ↔ gene | 7,401 |

**Graph:** 8,115 nodes (1,984 diseases, 1,094 genes, 5,036 drugs)

### Model performance

Evaluated on 3,466 held-out drug–disease pairs (15% test split):

| Model | MRR | H@1 | H@3 | H@10 |
|-------|-----|-----|-----|------|
| TransE (baseline) | 0.037 | 0.010 | 0.025 | 0.088 |
| **R-GCN** | **0.329** | **0.252** | **0.307** | **0.421** |

---

## Project structure

```
src/
  data.py       — load CSVs, build graph, resolve drug names
  model.py      — R-GCN architecture and checkpoint loading
  evaluate.py   — metrics and inference helpers
  api.py        — FastAPI service (POST /predict)
data/           — edge lists + trained checkpoint (rgcn_best.pt)
biokg.ipynb     — original end-to-end notebook
eval_test.py    — benchmark on held-out test set
continue_training.py — resume training from checkpoint
```

---

## Development

```bash
# Evaluate the saved checkpoint on the test set
python eval_test.py

# Continue training from checkpoint
python continue_training.py

# Explore the original notebook (re-fetches data from APIs)
jupyter notebook biokg.ipynb
```

Subgraph visualisation: open `data/biokg_subgraph.html` in a browser.
