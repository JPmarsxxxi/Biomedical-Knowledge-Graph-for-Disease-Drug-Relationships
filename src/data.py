"""Load biomedical edge lists and build a PyG heterogeneous graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

DEFAULT_DATA_DIR = Path("data")
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SPLIT_SEED = 42


@dataclass
class GraphBundle:
    """Everything needed to train or run inference on the knowledge graph."""

    data: HeteroData
    disease_ids: list[str]
    gene_ids: list[str]
    drug_ids: list[str]
    disease_to_idx: dict[str, int]
    gene_to_idx: dict[str, int]
    drug_to_idx: dict[str, int]
    drug_names: dict[str, str]
    disease_names: dict[str, str]
    drug_name_to_id: dict[str, str]

    @property
    def idx_to_disease(self) -> dict[int, str]:
        return {i: d for d, i in self.disease_to_idx.items()}

    @property
    def idx_to_drug(self) -> dict[int, str]:
        return {i: d for d, i in self.drug_to_idx.items()}

    def resolve_drug_id(self, query: str) -> str:
        """Resolve a ChEMBL ID or drug name to a graph drug_id."""
        token = query.strip()
        if not token:
            raise ValueError("Drug name cannot be empty.")

        chembl_id = token.upper()
        if chembl_id in self.drug_to_idx:
            return chembl_id

        drug_id = self.drug_name_to_id.get(token.lower())
        if drug_id is not None:
            return drug_id

        raise ValueError(f"Unknown drug: {query!r}")

def load_tables(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read the four edge-list CSVs from disk."""
    return {
        "disease_gene": pd.read_csv(data_dir / "disease_gene.csv"),
        "drug_gene": pd.read_csv(data_dir / "drug_gene.csv"),
        "drug_disease": pd.read_csv(data_dir / "drug_disease.csv"),
        "gene_gene": pd.read_csv(data_dir / "gene_gene.csv"),
    }


def build_node_maps(tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str], list[str]]:
    """Derive sorted node ID lists for each entity type."""
    df_disease_gene = tables["disease_gene"]
    df_drug_gene = tables["drug_gene"]
    df_drug_disease = tables["drug_disease"]
    df_gene_gene = tables["gene_gene"]

    disease_ids = sorted(
        set(df_disease_gene["disease_id"].dropna()) | set(df_drug_disease["disease_id"].dropna())
    )
    gene_ids = sorted(
        set(df_disease_gene["gene_symbol"].dropna())
        | set(df_drug_gene["gene_symbol"].dropna())
        | set(df_gene_gene["gene_a"].dropna())
        | set(df_gene_gene["gene_b"].dropna())
    )
    drug_ids = sorted(
        set(df_drug_gene["drug_id"].dropna()) | set(df_drug_disease["drug_id"].dropna())
    )
    return disease_ids, gene_ids, drug_ids


def build_entity_names(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build drug/disease display names and a case-insensitive drug name index."""
    drug_names: dict[str, str] = {}
    for df in (tables["drug_gene"], tables["drug_disease"]):
        if "drug_name" not in df.columns:
            continue
        for drug_id, drug_name in zip(df["drug_id"], df["drug_name"]):
            if pd.notna(drug_id) and pd.notna(drug_name) and drug_id not in drug_names:
                drug_names[str(drug_id)] = str(drug_name)

    disease_names: dict[str, str] = {}
    for df in (tables["disease_gene"], tables["drug_disease"]):
        if "disease_name" not in df.columns:
            continue
        for disease_id, disease_name in zip(df["disease_id"], df["disease_name"]):
            if pd.notna(disease_id) and pd.notna(disease_name) and disease_id not in disease_names:
                disease_names[str(disease_id)] = str(disease_name)

    drug_name_to_id: dict[str, str] = {}
    for drug_id, drug_name in drug_names.items():
        key = drug_name.lower()
        if key not in drug_name_to_id:
            drug_name_to_id[key] = drug_id

    return drug_names, disease_names, drug_name_to_id


def df_to_edge_index(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    src_map: dict[str, int],
    dst_map: dict[str, int],
) -> torch.Tensor:
    """Map string node IDs to integers and return a [2, E] edge tensor."""
    src = df[src_col].map(src_map)
    dst = df[dst_col].map(dst_map)
    mask = src.notna() & dst.notna()
    return torch.stack(
        [
            torch.tensor(src[mask].values.astype(int), dtype=torch.long),
            torch.tensor(dst[mask].values.astype(int), dtype=torch.long),
        ]
    )


def build_graph(
    tables: dict[str, pd.DataFrame],
    *,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SPLIT_SEED,
) -> GraphBundle:
    """Build HeteroData with train/val/test splits on drug-disease edges."""
    disease_ids, gene_ids, drug_ids = build_node_maps(tables)
    disease_to_idx = {d: i for i, d in enumerate(disease_ids)}
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}
    drug_names, disease_names, drug_name_to_id = build_entity_names(tables)

    data = HeteroData()
    data["disease"].num_nodes = len(disease_ids)
    data["gene"].num_nodes = len(gene_ids)
    data["drug"].num_nodes = len(drug_ids)

    data["gene", "associated_with", "disease"].edge_index = df_to_edge_index(
        tables["disease_gene"], "gene_symbol", "disease_id", gene_to_idx, disease_to_idx
    )
    data["drug", "targets", "gene"].edge_index = df_to_edge_index(
        tables["drug_gene"], "drug_id", "gene_symbol", drug_to_idx, gene_to_idx
    )
    data["gene", "interacts_with", "gene"].edge_index = df_to_edge_index(
        tables["gene_gene"], "gene_a", "gene_b", gene_to_idx, gene_to_idx
    )

    treats_ei = df_to_edge_index(
        tables["drug_disease"], "drug_id", "disease_id", drug_to_idx, disease_to_idx
    )
    num_treats = treats_ei.shape[1]
    torch.manual_seed(seed)
    perm = torch.randperm(num_treats)
    n_train = int(train_frac * num_treats)
    n_val = int(val_frac * num_treats)

    data["drug", "treats", "disease"].edge_index = treats_ei[:, perm[:n_train]]
    data["drug", "treats", "disease"].edge_index_val = treats_ei[:, perm[n_train : n_train + n_val]]
    data["drug", "treats", "disease"].edge_index_test = treats_ei[:, perm[n_train + n_val :]]

    data["disease", "rev_associated_with", "gene"].edge_index = data[
        "gene", "associated_with", "disease"
    ].edge_index.flip(0)
    data["gene", "rev_targets", "drug"].edge_index = data["drug", "targets", "gene"].edge_index.flip(
        0
    )

    return GraphBundle(
        data=data,
        disease_ids=disease_ids,
        gene_ids=gene_ids,
        drug_ids=drug_ids,
        disease_to_idx=disease_to_idx,
        gene_to_idx=gene_to_idx,
        drug_to_idx=drug_to_idx,
        drug_names=drug_names,
        disease_names=disease_names,
        drug_name_to_id=drug_name_to_id,
    )


def load_graph(data_dir: Path = DEFAULT_DATA_DIR) -> GraphBundle:
    """Convenience: load CSVs and build the graph in one call."""
    return build_graph(load_tables(data_dir))
