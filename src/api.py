"""FastAPI service for drug-disease link prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import DEFAULT_DATA_DIR, GraphBundle, load_graph
from src.evaluate import predict_top_diseases
from src.model import HeteroRGCN, load_checkpoint

DEFAULT_TOP_K = 10
CHECKPOINT_NAME = "rgcn_best.pt"


class PredictRequest(BaseModel):
    drug_name: str = Field(..., examples=["ADALIMUMAB", "CHEMBL1201580"])


class DiseasePrediction(BaseModel):
    rank: int
    disease_id: str
    disease_name: str
    score: float


class PredictResponse(BaseModel):
    drug_id: str
    drug_name: str | None
    predictions: list[DiseasePrediction]


@dataclass
class AppState:
    bundle: GraphBundle
    model: HeteroRGCN
    device: torch.device


def create_app(data_dir: Path = DEFAULT_DATA_DIR) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = load_graph(data_dir)
        model = HeteroRGCN(
            len(bundle.disease_ids),
            len(bundle.gene_ids),
            len(bundle.drug_ids),
        ).to(device)
        checkpoint = data_dir / CHECKPOINT_NAME
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {checkpoint}")
        load_checkpoint(model, checkpoint, device)
        model.eval()
        app.state.runtime = AppState(bundle=bundle, model=model, device=device)
        yield

    app = FastAPI(
        title="BioKG Drug-Disease Predictor",
        description="Predict likely disease indications for a drug using an R-GCN knowledge graph.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "BioKG Drug-Disease Predictor",
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict with JSON body: {\"drug_name\": \"ADALIMUMAB\"}",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        runtime: AppState = app.state.runtime
        bundle = runtime.bundle

        try:
            drug_id = bundle.resolve_drug_id(request.drug_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        drug_idx = bundle.drug_to_idx[drug_id]
        ranked = predict_top_diseases(
            runtime.model,
            bundle.data,
            drug_idx,
            runtime.device,
            top_k=DEFAULT_TOP_K,
        )

        predictions = [
            DiseasePrediction(
                rank=rank,
                disease_id=bundle.idx_to_disease[disease_idx],
                disease_name=bundle.disease_names.get(
                    bundle.idx_to_disease[disease_idx],
                    bundle.idx_to_disease[disease_idx],
                ),
                score=round(score, 4),
            )
            for rank, (disease_idx, score) in enumerate(ranked, start=1)
        ]

        return PredictResponse(
            drug_id=drug_id,
            drug_name=bundle.drug_names.get(drug_id),
            predictions=predictions,
        )

    return app


app = create_app()
