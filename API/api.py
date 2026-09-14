"""API FastAPI pour les previsions Elastic Net sauvegardees.

Les artefacts pickle sont des fichiers locaux de confiance produits par
``run_etape2_part2_models.py``. L'API ne re-entraine pas le modele a chaque
appel : elle reconstruit les caracteristiques causales puis applique le
modele final de la cible et de l'horizon demandes.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.etape2 import features as feat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_etape2.yaml"
MODELS_DIR = PROJECT_ROOT / "outputs" / "etape2" / "models"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


CONFIG = _load_config()
TARGETS = CONFIG["targets"]
HORIZONS = CONFIG["temporal_protocol"]["horizons"]
MAX_LAGS = CONFIG["features"]["max_lags"]
MA_WINDOWS = CONFIG["features"]["moving_stat_windows"]


class Observation(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    values: dict[str, float | None] = Field(..., min_length=1)


class PredictionRequest(BaseModel):
    target: str
    horizon: int = Field(..., ge=1, le=5)
    history: list[Observation] = Field(..., min_length=6)


class PredictionResponse(BaseModel):
    target: str
    horizon: int
    origin_year: int
    target_year: int
    prediction: float
    model: str
    model_origin: int
    features_used: list[str]


@lru_cache(maxsize=None)
def _load_artifact(target: str, horizon: int) -> dict[str, Any]:
    artifact_path = MODELS_DIR / f"elasticnet_{target}_h{horizon}_finale.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    with artifact_path.open("rb") as artifact_file:
        return pickle.load(artifact_file)


def _feature_frame(request: PredictionRequest, target: str) -> pd.DataFrame:
    rows = []
    for observation in request.history:
        row = {"country_iso3": "MAR", "year": observation.year}
        row.update(observation.values)
        rows.append(row)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise HTTPException(status_code=422, detail="L'historique ne contient aucune observation.")
    if raw["year"].duplicated().any():
        raise HTTPException(status_code=422, detail="Les annees doivent etre uniques.")
    raw = raw.sort_values("year")
    columns = CONFIG["target_mechanisms"][target]["disponibles"]
    frame = feat.build_feature_frame(
        raw,
        columns,
        MAX_LAGS,
        MA_WINDOWS,
        int(raw["year"].min()),
        int(raw["year"].max()),
    )
    return feat.drop_all_nan_columns(frame)


app = FastAPI(
    title="MEF Maroc UC-S1 Forecast API",
    version="1.0.0",
    description="Previsions Elastic Net fondees sur les artefacts de l'etape 2.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mef-maroc-ucs1"}


@app.get("/models")
def models() -> dict[str, Any]:
    available = []
    for target in TARGETS:
        for horizon in HORIZONS:
            path = MODELS_DIR / f"elasticnet_{target}_h{horizon}_finale.pkl"
            if path.exists():
                available.append({"target": target, "horizon": horizon})
    return {"count": len(available), "models": available}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if request.target not in TARGETS:
        raise HTTPException(status_code=422, detail=f"Cible inconnue: {request.target}")
    if request.horizon not in HORIZONS:
        raise HTTPException(status_code=422, detail=f"Horizon disponible: {HORIZONS}")

    try:
        artifact = _load_artifact(request.target, request.horizon)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Artefact de modele indisponible.") from error

    frame = _feature_frame(request, request.target)
    origin_year = int(frame.index.max())
    required_features = list(artifact["features"])
    missing_features = [column for column in required_features if column not in frame.columns]
    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={"message": "Variables ou historique insuffisant.", "missing_features": missing_features},
        )
    row = frame.loc[[origin_year], required_features]
    if row.isna().any(axis=None):
        missing_values = row.columns[row.isna().iloc[0]].tolist()
        raise HTTPException(
            status_code=422,
            detail={"message": "Valeurs manquantes pour l'origine de prevision.", "missing_features": missing_values},
        )

    prediction = float(artifact["model"].predict(artifact["scaler"].transform(row))[0])
    return PredictionResponse(
        target=request.target,
        horizon=request.horizon,
        origin_year=origin_year,
        target_year=origin_year + request.horizon,
        prediction=prediction,
        model="elasticnet_finale",
        model_origin=int(artifact["origin"]),
        features_used=required_features,
    )