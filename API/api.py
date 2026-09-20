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
DATA_DIR = PROJECT_ROOT / "data"

DATASETS = {
    "wide_gold_full": {
        "file": "base_B_revised_wide.csv",
        "label": "Base finale nettoyée",
        "description": "Base large finale nettoyée et harmonisée, utilisée pour les analyses et les comparaisons.",
    },
    "wide_gold_train_only": {
        "file": "donnees_wide_gold_train_only_20260906T112701Z.csv",
        "label": "Base large entraînement",
        "description": "Version réservée à l'entraînement, hors années de holdout.",
    },
    "wide_standardisees": {
        "file": "donnees_wide_standardisees_20260906T112701Z.csv",
        "label": "Base large standardisée",
        "description": "Indicateurs transformés pour les analyses comparatives.",
    },
    "long_full": {
        "file": "donnees_long_full_20260906T112701Z.csv",
        "label": "Base longue complète",
        "description": "Une ligne par pays, année et indicateur, avec les statuts de qualité.",
    },
    "imputations": {
        "file": "journal_imputations_20260906T112701Z.csv",
        "label": "Journal des imputations",
        "description": "Traçabilité des valeurs imputées et des méthodes appliquées.",
    },
    "api_calls": {
        "file": "journal_appels_api_20260906T112701Z.csv",
        "label": "Journal des appels API",
        "description": "Historique des appels aux sources de données externes.",
    },
    "traceability": {
        "file": "derived_traceability_20260906T112701Z.csv",
        "label": "Traçabilité dérivée",
        "description": "Statuts et provenance des variables dérivées.",
    },
}


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


@lru_cache(maxsize=None)
def _load_dataset(dataset_id: str) -> pd.DataFrame:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise KeyError(dataset_id)
    path = DATA_DIR / dataset["file"]
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    return frame


ANALYSIS_INDICATORS = [
    "CROISSANCE_PIB",
    "INFLATION_CPI",
    "DETTE_PUBLIQUE",
    "SOBG",
    "BALANCE_COURANTE",
    "TACH",
    "FBCF",
]


def _filtered_dataset(
    dataset_id: str,
    country: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise _dataset_error(dataset_id)
    try:
        frame = _load_dataset(dataset_id).copy()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Fichier de données indisponible.") from error
    if country and "country_iso3" in frame.columns:
        frame = frame[frame["country_iso3"].astype(str).str.upper() == country.upper()]
    if year_start is not None and "year" in frame.columns:
        frame = frame[pd.to_numeric(frame["year"], errors="coerce") >= year_start]
    if year_end is not None and "year" in frame.columns:
        frame = frame[pd.to_numeric(frame["year"], errors="coerce") <= year_end]
    return frame


def _dataset_error(dataset_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Jeu de données inconnu: {dataset_id}")


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


@app.get("/data/datasets")
def data_catalog() -> dict[str, Any]:
    datasets = []
    for dataset_id, metadata in DATASETS.items():
        try:
            frame = _load_dataset(dataset_id)
        except FileNotFoundError:
            continue
        datasets.append({
            "id": dataset_id,
            "label": metadata["label"],
            "description": metadata["description"],
            "rows": len(frame),
            "columns": len(frame.columns),
            "column_names": list(frame.columns),
            "countries": sorted(frame["country_iso3"].dropna().astype(str).unique().tolist())
            if "country_iso3" in frame.columns else [],
        })
    return {"count": len(datasets), "datasets": datasets}


@app.get("/data/analysis")
def data_analysis(
    dataset_id: str = "wide_gold_full",
    country: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> dict[str, Any]:
    """Return descriptive statistics and chart-ready aggregates for the final base."""
    frame = _filtered_dataset(dataset_id, country, year_start, year_end)
    numeric_columns = [column for column in frame.columns if column not in {"country_iso3", "year"}]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    indicators = [column for column in ANALYSIS_INDICATORS if column in numeric.columns]

    summary = []
    for column in indicators:
        values = numeric[column].dropna()
        summary.append({
            "indicator": column,
            "observations": int(values.size),
            "missing": int(numeric[column].isna().sum()),
            "mean": float(values.mean()) if not values.empty else None,
            "median": float(values.median()) if not values.empty else None,
            "std": float(values.std()) if len(values) > 1 else None,
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
        })

    missingness = [
        {"indicator": column, "missing": int(numeric[column].isna().sum()), "completeness": round(float(numeric[column].notna().mean() * 100), 2)}
        for column in numeric_columns
    ]
    trend_frame = frame.copy()
    trend_frame["year"] = pd.to_numeric(trend_frame["year"], errors="coerce")
    trends = []
    for year, group in trend_frame.groupby("year", dropna=True):
        point = {"year": int(year)}
        for column in indicators:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            point[column] = float(values.mean()) if not values.empty else None
        trends.append(point)

    latest_year = int(trend_frame["year"].max()) if not trend_frame.empty else None
    latest = []
    if latest_year is not None:
        latest_frame = trend_frame[trend_frame["year"] == latest_year]
        for country_code, group in latest_frame.groupby("country_iso3", dropna=True):
            values = {column: (float(pd.to_numeric(group[column], errors="coerce").iloc[0]) if pd.notna(group[column].iloc[0]) else None) for column in indicators}
            latest.append({"country": str(country_code), "values": values})

    return {
        "dataset": dataset_id,
        "filters": {"country": country, "year_start": year_start, "year_end": year_end},
        "scope": {"rows": int(len(frame)), "countries": int(frame["country_iso3"].nunique()) if "country_iso3" in frame else 0, "year_min": int(trend_frame["year"].min()) if not trend_frame.empty else None, "year_max": latest_year, "columns": len(frame.columns)},
        "indicators": indicators,
        "summary": summary,
        "missingness": missingness,
        "trends": trends,
        "latest": latest,
    }


@app.get("/data/{dataset_id}")
def data_rows(
    dataset_id: str,
    country: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    search: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    frame = _filtered_dataset(dataset_id, country, year_start, year_end)
    if search:
        searchable = frame.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False))
        frame = frame[searchable.any(axis=1)]

    total = len(frame)
    offset = max(offset, 0)
    limit = min(max(limit, 1), 200)
    page = frame.iloc[offset:offset + limit]
    rows = __import__("json").loads(page.to_json(orient="records"))
    return {
        "dataset": dataset_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "columns": list(frame.columns),
        "rows": rows,
    }


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