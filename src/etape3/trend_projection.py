"""Projection des trajectoires tendancielles (Maroc et donneurs), pour la
construction du benchmark combine (Eq. 3.14) et de la trajectoire
scenarisee (Eq. 3.15).

Reutilise integralement l'architecture de l'etape 2 (prevision directe par
horizon, memes plafonds de retards/caracteristiques, meme selection a 3
niveaux) -- ne remplace pas la dynamique validee, l'etend seulement :
- Maroc, horizons 1-5 : reutilise TEL QUEL le modele fige et valide en
  etape 2 (caracteristiques + hyperparametres geles, rechargeables depuis
  outputs/etape2/models/e2_model_registry.json).
- Maroc, horizons 6-10 : meme architecture, nouveau reglage (jamais evalue
  en biais/RMSE faute de realisations) -- extrapolation signalee comme
  telle partout en aval.
- Donneurs, horizons 1-10 : meme architecture, nouveau reglage pour chaque
  pays (l'etape 2 ne portait que sur le Maroc).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.etape2 import selection as sel
from src.etape2 import temporal as tmp


def load_frozen_mar_spec(models_dir: Path, target: str, horizon: int) -> dict | None:
    registry_path = models_dir.parent.parent / "etape2" / "models" / "e2_model_registry.json"
    if not registry_path.exists():
        return None
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    key = f"{target}_h{horizon}"
    if key not in registry:
        return None
    entry = registry[key]
    return {"features": entry["frozen_features"], "alpha": entry["alpha"], "l1_ratio": entry["l1_ratio"]}


def fit_and_project(
    feature_frame: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
    candidate_columns: list[str],
    origin_year: int,
    min_origin: int,
    redundancy_threshold: float,
    alphas: list[float],
    l1_ratios: list[float],
    frozen_spec: dict | None = None,
    random_state: int = 42,
) -> dict | None:
    """Ajuste (avec specification gelee si fournie, sinon reglage complet
    sur l'historique accessible) et projette y_{origin_year+horizon}."""
    ds = tmp.build_direct_dataset(feature_frame[candidate_columns], target_series, horizon)
    train_rows = tmp.training_rows(ds, origin_year, min_origin)
    if train_rows.shape[0] < 6:
        return None

    if frozen_spec is not None:
        features = [c for c in frozen_spec["features"] if c in train_rows.columns]
        train_rows = train_rows.dropna(subset=features)
        if train_rows.shape[0] < 6 or not features:
            return None
        alpha, l1_ratio = frozen_spec["alpha"], frozen_spec["l1_ratio"]
        source = "etape2_fige"
    else:
        X = train_rows[candidate_columns].dropna(axis=1, how="all").dropna(axis=0)
        if X.shape[0] < 6 or X.shape[1] == 0:
            return None
        y = train_rows.loc[X.index, "y"]
        features = sel.filter_redundant(X, redundancy_threshold, priority_order=candidate_columns)
        tuning = sel.tune_elasticnet_walk_forward(X[features], y, alphas, l1_ratios, min_train=6, n_folds=5, random_state=random_state)
        alpha, l1_ratio = tuning["alpha"], tuning["l1_ratio"]
        train_rows = train_rows.dropna(subset=features)
        source = "etape3_nouveau"

    X_train, y_train = train_rows[features], train_rows["y"]
    test_row = tmp.test_row(ds, origin_year)
    if test_row.empty or test_row[features].isna().any(axis=None):
        return None

    model, scaler, selected_nonzero = sel.fit_elasticnet_final(X_train, y_train, alpha, l1_ratio, random_state)
    pred = float(model.predict(scaler.transform(test_row[features]))[0])
    target_year = int(test_row["target_year"].iloc[0])

    return {
        "origin": origin_year, "target_year": target_year, "horizon": horizon,
        "prediction": pred, "n_train": len(X_train), "features": features,
        "alpha": alpha, "l1_ratio": l1_ratio, "source_specification": source,
        "model": model, "scaler": scaler,
    }


def project_country_all_horizons(
    feature_frame: pd.DataFrame,
    target_series: pd.Series,
    country: str,
    target: str,
    horizons: list[int],
    candidate_columns: list[str],
    origin_year: int,
    min_origin: int,
    redundancy_threshold: float,
    alphas: list[float],
    l1_ratios: list[float],
    frozen_specs_by_horizon: dict[int, dict] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        frozen = (frozen_specs_by_horizon or {}).get(h)
        res = fit_and_project(
            feature_frame, target_series, h, candidate_columns, origin_year, min_origin,
            redundancy_threshold, alphas, l1_ratios, frozen_spec=frozen, random_state=random_state,
        )
        if res is None:
            continue
        rows.append({
            "pays": country, "cible": target, "horizon": h,
            "origine": res["origin"], "annee_projetee": res["target_year"],
            "valeur_projetee": res["prediction"], "n_entrainement": res["n_train"],
            "specification": res["source_specification"],
        })
    return pd.DataFrame(rows)
