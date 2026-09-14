"""Selection a trois niveaux (demande, section 2) :
1. plausibilite/disponibilite -> configs/config_etape2.yaml::target_mechanisms
   (applique en amont, au moment de choisir les colonnes passees a
   `features.build_feature_frame`)
2. filtrage des redondances (colinearite entre caracteristiques)
3. selection supervisee parcimonieuse (Elastic Net), reglee UNIQUEMENT sur
   les origines internes (target_year <= internal_train_end), jamais sur le
   bloc final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


def filter_redundant(X: pd.DataFrame, threshold: float, priority_order: list[str] | None = None) -> list[str]:
    """Retourne la liste des colonnes a conserver apres filtrage des paires
    fortement correlees (|corr| > threshold) calcule sur X (entrainement
    uniquement). En cas de paire redondante, conserve la colonne la plus
    prioritaire (ordre donne, sinon ordre d'apparition = retard le plus
    court / caracteristique la plus simple en general)."""
    cols = list(X.columns)
    if priority_order:
        cols = sorted(cols, key=lambda c: priority_order.index(c) if c in priority_order else len(priority_order))
    corr = X[cols].corr().abs()
    keep: list[str] = []
    dropped: set[str] = set()
    for c in cols:
        if c in dropped:
            continue
        keep.append(c)
        for other in cols:
            if other == c or other in dropped or other in keep:
                continue
            if corr.loc[c, other] > threshold:
                dropped.add(other)
    return keep


def _walk_forward_folds(origins: list[int], n_folds: int = 5, min_train: int = 6) -> list[tuple[list[int], int]]:
    """Genere des scissions (train_origins, val_origin) en fenetre
    expansive a l'interieur d'une liste d'origines triees (usage interne
    uniquement, jamais sur le bloc final)."""
    origins = sorted(origins)
    folds = []
    start = max(min_train, len(origins) - n_folds)
    for i in range(start, len(origins)):
        train = origins[:i]
        val = origins[i]
        if len(train) >= min_train:
            folds.append((train, val))
    return folds


def tune_elasticnet_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: list[float],
    l1_ratios: list[float],
    min_train: int = 6,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Reglage par validation croisee chronologique (walk-forward), jamais
    par K-fold aleatoire. X/y indexes par annee d'origine, deja restreints
    aux origines internes par l'appelant. Retourne les meilleurs
    hyperparametres et le detail des scores."""
    origins = list(X.index)
    folds = _walk_forward_folds(origins, n_folds=n_folds, min_train=min_train)

    if not folds:
        return {"alpha": alphas[len(alphas) // 2], "l1_ratio": l1_ratios[len(l1_ratios) // 2], "n_folds": 0, "note": "pas assez d'origines internes pour un reglage walk-forward ; hyperparametres medians retenus par defaut"}

    results = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            errors = []
            for train_o, val_o in folds:
                X_train, y_train = X.loc[train_o], y.loc[train_o]
                if X_train.shape[0] < 3 or X_train.shape[1] == 0:
                    continue
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X.loc[[val_o]])
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=20000)
                model.fit(X_train_s, y_train)
                pred = model.predict(X_val_s)[0]
                errors.append((pred - y.loc[val_o]) ** 2)
            if errors:
                results.append({"alpha": alpha, "l1_ratio": l1_ratio, "rmse": float(np.sqrt(np.mean(errors))), "n_val": len(errors)})

    if not results:
        return {"alpha": alphas[len(alphas) // 2], "l1_ratio": l1_ratios[len(l1_ratios) // 2], "n_folds": 0, "note": "aucun pli exploitable"}

    best = min(results, key=lambda r: r["rmse"])
    return {"alpha": best["alpha"], "l1_ratio": best["l1_ratio"], "n_folds": len(folds), "grid_results": results}


def tune_elasticnet_walk_forward_panel(
    X: pd.DataFrame,
    y: pd.Series,
    origin_years: pd.Series,
    alphas: list[float],
    l1_ratios: list[float],
    min_train_years: int = 6,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Variante panel de `tune_elasticnet_walk_forward` : les plis sont
    definis par ANNEE D'ORIGINE reelle (colonne `origin_years`), pas par
    position dans l'index -- indispensable des lors que X regroupe
    plusieurs pays (meme annee d'origine repetee), sans quoi un decoupage
    par position melangerait des pays sans respecter la chronologie.
    Entrainement et validation utilisent toutes les lignes (tous pays) de
    chaque cote de la coupure temporelle."""
    years = sorted(origin_years.unique())
    start = max(min_train_years, len(years) - n_folds)
    folds = [(years[:i], years[i]) for i in range(start, len(years))]

    if not folds:
        return {"alpha": alphas[len(alphas) // 2], "l1_ratio": l1_ratios[len(l1_ratios) // 2], "n_folds": 0, "note": "pas assez d'annees internes pour un reglage walk-forward panel"}

    results = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            errors = []
            for train_years, val_year in folds:
                train_mask = origin_years.isin(train_years)
                val_mask = origin_years == val_year
                if train_mask.sum() < 5 or val_mask.sum() == 0:
                    continue
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X.loc[train_mask])
                X_val_s = scaler.transform(X.loc[val_mask])
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=20000)
                model.fit(X_train_s, y.loc[train_mask])
                preds = model.predict(X_val_s)
                errors.extend(((preds - y.loc[val_mask].to_numpy()) ** 2).tolist())
            if errors:
                results.append({"alpha": alpha, "l1_ratio": l1_ratio, "rmse": float(np.sqrt(np.mean(errors))), "n_val": len(errors)})

    if not results:
        return {"alpha": alphas[len(alphas) // 2], "l1_ratio": l1_ratios[len(l1_ratios) // 2], "n_folds": 0, "note": "aucun pli exploitable"}

    best = min(results, key=lambda r: r["rmse"])
    return {"alpha": best["alpha"], "l1_ratio": best["l1_ratio"], "n_folds": len(folds), "grid_results": results}


def fit_elasticnet_final(
    X: pd.DataFrame, y: pd.Series, alpha: float, l1_ratio: float, random_state: int = 42
) -> tuple[ElasticNet, StandardScaler, list[str]]:
    """Reajuste l'Elastic Net avec les hyperparametres geles sur l'ensemble
    d'entrainement fourni (deja restreint a target_year <= origine
    courante par l'appelant, cf. temporal.training_rows)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=20000)
    model.fit(X_s, y)
    selected = [c for c, coef in zip(X.columns, model.coef_) if abs(coef) > 1e-10]
    return model, scaler, selected
