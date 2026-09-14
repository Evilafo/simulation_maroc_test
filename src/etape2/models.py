"""Portefeuille predictif : Elastic Net (pays unique, prevision directe par
horizon) et panel pooled parcimonieux (avec/sans effets fixes pays, et
variante leave-one-country-out).

Principe commun de gel des choix (demande, section 1) : les
hyperparametres et la liste de caracteristiques retenues sont choisis UNE
SEULE FOIS a partir du bloc interne (target_year <= internal_train_end),
via validation croisee chronologique interne a ce bloc (cf. selection.py),
puis geles. Chaque origine (interne ou finale) ne fait que reajuster les
coefficients du modele sur les donnees strictement anterieures a cette
origine (cf. temporal.training_rows) -- jamais de nouveau reglage
d'hyperparametres sur le bloc final.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import selection as sel
from . import temporal as tmp


@dataclass
class WalkForwardResult:
    target: str
    horizon: int
    frozen_features: list[str]
    frozen_alpha: float
    frozen_l1_ratio: float
    tuning_detail: dict
    forecasts: pd.DataFrame  # origin, target_year, y_true, y_pred, block ("interne"/"finale")
    deployable_models: dict = field(default_factory=dict)  # "interne"/"finale" -> {"origin", "model", "scaler"}


def run_elasticnet_direct_h(
    feature_frame: pd.DataFrame,
    target_series: pd.Series,
    target: str,
    horizon: int,
    candidate_columns: list[str],
    internal_train_end: int,
    final_start: int,
    final_end: int,
    min_origin: int,
    max_origin: int,
    redundancy_threshold: float,
    alphas: list[float],
    l1_ratios: list[float],
    random_state: int = 42,
) -> WalkForwardResult | None:
    ds = tmp.build_direct_dataset(feature_frame[candidate_columns], target_series, horizon)
    origins = tmp.evaluation_origins(ds, internal_train_end, final_start, final_end, min_origin, max_origin)
    if not origins["internal"]:
        return None

    # --- Gel des caracteristiques et des hyperparametres sur le bloc interne ---
    internal_train_max = max(origins["internal"])
    internal_rows = tmp.training_rows(ds, internal_train_max, min_origin)
    if internal_rows.shape[0] < 6:
        return None
    X_internal = internal_rows[candidate_columns].dropna(axis=1, how="all")
    X_internal = X_internal.dropna(axis=0)
    y_internal = internal_rows.loc[X_internal.index, "y"]
    if X_internal.shape[0] < 6 or X_internal.shape[1] == 0:
        return None

    frozen_features = sel.filter_redundant(X_internal, redundancy_threshold, priority_order=candidate_columns)
    X_internal = X_internal[frozen_features]

    tuning = sel.tune_elasticnet_walk_forward(X_internal, y_internal, alphas, l1_ratios, min_train=6, n_folds=5, random_state=random_state)
    alpha, l1_ratio = tuning["alpha"], tuning["l1_ratio"]

    # --- Walk-forward : reajustement des coefficients a chaque origine ---
    records = []
    deployable_models: dict = {}
    for block_name, block_origins in origins.items():
        block_label = "interne" if block_name == "internal" else "finale"
        last_origin_in_block = max(block_origins) if block_origins else None
        for origin in block_origins:
            train_rows = tmp.training_rows(ds, origin, min_origin)
            train_rows = train_rows.dropna(subset=frozen_features)
            if train_rows.shape[0] < 6:
                continue
            X_train, y_train = train_rows[frozen_features], train_rows["y"]
            test_row = tmp.test_row(ds, origin)
            if test_row.empty or test_row[frozen_features].isna().any(axis=None):
                continue
            model, scaler, selected_nonzero = sel.fit_elasticnet_final(X_train, y_train, alpha, l1_ratio, random_state)
            X_test_s = scaler.transform(test_row[frozen_features])
            pred = float(model.predict(X_test_s)[0])
            target_year = int(test_row["target_year"].iloc[0])
            y_true = target_series.get(target_year, np.nan)
            records.append({
                "origin": origin, "target_year": target_year, "y_true": y_true, "y_pred": pred,
                "block": block_label,
                "n_train": len(X_train), "n_features_non_nulles": len(selected_nonzero),
            })
            if origin == last_origin_in_block:
                # Objet deployable : dernier reajustement de chaque bloc,
                # conserve tel quel (jamais ecrase par une reestimation
                # ulterieure -- cf. consigne de non-ecrasement des objets
                # necessaires au backtest).
                deployable_models[block_label] = {
                    "origin": origin, "target_year": target_year,
                    "model": model, "scaler": scaler, "features": frozen_features,
                }

    if not records:
        return None

    return WalkForwardResult(
        target=target, horizon=horizon, frozen_features=frozen_features,
        frozen_alpha=alpha, frozen_l1_ratio=l1_ratio, tuning_detail=tuning,
        forecasts=pd.DataFrame(records), deployable_models=deployable_models,
    )


def run_panel_pooled_direct_h(
    panel_feature_frames: dict[str, pd.DataFrame],
    panel_target_series: dict[str, pd.Series],
    target: str,
    horizon: int,
    candidate_columns: list[str],
    internal_train_end: int,
    final_start: int,
    final_end: int,
    min_origin: int,
    max_origin: int,
    redundancy_threshold: float,
    alphas: list[float],
    l1_ratios: list[float],
    focus_country: str,
    exclude_country_from_training: str | None = None,
    use_country_dummies: bool = True,
    random_state: int = 42,
) -> WalkForwardResult | None:
    """Panel pooled Elastic Net. Si `exclude_country_from_training` est
    fourni (leave-one-country-out), ce pays est retire de TOUT
    l'entrainement et de la construction des transformations (redondance,
    standardisation) -- pas seulement de la ligne de test -- et aucun effet
    fixe pays n'est utilise (un effet fixe non estime pour un pays absent de
    l'apprentissage ne peut pas etre applique a sa prevision)."""
    countries_train = [c for c in panel_feature_frames if c != exclude_country_from_training]
    dummies_ok = use_country_dummies and exclude_country_from_training is None

    per_country_ds = {
        c: tmp.build_direct_dataset(panel_feature_frames[c][candidate_columns], panel_target_series[c], horizon)
        for c in panel_feature_frames
    }

    focus_origins = tmp.evaluation_origins(per_country_ds[focus_country], internal_train_end, final_start, final_end, min_origin, max_origin)
    if not focus_origins["internal"]:
        return None

    def pooled_rows(origin: int, countries: list[str]) -> pd.DataFrame:
        """Concatene les lignes d'entrainement de plusieurs pays. Reindexe
        immediatement sur une plage unique (0..n-1) : les DataFrames par
        pays partagent la meme etiquette d'index (l'annee d'origine), donc
        un `.loc[index, ...]` sur la table concatenee sans reindexation
        prealable duplique silencieusement les lignes (piege classique des
        index non uniques de pandas)."""
        parts = []
        for c in countries:
            r = tmp.training_rows(per_country_ds[c], origin, min_origin).copy()
            if r.empty:
                continue
            r["origin_year"] = r.index
            r["country_iso3"] = c
            parts.append(r)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, axis=0).reset_index(drop=True)

    internal_train_max = max(focus_origins["internal"])
    internal_rows = pooled_rows(internal_train_max, countries_train)
    if internal_rows.shape[0] < 10:
        return None
    X_internal = internal_rows[candidate_columns].dropna(axis=1, how="all")
    X_internal = X_internal.dropna(axis=0)
    if X_internal.shape[0] < 10 or X_internal.shape[1] == 0:
        return None
    y_internal = internal_rows.loc[X_internal.index, "y"]

    frozen_features = sel.filter_redundant(X_internal, redundancy_threshold, priority_order=candidate_columns)
    origin_years_internal = internal_rows.loc[X_internal.index, "origin_year"]

    tuning = sel.tune_elasticnet_walk_forward_panel(
        X_internal[frozen_features], y_internal, origin_years_internal,
        alphas, l1_ratios, min_train_years=6, n_folds=5, random_state=random_state,
    )
    alpha, l1_ratio = tuning["alpha"], tuning["l1_ratio"]

    records = []
    for block_name, block_origins in focus_origins.items():
        for origin in block_origins:
            train_rows = pooled_rows(origin, countries_train)
            if train_rows.empty:
                continue
            train_rows = train_rows.dropna(subset=frozen_features)
            if train_rows.shape[0] < 10:
                continue
            X_train = train_rows[frozen_features].copy()
            if dummies_ok:
                dummies = pd.get_dummies(train_rows["country_iso3"], prefix="pays", drop_first=True)
                X_train = pd.concat([X_train.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            y_train = train_rows["y"].reset_index(drop=True)

            test_row = tmp.test_row(per_country_ds[focus_country], origin)
            if test_row.empty or test_row[frozen_features].isna().any(axis=None):
                continue
            X_test = test_row[frozen_features].copy()
            if dummies_ok:
                for col in [c for c in X_train.columns if c.startswith("pays_")]:
                    X_test[col] = 1 if col == f"pays_{focus_country}" else 0
                X_test = X_test[X_train.columns]

            model, scaler, selected_nonzero = sel.fit_elasticnet_final(X_train, y_train, alpha, l1_ratio, random_state)
            X_test_s = scaler.transform(X_test)
            pred = float(model.predict(X_test_s)[0])
            target_year = int(test_row["target_year"].iloc[0])
            y_true = panel_target_series[focus_country].get(target_year, np.nan)
            records.append({
                "origin": origin, "target_year": target_year, "y_true": y_true, "y_pred": pred,
                "block": "interne" if block_name == "internal" else "finale",
                "n_train": len(X_train), "n_pays_entrainement": train_rows["country_iso3"].nunique(),
                "effets_fixes_pays": dummies_ok,
            })

    if not records:
        return None

    return WalkForwardResult(
        target=target, horizon=horizon, frozen_features=frozen_features,
        frozen_alpha=alpha, frozen_l1_ratio=l1_ratio, tuning_detail=tuning,
        forecasts=pd.DataFrame(records),
    )
