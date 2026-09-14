"""Caracteristiques causales (retards, differences, statistiques mobiles),
calculees separement par pays apres reindexation annuelle stricte
(consigne : jamais de retard/fenetre a cheval sur une frontiere de pays).

Toutes les fonctions utilisent uniquement des operations causales
(`shift`, fenetres arriere) : la valeur a l'annee O ne depend jamais d'une
observation posterieure a O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reindex_annual(df_country: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    """Reindexe un pays sur une grille annuelle complete (year_start..year_end),
    trie chronologiquement. N'introduit aucune valeur : les annees deja
    manquantes du fichier de travail restent NaN."""
    out = df_country.set_index("year").reindex(range(year_start, year_end + 1))
    return out.sort_index()


def build_feature_frame(
    df_country: pd.DataFrame,
    columns: list[str],
    max_lags: int,
    ma_windows: list[int],
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """Construit, pour un seul pays, un tableau indexe par annee avec :
    niveau courant (colonne brute), retards 1..max_lags, difference
    premiere, et moyennes/ecarts-types mobiles passes (fenetre se terminant
    a l'annee courante INCLUSE, donc utilisable a l'origine O). Le log
    n'est calcule que si la serie est strictement positive sur toute la
    fenetre (sinon colonne omise, pas de log de valeur negative/nulle)."""
    base = reindex_annual(df_country, year_start, year_end)
    feats = pd.DataFrame(index=base.index)

    for col in columns:
        if col not in base.columns:
            continue
        s = base[col]
        feats[f"{col}_lag0"] = s
        for lag in range(1, max_lags + 1):
            feats[f"{col}_lag{lag}"] = s.shift(lag)
        feats[f"{col}_diff1"] = s.diff(1)
        for w in ma_windows:
            feats[f"{col}_ma{w}"] = s.rolling(window=w, min_periods=w).mean().shift(1)
            feats[f"{col}_std{w}"] = s.rolling(window=w, min_periods=w).std().shift(1)
        if (s.dropna() > 0).all() and s.notna().any():
            feats[f"{col}_log"] = np.log(s)
            feats[f"{col}_logdiff1"] = np.log(s).diff(1)

    return feats


def drop_all_nan_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(axis=1, how="all")
