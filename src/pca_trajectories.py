"""ACP2 - Trajectoires annuelles (Etape 1, UC-S1).

Une seule ACP globale sur les observations pays-annee, avec une echelle
commune (memes parametres de standardisation pour toutes les annees et tous
les pays). Les coordonnees sont ensuite reliees dans l'ordre chronologique
pour chaque pays. Ce n'est pas un modele factoriel dynamique : les axes ne
sont pas recalcules chaque annee.
"""
from __future__ import annotations

import pandas as pd

from .io_utils import indicator_columns
from .pca_core import PCAResult, run_pca


def build_complete_case_panel(df: pd.DataFrame, year_start: int, year_end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retient uniquement les etats pays-annee sans aucune valeur manquante
    (cas complets) sur la fenetre demandee. Retourne le panel complet et la
    table des etats exclus (incomplets), pour tracabilite."""
    sub = df[(df["year"] >= year_start) & (df["year"] <= year_end)].copy()
    ind_cols = indicator_columns(sub)
    n_missing = sub[ind_cols].isna().sum(axis=1)
    complete = sub[n_missing == 0].reset_index(drop=True)
    incomplete = sub[n_missing > 0][["country_iso3", "year"]].copy()
    incomplete["n_manquants"] = n_missing[n_missing > 0].values
    return complete, incomplete.reset_index(drop=True)


def run_acp2(df: pd.DataFrame, year_start: int, year_end: int) -> tuple[PCAResult, pd.DataFrame, pd.DataFrame]:
    complete, incomplete = build_complete_case_panel(df, year_start, year_end)
    ind_cols = indicator_columns(complete)
    index_labels = pd.MultiIndex.from_frame(complete[["country_iso3", "year"]])
    result = run_pca(complete, ind_cols, index_labels)

    # Reconstruit un tableau de coordonnees "plat" avec country_iso3/year en
    # colonnes explicites (plus pratique pour le trace des trajectoires).
    scores_flat = result.scores.reset_index()
    return result, scores_flat, incomplete
