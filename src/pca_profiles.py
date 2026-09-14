"""ACP1 - Profils moyens par pays (Etape 1, UC-S1).

Une ligne par pays, une colonne par indicateur : moyenne des annees
disponibles sur 1991-2024 (ou sous-fenetre documentee si le pays n'a pas
d'observation sur toute la periode pour une variable donnee). Standardisation
puis ACP sur matrice de correlations. Analyse exploratoire : avec 9 pays, le
rang centre maximal est 8.
"""
from __future__ import annotations

import pandas as pd

from .io_utils import indicator_columns
from .pca_core import PCAResult, run_pca


def build_country_profiles(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    min_years_required: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit la matrice pays x indicateurs (moyennes) et la table de
    couverture (nombre d'annees utilisees par couple pays-variable)."""
    sub = df[(df["year"] >= year_start) & (df["year"] <= year_end)]
    ind_cols = indicator_columns(sub)

    profiles = sub.groupby("country_iso3")[ind_cols].mean()

    coverage_rows = []
    for country, g in sub.groupby("country_iso3"):
        for col in ind_cols:
            valid = g[col].dropna()
            coverage_rows.append({
                "country_iso3": country,
                "variable": col,
                "n_annees_utilisees": len(valid),
                "premiere_annee": int(g.loc[valid.index, "year"].min()) if len(valid) else None,
                "derniere_annee": int(g.loc[valid.index, "year"].max()) if len(valid) else None,
                "sous_seuil_robustesse": len(valid) < min_years_required,
            })
    coverage = pd.DataFrame(coverage_rows)
    return profiles, coverage


def run_acp1(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    min_years_required: int = 5,
) -> tuple[PCAResult, pd.DataFrame, pd.DataFrame]:
    profiles, coverage = build_country_profiles(df, year_start, year_end, min_years_required)
    variables = list(profiles.columns)
    result = run_pca(profiles, variables, profiles.index)
    return result, profiles, coverage


def run_acp1_leave_one_country_out(
    df: pd.DataFrame, year_start: int, year_end: int,
) -> dict[str, PCAResult]:
    """Sensibilite : refait l'ACP1 en retirant tour a tour chaque pays, pour
    verifier la stabilite de la structure des axes (petite taille
    d'echantillon, n=9)."""
    profiles, _ = build_country_profiles(df, year_start, year_end)
    variables = list(profiles.columns)
    out = {}
    for country in profiles.index:
        sub_profiles = profiles.drop(index=country)
        out[country] = run_pca(sub_profiles, variables, sub_profiles.index)
    return out
