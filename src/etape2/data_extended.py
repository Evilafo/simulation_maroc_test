"""Jeu de travail etendu pour l'Etape 2 (30 indicateurs, 1991-2024, 9 pays).

Reintegre DETTE_PUBLIQUE et TCER depuis la base source
(base_B_revised_wide.csv), verifies identiques aux 28 colonnes du fichier
de travail de l'etape 1 sur la fenetre commune (cf.
configs/config_etape2.yaml -> reintegrated_variables et
docs/SUIVI_MODELISATION.md, decision qui met a jour le point ouvert #1 de
l'etape 1).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.io_utils import load_config, load_source_base, resolve_path


def load_extended_work_base(
    cfg_etape1: dict,
    cfg_etape2: dict,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Retourne le panel 1991-2024, 9 pays, 30 indicateurs (28 de l'etape 1
    + DETTE_PUBLIQUE + TCER), filtre depuis la base source et verifie
    identique aux colonnes deja utilisees en etape 1."""
    year_start = cfg_etape2["scope"]["year_start"]
    year_end = cfg_etape2["scope"]["year_end"]
    countries = cfg_etape2["scope"]["countries"]

    source_df = load_source_base(cfg_etape1, logger)
    sub = source_df[
        (source_df["year"] >= year_start)
        & (source_df["year"] <= year_end)
        & (source_df["country_iso3"].isin(countries))
    ].reset_index(drop=True)

    expected_rows = len(countries) * (year_end - year_start + 1)
    if len(sub) != expected_rows:
        raise ValueError(
            f"Jeu etendu : {len(sub)} lignes obtenues, {expected_rows} attendues "
            f"({len(countries)} pays x {year_end - year_start + 1} annees)."
        )

    if logger:
        logger.info(
            "Jeu de travail etendu construit : shape=%s (28 indicateurs etape 1 + DETTE_PUBLIQUE + TCER)",
            sub.shape,
        )
    return sub


def verify_against_etape1_work_base(
    extended_df: pd.DataFrame, cfg_etape1: dict, logger: logging.Logger | None = None
) -> bool:
    """Confirme que les 28 colonnes partagees avec le fichier de travail de
    l'etape 1 sont identiques (garde-fou de non-regression, deja verifie
    manuellement lors de la construction de configs/config_etape2.yaml)."""
    from src.io_utils import load_work_base

    work_df = load_work_base(cfg_etape1, logger)
    shared = [c for c in work_df.columns if c in extended_df.columns and c not in ("country_iso3", "year")]
    merged = work_df.merge(extended_df, on=["country_iso3", "year"], suffixes=("_etape1", "_etendu"))
    ok = True
    for c in shared:
        a, b = merged[f"{c}_etape1"], merged[f"{c}_etendu"]
        both_na = a.isna() & b.isna()
        diff = (a - b).abs()
        diff[both_na] = 0
        if diff.max() > 1e-6:
            ok = False
            if logger:
                logger.warning("Ecart detecte sur %s : max_abs_diff=%.6f", c, diff.max())
    if logger:
        logger.info("Verification jeu etendu vs fichier de travail etape 1 : %s (%d colonnes comparees)", "OK" if ok else "ECART", len(shared))
    return ok
