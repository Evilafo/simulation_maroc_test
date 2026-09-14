"""Audit de qualite des donnees et verification du dictionnaire (Etape 1).

Ne modifie jamais les sources. Produit des tables de diagnostic destinees a
outputs/tables/ et sert de base a docs/MATRICE_CONFORMITE.md et
docs/AUDIT_DONNEES_ETAPE1.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .io_utils import indicator_columns


def check_duplicates_and_year_gaps(df: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    rows = []
    dup_mask = df.duplicated(subset=["country_iso3", "year"])
    n_dups_total = int(dup_mask.sum())
    expected_years = set(range(year_start, year_end + 1))
    for country, g in df.groupby("country_iso3"):
        years = set(g["year"].tolist())
        missing_years = sorted(expected_years - years)
        extra_years = sorted(years - expected_years)
        n_dups = int(g.duplicated(subset=["year"]).sum())
        rows.append({
            "country_iso3": country,
            "n_annees_presentes": len(years),
            "n_annees_attendues": len(expected_years),
            "annees_manquantes": ",".join(map(str, missing_years)) if missing_years else "",
            "annees_hors_fenetre": ",".join(map(str, extra_years)) if extra_years else "",
            "doublons_annee": n_dups,
        })
    out = pd.DataFrame(rows)
    out.attrs["n_doublons_pays_annee_total"] = n_dups_total
    return out


def completeness_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne = un etat pays-annee, avec le nombre de valeurs manquantes
    parmi les indicateurs du fichier."""
    ind_cols = indicator_columns(df)
    out = df[["country_iso3", "year"]].copy()
    out["n_indicateurs"] = len(ind_cols)
    out["n_manquants"] = df[ind_cols].isna().sum(axis=1).values
    out["etat_complet"] = out["n_manquants"] == 0
    return out


def missingness_by_indicator(df: pd.DataFrame) -> pd.DataFrame:
    ind_cols = indicator_columns(df)
    rows = []
    for col in ind_cols:
        miss = df[df[col].isna()]
        by_country = (
            miss.groupby("country_iso3")["year"]
            .apply(lambda s: f"{int(s.min())}-{int(s.max())} (n={len(s)})")
            .to_dict()
        )
        rows.append({
            "indicateur": col,
            "n_manquant": int(df[col].isna().sum()),
            "part_manquante": round(float(df[col].isna().mean()), 4),
            "pays_concernes": "; ".join(f"{k}: {v}" for k, v in by_country.items()) if by_country else "",
        })
    return pd.DataFrame(rows).sort_values("n_manquant", ascending=False).reset_index(drop=True)


def find_common_complete_window(df: pd.DataFrame, year_start: int, year_end: int) -> dict[str, Any]:
    """Cherche, en balayant les annees de fin de fenetre glissante croissante
    a partir de year_start, la premiere annee de depart telle que
    [depart, year_end] soit entierement complet (0 valeur manquante) pour
    toutes les combinaisons pays x indicateur presentes dans df."""
    ind_cols = indicator_columns(df)
    for candidate_start in range(year_start, year_end + 1):
        sub = df[(df["year"] >= candidate_start) & (df["year"] <= year_end)]
        n_missing = int(sub[ind_cols].isna().sum().sum())
        if n_missing == 0:
            return {
                "fenetre_commune_complete": [candidate_start, year_end],
                "n_annees": year_end - candidate_start + 1,
                "n_etats": len(sub),
            }
    return {"fenetre_commune_complete": None}


def constants_and_impossible_values(df: pd.DataFrame, registry: dict[str, Any] | None = None) -> pd.DataFrame:
    ind_cols = indicator_columns(df)
    rows = []
    reg_vars = (registry or {}).get("variables", {})
    for col in ind_cols:
        series = df[col]
        is_constant = series.nunique(dropna=True) <= 1
        n_inf = int(np.isinf(series.astype(float)).sum())
        allowed = reg_vars.get(col, {}).get("allowed_range")
        n_out_of_range = 0
        if allowed:
            lo, hi = allowed
            n_out_of_range = int(((series < lo) | (series > hi)).sum())
        rows.append({
            "indicateur": col,
            "constante": is_constant,
            "n_infini": n_inf,
            "plage_autorisee_registre": str(allowed) if allowed else "non trouvee dans le registre",
            "n_hors_plage": n_out_of_range,
            "min_observe": float(series.min()) if series.notna().any() else np.nan,
            "max_observe": float(series.max()) if series.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def cross_check_targets_dictionary(cfg: dict[str, Any], work_df: pd.DataFrame, registry: dict[str, Any]) -> pd.DataFrame:
    """Verifie chaque cible du mapping cadrage (config.targets) contre :
    - la presence effective de la colonne dans le fichier de travail
    - la definition du registre canonique (label, uc_s1_target)
    """
    rows = []
    reg_vars = registry.get("variables", {})
    for code, spec in cfg["targets"].items():
        col = spec["column"]
        in_work_base = col in work_df.columns
        reg_entry = reg_vars.get(col, {})
        rows.append({
            "code_cadrage": code,
            "cible_cadrage": spec["label"],
            "colonne_attendue": col,
            "presente_fichier_travail": in_work_base,
            "presente_registre": col in reg_vars,
            "uc_s1_target_registre": reg_entry.get("uc_s1_target"),
            "label_registre": reg_entry.get("label_fr"),
            "complement_documente": spec.get("complement", ""),
            "note": spec.get("note", ""),
        })
    return pd.DataFrame(rows)


def exact_linear_identities(df: pd.DataFrame) -> pd.DataFrame:
    """Verifie les identites comptables/compositionnelles exactes reperees
    dans le fichier de travail (mises en evidence par le rang deficient de
    3 observe empiriquement sur l'ACP2 en cas complets : 28 variables mais
    seulement 25 valeurs propres non nulles). A ne jamais combiner comme
    predicteurs simultanes d'une meme cible sans le savoir (etape 2)."""
    checks = []

    if {"SOBG", "RETO", "DETO"}.issubset(df.columns):
        sub = df.dropna(subset=["SOBG", "RETO", "DETO"])
        resid = sub["RETO"] - sub["DETO"] - sub["SOBG"]
        checks.append({
            "identite": "SOBG = RETO - DETO",
            "n_observations_testees": len(sub),
            "ecart_max_absolu": float(resid.abs().max()) if len(sub) else None,
            "exacte": bool(len(sub) and resid.abs().max() < 1e-6),
            "implication": "Ne jamais utiliser RETO et DETO simultanement comme predicteurs de SOBG (identite comptable, pas relation estimee).",
        })

    if {"OUVERTURE_COM", "EXPO", "IMPO"}.issubset(df.columns):
        sub = df.dropna(subset=["OUVERTURE_COM", "EXPO", "IMPO"])
        resid = sub["EXPO"] + sub["IMPO"] - sub["OUVERTURE_COM"]
        checks.append({
            "identite": "OUVERTURE_COM = EXPO + IMPO",
            "n_observations_testees": len(sub),
            "ecart_max_absolu": float(resid.abs().max()) if len(sub) else None,
            "exacte": bool(len(sub) and resid.abs().max() < 1e-6),
            "implication": "Ne pas inclure EXPO, IMPO et OUVERTURE_COM ensemble dans un meme modele lineaire (colinearite exacte par definition).",
        })

    if {"EMAG", "EMIN", "EMSE"}.issubset(df.columns):
        sub = df.dropna(subset=["EMAG", "EMIN", "EMSE"])
        resid = sub["EMAG"] + sub["EMIN"] + sub["EMSE"] - 100
        checks.append({
            "identite": "EMAG + EMIN + EMSE = 100",
            "n_observations_testees": len(sub),
            "ecart_max_absolu": float(resid.abs().max()) if len(sub) else None,
            "exacte": bool(len(sub) and resid.abs().max() < 1e-3),
            "implication": "Parts d'emploi sectoriel compositionnelles : n'en retenir que 2 sur 3 comme predicteurs independants.",
        })

    if {"VAAG", "VAIN", "VASE"}.issubset(df.columns):
        sub = df.dropna(subset=["VAAG", "VAIN", "VASE"])
        resid = sub["VAAG"] + sub["VAIN"] + sub["VASE"] - 100
        checks.append({
            "identite": "VAAG + VAIN + VASE = 100 (testee, NON verifiee exactement)",
            "n_observations_testees": len(sub),
            "ecart_max_absolu": float(resid.abs().max()) if len(sub) else None,
            "exacte": bool(len(sub) and resid.abs().max() < 1e-3),
            "implication": "Parts de valeur ajoutee : somme proche de 100 mais pas exacte (residuel/statistique non couvert) - pas une identite stricte, colinearite forte mais pas totale.",
        })

    return pd.DataFrame(checks)


def variables_registry_vs_work_base(work_df: pd.DataFrame, registry: dict[str, Any]) -> pd.DataFrame:
    """Compare l'ensemble des variables du registre a celles presentes dans
    le fichier de travail 28 indicateurs, pour rendre visibles toutes les
    exclusions (pas seulement DETTE_PUBLIQUE/TCER)."""
    ind_cols = set(indicator_columns(work_df))
    reg_vars = registry.get("variables", {})
    rows = []
    for name, spec in reg_vars.items():
        rows.append({
            "variable": name,
            "label_registre": spec.get("label_fr"),
            "categorie": spec.get("category"),
            "uc_s1_target": spec.get("uc_s1_target", False),
            "variable_type": spec.get("variable_type"),
            "status_registre": spec.get("status"),
            "presente_fichier_travail_28ind": name in ind_cols,
        })
    out = pd.DataFrame(rows).sort_values(
        ["presente_fichier_travail_28ind", "uc_s1_target", "variable"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return out
