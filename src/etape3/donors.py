"""Grille d'eligibilite des donneurs (cadrage Ch.3.2) et optimisation de
portefeuille (Ch.3.2.5, Eq. 3.12-3.13).

Score S_POC (exploratoire) : construit uniquement a partir des composantes
calculables avec les 30 indicateurs disponibles -- S2 (structure
productive), S3 (ouverture commerciale), S6 (completude des donnees), S7
partiel (proxy de transformation). S1 (PIB/habitant PPA), S4 (contraintes
en ressources) et S5 (qualite institutionnelle WGI) sont ABSENTS de cette
base et donc explicitement exclus, poids renormalises sur les 4 composantes
restantes. S_POC n'est jamais compare au seuil 0.55 du cadrage (Eq. 3.11),
qui s'applique au score complet S_c a 7 criteres -- non reconstituable ici.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.etape2.features import reindex_annual


def _s2_structure(df_country: pd.DataFrame, df_mar: pd.DataFrame) -> float:
    """Eq. 3.4 du cadrage : dc = sqrt(sum((v_j,c - v_j,MAR)^2)), formule
    definie pour des parts EXPRIMEES EN FRACTIONS (0-1), bornant la
    distance max a sqrt(2) (deux economies mono-sectorielles opposees).
    VAAG/VAIN/VASE sont stockees en POINTS DE POURCENTAGE (0-100) dans ce
    projet : conversion /100 indispensable, sans quoi s2 est nul pour
    toute paire de pays (bug repere et corrige ici)."""
    cols = ["VAAG", "VAIN", "VASE"]
    vc = df_country[cols].mean() / 100
    vm = df_mar[cols].mean() / 100
    if vc.isna().any() or vm.isna().any():
        return np.nan
    d = np.sqrt(((vc - vm) ** 2).sum())
    return max(0.0, 1 - d / np.sqrt(2))


def _s3_ouverture(df_country: pd.DataFrame, df_mar: pd.DataFrame, tau_o: float) -> float:
    oc = df_country["OUVERTURE_COM"].mean()
    om = df_mar["OUVERTURE_COM"].mean()
    if pd.isna(oc) or pd.isna(om) or tau_o == 0:
        return np.nan
    return max(0.0, 1 - abs(oc - om) / tau_o)


def _s6_completude(df_country: pd.DataFrame, indicator_cols: list[str]) -> float:
    n_expected = df_country.shape[0] * len(indicator_cols)
    n_valid = df_country[indicator_cols].notna().sum().sum()
    return n_valid / n_expected if n_expected else np.nan


def _s7_partiel(df_country: pd.DataFrame, all_candidates_fbcf_delta: pd.Series) -> float:
    """Proxy partiel (une seule dimension disponible sur les 5 du cadrage) :
    progression de la part de FBCF dans le PIB entre debut et fin de
    fenetre, normalisee en min-max sur l'ensemble des candidats. Documente
    explicitement comme proxy partiel, pas la mesure complete du cadrage."""
    years = df_country["year"].to_numpy()
    if len(years) < 6:
        return np.nan
    early = df_country[df_country["year"] <= years.min() + 4]["FBCF"].mean()
    late = df_country[df_country["year"] >= years.max() - 4]["FBCF"].mean()
    if pd.isna(early) or pd.isna(late):
        return np.nan
    delta = late - early
    lo, hi = all_candidates_fbcf_delta.min(), all_candidates_fbcf_delta.max()
    if hi == lo:
        return 0.5
    return float((delta - lo) / (hi - lo))


def compute_score_poc(
    panel_df: pd.DataFrame,
    focus_country: str,
    candidate_countries: list[str],
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """Calcule S_POC (et ses composantes) pour chaque pays candidat, sur la
    fenetre [year_start, year_end]. Renormalise les poids sur les 4
    composantes disponibles (poids egaux par defaut : 0.25 chacune -- choix
    documente, le cadrage ne fournissant pas de poids pour un score
    partiel)."""
    sub = panel_df[(panel_df["year"] >= year_start) & (panel_df["year"] <= year_end)]
    df_mar = sub[sub["country_iso3"] == focus_country]

    indicator_cols = [c for c in sub.columns if c not in ("country_iso3", "year")]
    tau_o = sub["OUVERTURE_COM"].std()

    fbcf_deltas = {}
    for c in candidate_countries:
        dc = sub[sub["country_iso3"] == c]
        years = dc["year"].to_numpy()
        if len(years) < 6:
            fbcf_deltas[c] = np.nan
            continue
        early = dc[dc["year"] <= years.min() + 4]["FBCF"].mean()
        late = dc[dc["year"] >= years.max() - 4]["FBCF"].mean()
        fbcf_deltas[c] = late - early if pd.notna(early) and pd.notna(late) else np.nan
    fbcf_delta_series = pd.Series(fbcf_deltas)

    rows = []
    for c in candidate_countries:
        dc = sub[sub["country_iso3"] == c]
        s2 = _s2_structure(dc, df_mar)
        s3 = _s3_ouverture(dc, df_mar, tau_o)
        s6 = _s6_completude(dc, indicator_cols)
        s7 = _s7_partiel(dc, fbcf_delta_series)
        components = {"S2_structure": s2, "S3_ouverture": s3, "S6_completude": s6, "S7_partiel": s7}
        valid = {k: v for k, v in components.items() if pd.notna(v)}
        s_poc = float(np.mean(list(valid.values()))) if valid else np.nan
        rows.append({"pays": c, **components, "n_composantes_valides": len(valid), "S_POC": s_poc})

    out = pd.DataFrame(rows).sort_values("S_POC", ascending=False).reset_index(drop=True)
    out.attrs["composantes_manquantes"] = ["S1_niveau_developpement", "S4_contraintes_ressources", "S5_qualite_institutionnelle"]
    out.attrs["seuil_0_55_applicable"] = False
    return out


def redundancy_matrix(panel_df: pd.DataFrame, candidate_countries: list[str], year_start: int, year_end: int, proxy_cols: list[str]) -> pd.DataFrame:
    """Similitude rho_c,d entre pays (Eq. 3.13), approximee par la
    correlation des profils moyens sur les leviers de transformation
    disponibles (proxy_cols), ramenee dans [0,1]. Chaque variable est
    d'abord standardisee ENTRE PAYS (z-score par colonne) : sans cette
    etape, des variables partageant une meme forme generale entre pays
    (ex. VASE > VAIN > VAAG pour la quasi-totalite des economies) gonflent
    artificiellement la correlation de paires de pays par ailleurs tres
    differents -- limite methodologique reperee et corrigee ici."""
    sub = panel_df[(panel_df["year"] >= year_start) & (panel_df["year"] <= year_end)]
    profiles = sub[sub["country_iso3"].isin(candidate_countries)].groupby("country_iso3")[proxy_cols].mean()
    standardized = (profiles - profiles.mean()) / profiles.std()
    corr = standardized.T.corr()
    rho = (corr + 1) / 2  # ramene [-1,1] -> [0,1]
    return rho


def portfolio_redundancy(rho: pd.DataFrame, portfolio: list[str]) -> float:
    if len(portfolio) < 2:
        return 0.0
    pairs = list(combinations(portfolio, 2))
    vals = [rho.loc[c, d] for c, d in pairs]
    return float(2 / (len(portfolio) * (len(portfolio) - 1)) * sum(vals))


def optimize_portfolio(
    scores: pd.DataFrame, rho: pd.DataFrame, m: int, diversity_lambda: float
) -> dict:
    """Optimise B* = argmax [mean(S_c) - lambda*R(B)] (Eq. 3.12) par
    recherche exhaustive sur les combinaisons de taille m (nombre de
    candidats <= 8 : exhaustif praticable, pas d'heuristique necessaire)."""
    candidates = scores["pays"].tolist()
    score_map = dict(zip(scores["pays"], scores["S_POC"]))
    best = None
    all_results = []
    for combo in combinations(candidates, m):
        mean_s = float(np.mean([score_map[c] for c in combo]))
        r = portfolio_redundancy(rho, list(combo))
        objective = mean_s - diversity_lambda * r
        result = {"portefeuille": combo, "score_moyen": mean_s, "redondance": r, "objectif": objective}
        all_results.append(result)
        if best is None or objective > best["objectif"]:
            best = result
    return {"meilleur": best, "tous": pd.DataFrame(all_results).sort_values("objectif", ascending=False).reset_index(drop=True)}
