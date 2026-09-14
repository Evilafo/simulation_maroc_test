"""Poids de combinaison (Eq. 3.14). Deux categories distinctes, jamais
melangees :

- poids STATISTIQUES : estimes par un programme quadratique sous
  contraintes (omega>=0, somme=1), regularise (L2 leger, justifie par le
  faible nombre de donneurs et l'historique court), au sens du controle
  synthetique cite par le cadrage (Abadie et al.) -- reproduisent au mieux
  la trajectoire historique marocaine. Utilisables pour la reference
  centrale.
- poids POLITIQUES : hypotheses normatives fixees (ex. omega_VNM=1).
  RESERVES aux scenarios normatifs, jamais a la prevision centrale
  (garde-fou explicite du cadrage Ch.3.5.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _solve_synthetic_weights(Y_mar: np.ndarray, X_donors: np.ndarray, l2_reg: float) -> np.ndarray:
    """omega* = argmin ||Y_mar - X_donors @ omega||^2 + l2_reg*||omega||^2,
    sous omega>=0, somme(omega)=1. X_donors : (T, n_donors)."""
    n = X_donors.shape[1]
    w0 = np.full(n, 1 / n)

    def objective(w):
        resid = Y_mar - X_donors @ w
        return float(resid @ resid + l2_reg * (w @ w))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
    return w


def estimate_common_weights(
    panel_df: pd.DataFrame,
    focus_country: str,
    donors: list[str],
    targets: list[str],
    year_start: int,
    year_end: int,
    l2_reg: float,
) -> dict:
    """Poids statistiques communs (un seul vecteur omega), objectif poole
    sur les cibles standardisees (moyenne/ecart-type marocains sur la
    fenetre) pour equilibrer les unites, comme demande."""
    sub = panel_df[(panel_df["year"] >= year_start) & (panel_df["year"] <= year_end)]
    mar = sub[sub["country_iso3"] == focus_country].set_index("year")

    y_parts, x_parts = [], []
    for t in targets:
        mu, sigma = mar[t].mean(), mar[t].std()
        if pd.isna(sigma) or sigma == 0:
            continue
        y_t = (mar[t] - mu) / sigma
        x_cols = []
        valid = True
        for d in donors:
            dd = sub[sub["country_iso3"] == d].set_index("year")
            if t not in dd.columns:
                valid = False
                break
            x_cols.append((dd[t] - mu) / sigma)
        if not valid:
            continue
        common_years = y_t.dropna().index
        for xc in x_cols:
            common_years = common_years.intersection(xc.dropna().index)
        if len(common_years) < 5:
            continue
        y_parts.append(y_t.loc[common_years].to_numpy())
        x_parts.append(np.column_stack([xc.loc[common_years].to_numpy() for xc in x_cols]))

    if not y_parts:
        return {"weights": {d: np.nan for d in donors}, "n_targets_used": 0}

    Y = np.concatenate(y_parts)
    X = np.concatenate(x_parts, axis=0)
    w = _solve_synthetic_weights(Y, X, l2_reg)
    return {"weights": dict(zip(donors, w)), "n_targets_used": len(y_parts), "n_obs_pooled": len(Y)}


def estimate_per_target_weights(
    panel_df: pd.DataFrame,
    focus_country: str,
    donors: list[str],
    targets: list[str],
    year_start: int,
    year_end: int,
    l2_reg: float,
) -> pd.DataFrame:
    sub = panel_df[(panel_df["year"] >= year_start) & (panel_df["year"] <= year_end)]
    mar = sub[sub["country_iso3"] == focus_country].set_index("year")
    rows = []
    for t in targets:
        y = mar[t].dropna()
        x_series = {}
        ok = True
        for d in donors:
            dd = sub[sub["country_iso3"] == d].set_index("year")
            if t not in dd.columns:
                ok = False
                break
            x_series[d] = dd[t]
        if not ok:
            continue
        common_years = y.index
        for s in x_series.values():
            common_years = common_years.intersection(s.dropna().index)
        if len(common_years) < 5:
            rows.append({"cible": t, **{d: np.nan for d in donors}, "n_obs": len(common_years)})
            continue
        Y = y.loc[common_years].to_numpy()
        X = np.column_stack([x_series[d].loc[common_years].to_numpy() for d in donors])
        w = _solve_synthetic_weights(Y, X, l2_reg)
        rows.append({"cible": t, **dict(zip(donors, w)), "n_obs": len(common_years)})
    return pd.DataFrame(rows)


def weight_coherence_across_targets(per_target_weights: pd.DataFrame, donors: list[str]) -> pd.DataFrame:
    """Verifie la coherence des poids par cible (variante) : correlation de
    rang et distance L1 entre chaque paire de vecteurs de poids par cible."""
    valid = per_target_weights.dropna(subset=donors)
    rows = []
    targets = valid["cible"].tolist()
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            wi = valid.iloc[i][donors].to_numpy(dtype=float)
            wj = valid.iloc[j][donors].to_numpy(dtype=float)
            l1 = float(np.abs(wi - wj).sum())
            corr = float(np.corrcoef(wi, wj)[0, 1]) if wi.std() > 0 and wj.std() > 0 else np.nan
            rows.append({"cible_1": targets[i], "cible_2": targets[j], "distance_L1": l1, "correlation": corr})
    return pd.DataFrame(rows)


def apply_political_scenario(donors: list[str], scenario_weights: dict[str, float]) -> dict[str, float]:
    """Complete un scenario politique avec des poids nuls pour les
    donneurs non mentionnes (somme deja verifiee = 1 par config_check)."""
    return {d: scenario_weights.get(d, 0.0) for d in donors}
