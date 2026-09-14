"""Simulation Monte Carlo jointe (cadrage Ch.6.4).

Simule conjointement les 7 cibles (jamais des bandes marginales
independantes, pour ne pas compter deux fois un meme choc macroeconomique
commun -- ex. un choc de croissance affecte simultanement chomage et solde
budgetaire). La matrice de covariance des residus historiques (modele
Elastic Net Maroc, horizon 1, etape 2) est regularisee par un retrait
("shrinkage") vers sa diagonale, justifie par le faible effectif
disponible (T~30) relativement au nombre de cibles (7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_residual_matrix(forecasts_df: pd.DataFrame, targets: list[str], horizon: int, model: str = "elasticnet") -> pd.DataFrame:
    """Recupere, pour un horizon donne, la matrice (annee_cible x cible)
    des residus (y_pred - y_true) du modele Elastic Net Maroc sur les blocs
    interne + finale de l'etape 2 (maximise l'effectif disponible pour
    estimer la covariance)."""
    sub = forecasts_df[(forecasts_df["modele"] == model) & (forecasts_df["horizon"] == horizon) & (forecasts_df["cible"].isin(targets))].copy()
    sub["residu"] = sub["y_pred"] - sub["y_true"]
    wide = sub.pivot_table(index="target_year", columns="cible", values="residu")
    return wide[targets]


def shrinkage_covariance(residuals: pd.DataFrame, shrinkage_intensity: float) -> pd.DataFrame:
    """Sigma_shrunk = (1-delta)*Sigma_empirique + delta*diag(Sigma_empirique).
    Approche simple et documentee (pas l'estimateur optimal de Ledoit-Wolf),
    suffisante pour stabiliser une covariance 7x7 estimee sur ~20-30 points."""
    clean = residuals.dropna()
    sigma_emp = clean.cov()
    diag_only = pd.DataFrame(np.diag(np.diag(sigma_emp)), index=sigma_emp.index, columns=sigma_emp.columns)
    sigma_shrunk = (1 - shrinkage_intensity) * sigma_emp + shrinkage_intensity * diag_only
    return sigma_shrunk, sigma_emp, len(clean)


def extrapolated_covariance(base_cov: pd.DataFrame, base_horizon: int, target_horizon: int) -> pd.DataFrame:
    """Pour les horizons d'extrapolation (6-10, non evalues en etape 2),
    fait croitre la variance selon une hypothese de marche aleatoire
    (variance proportionnelle a l'horizon), en conservant la structure de
    correlation de la covariance de base. Hypothese documentee, pas une
    estimation directe (aucune realisation disponible a ces horizons)."""
    scale = target_horizon / base_horizon
    return base_cov * scale


def simulate_joint(
    central: dict[str, float],
    cov_matrix: pd.DataFrame,
    n_draws: int,
    distribution: str,
    random_state: int,
    residuals_for_bootstrap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Simule n_draws trajectoires jointes pour les cibles de `central`
    (dict cible -> valeur centrale), en ajoutant un vecteur de chocs corres
    -pondant a la distribution demandee."""
    targets = list(central.keys())
    rng = np.random.default_rng(random_state)
    mean_vec = np.zeros(len(targets))

    if distribution == "gaussienne":
        shocks = rng.multivariate_normal(mean_vec, cov_matrix.loc[targets, targets].to_numpy(), size=n_draws)
    elif distribution == "bootstrap_empirique":
        if residuals_for_bootstrap is None or residuals_for_bootstrap.dropna().empty:
            raise ValueError("bootstrap_empirique requiert des residus historiques non vides.")
        clean = residuals_for_bootstrap[targets].dropna()
        idx = rng.integers(0, len(clean), size=n_draws)
        shocks = clean.to_numpy()[idx]
    else:
        raise ValueError(f"Distribution inconnue : {distribution}")

    central_vec = np.array([central[t] for t in targets])
    draws = central_vec[None, :] + shocks
    return pd.DataFrame(draws, columns=targets)


def compute_bands(draws: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    rows = []
    for target in draws.columns:
        row = {"cible": target, "mediane": float(draws[target].median()), "moyenne": float(draws[target].mean())}
        for level in levels:
            lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
            row[f"borne_basse_{int(level*100)}"] = float(draws[target].quantile(lo_q))
            row[f"borne_haute_{int(level*100)}"] = float(draws[target].quantile(hi_q))
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_check(
    central: dict[str, float], cov_matrix: pd.DataFrame, draw_counts: list[int], distribution: str, random_state: int,
    residuals_for_bootstrap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Verifie la stabilite des quantiles 2.5%/50%/97.5% en augmentant le
    nombre de tirages (consigne explicite de controle de convergence)."""
    rows = []
    for n in draw_counts:
        draws = simulate_joint(central, cov_matrix, n, distribution, random_state, residuals_for_bootstrap)
        for target in draws.columns:
            rows.append({
                "n_tirages": n, "cible": target,
                "q2_5": float(draws[target].quantile(0.025)),
                "q50": float(draws[target].quantile(0.5)),
                "q97_5": float(draws[target].quantile(0.975)),
            })
    out = pd.DataFrame(rows)
    pivot = out.pivot_table(index="cible", columns="n_tirages", values="q97_5")
    max_n = max(draw_counts)
    ref = pivot[max_n]
    rel_change = pivot.subtract(ref, axis=0).div(ref.abs(), axis=0).abs().drop(columns=[max_n])
    out.attrs["ecart_relatif_max_vs_n_max"] = float(rel_change.max().max())
    return out
