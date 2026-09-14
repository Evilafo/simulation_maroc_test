"""Metriques d'evaluation (cadrage Ch.6.5.3) : RMSE/MAE + seuil de gain de
10% (cibles en variation), RMSE relative + biais moyen (cibles en niveau/
ratio), test de Diebold-Mariano avec correction Harvey-Leybourne-Newbold
(HLN) pour petits echantillons et erreurs previsionnelles serialement
correlees (horizons > 1), couverture empirique des intervalles a 68%/95%.

Consigne explicite respectee : sous le seuil `min_forecasts_for_conclusion`,
le statut est force a "non concluant" plutot que de rapporter une
significativite ou une couverture non fiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE relative = RMSE / moyenne(|y_true|). Definition explicite,
    documentee ici : evite de diviser par une moyenne signee proche de zero
    pour des cibles pouvant changer de signe (soldes budgetaires,
    compte courant)."""
    denom = np.mean(np.abs(np.asarray(y_true)))
    if denom == 0 or np.isnan(denom):
        return np.nan
    return rmse(y_true, y_pred) / denom


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def diebold_mariano_hln(
    errors_1: np.ndarray, errors_2: np.ndarray, horizon: int, loss: str = "squared"
) -> dict:
    """Test DM avec correction petit-echantillon de Harvey, Leybourne et
    Newbold (1997), variance de la moyenne des differentiels de perte
    calculee par Newey-West (retard = horizon-1, pour des erreurs de
    prevision a h pas serialement correlees par construction).
    H0 : les deux modeles ont la meme precision moyenne de prevision.
    Statistique comparee a une loi de Student a (T-1) degres de liberte
    (recommandation HLN pour petits T), pas a une loi normale."""
    e1, e2 = np.asarray(errors_1, dtype=float), np.asarray(errors_2, dtype=float)
    T = len(e1)
    if T < 3:
        return {"statut": "non_calculable", "motif": f"T={T} < 3, DM non calculable"}

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss doit etre 'squared' ou 'absolute'")

    d_mean = d.mean()
    max_lag = max(0, horizon - 1)
    gamma0 = np.var(d, ddof=0)
    nw_var = gamma0
    for lag in range(1, min(max_lag, T - 1) + 1):
        w = 1 - lag / (max_lag + 1)
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if T - lag > 1 else 0.0
        nw_var += 2 * w * cov
    nw_var = max(nw_var, 1e-12)
    dm_stat = d_mean / np.sqrt(nw_var / T)

    hln_factor = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    dm_hln = dm_stat * hln_factor
    pvalue = 2 * (1 - stats.t.cdf(np.abs(dm_hln), df=T - 1))

    return {
        "statut": "calcule", "T": T, "horizon": horizon,
        "dm_stat_brut": float(dm_stat), "dm_stat_hln": float(dm_hln),
        "pvalue_hln_student_t": float(pvalue), "df": T - 1,
        "perte_moyenne_modele1": float(np.mean(e1 ** 2 if loss == "squared" else np.abs(e1))),
        "perte_moyenne_modele2": float(np.mean(e2 ** 2 if loss == "squared" else np.abs(e2))),
    }


def coverage_rate(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def parametric_interval(pred: float, resid_std: float, level: float) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + level / 2)
    return pred - z * resid_std, pred + z * resid_std


def evaluate_forecasts(
    forecasts: pd.DataFrame,
    baseline_forecasts: pd.DataFrame,
    family: str,
    gain_threshold_pct: float,
    min_forecasts_for_conclusion: int,
    dm_test_min_forecasts: int,
    horizon: int,
) -> dict:
    """Compare un jeu de previsions (colonnes origin, target_year, y_true,
    y_pred, block) a une baseline appariee sur (origin, target_year), pour
    un bloc donne (interne ou finale). `family` in {"variation","niveau_ratio"}
    pilote la metrique principale (cadrage Ch.6.5.3)."""
    merged = forecasts.merge(baseline_forecasts, on=["origin", "target_year"], suffixes=("_modele", "_baseline"))
    merged = merged.dropna(subset=["y_true_modele"])
    n = len(merged)

    out = {"n_previsions": n}
    if n < min_forecasts_for_conclusion:
        out["statut"] = "non_concluant"
        out["motif"] = f"n={n} < seuil minimal {min_forecasts_for_conclusion} : effectif insuffisant pour conclure."
        return out

    y_true = merged["y_true_modele"].to_numpy()
    y_pred = merged["y_pred_modele"].to_numpy()
    y_base = merged["y_pred_baseline"].to_numpy()
    err_model = y_pred - y_true
    err_base = y_base - y_true

    out["rmse_modele"] = rmse(y_true, y_pred)
    out["mae_modele"] = mae(y_true, y_pred)
    out["rmse_baseline"] = rmse(y_true, y_base)
    out["mae_baseline"] = mae(y_true, y_base)
    out["biais_moyen_modele"] = mean_bias(y_true, y_pred)

    if family == "variation":
        gain_pct = 100 * (out["rmse_baseline"] - out["rmse_modele"]) / out["rmse_baseline"] if out["rmse_baseline"] else np.nan
        out["gain_rmse_vs_baseline_pct"] = gain_pct
        out["seuil_gain_atteint"] = bool(gain_pct is not None and not np.isnan(gain_pct) and gain_pct >= gain_threshold_pct)
    else:
        out["rmse_relative_modele"] = relative_rmse(y_true, y_pred)
        out["rmse_relative_baseline"] = relative_rmse(y_true, y_base)
        out["derive_systematique"] = bool(abs(out["biais_moyen_modele"]) > 0.5 * out["rmse_modele"]) if out["rmse_modele"] else None

    if n >= dm_test_min_forecasts:
        out["diebold_mariano"] = diebold_mariano_hln(err_model, err_base, horizon=horizon)
    else:
        out["diebold_mariano"] = {"statut": "non_calcule", "motif": f"n={n} < seuil DM {dm_test_min_forecasts} : test non informatif a cet effectif, non calcule pour eviter une fausse precision."}

    out["statut"] = "evalue"
    return out
