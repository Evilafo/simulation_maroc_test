"""Modeles de reference obligatoires (cadrage Ch.6.5.2) : marche aleatoire
(avec/sans derive) et AR(p) univarie avec p choisi par critere
d'information. Prevision iteree (naturelle pour ces modeles), a l'horizon h
depuis l'origine, en n'utilisant que l'historique connu a l'origine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def random_walk_forecast(history: pd.Series, horizon: int) -> float:
    """y_{O+h} = y_O (derniere valeur observee)."""
    return float(history.dropna().iloc[-1])


def random_walk_drift_forecast(history: pd.Series, horizon: int) -> float:
    """y_{O+h} = y_O + h * derive moyenne (moyenne des differences
    premieres sur l'historique disponible a l'origine)."""
    h = history.dropna()
    last = h.iloc[-1]
    drift = h.diff().mean()
    return float(last + horizon * drift)


def select_ar_order(history: pd.Series, max_p: int) -> int:
    h = history.dropna()
    n = len(h)
    max_feasible = max(1, min(max_p, n // 3))
    best_p, best_aic = 1, np.inf
    for p in range(1, max_feasible + 1):
        try:
            fitted = AutoReg(h.to_numpy(), lags=p, old_names=False).fit()
            if fitted.aic < best_aic:
                best_aic, best_p = fitted.aic, p
        except Exception:
            continue
    return best_p


def ar_p_forecast(history: pd.Series, horizon: int, max_p: int) -> dict:
    """Ajuste un AR(p) (p choisi par AIC, borne par max_p) sur l'historique
    disponible a l'origine, puis produit une prevision iteree a h pas."""
    h = history.dropna()
    p = select_ar_order(h, max_p)
    model = AutoReg(h.to_numpy(), lags=p, old_names=False).fit()
    fc = model.predict(start=len(h), end=len(h) + horizon - 1)
    return {"p": p, "forecast": float(fc[-1]), "aic": model.aic}


def walk_forward_baselines(
    target_series: pd.Series,
    reference_forecasts: pd.DataFrame,
    horizon: int,
    max_p: int,
) -> dict[str, pd.DataFrame]:
    """Reproduit, pour chaque (origin, target_year, block) deja evalue par
    un modele candidat (`reference_forecasts`), les trois previsions de
    reference RW/RW+derive/AR(p) -- alignees pour permettre une comparaison
    directe et un test de Diebold-Mariano apparie. N'utilise, a chaque
    origine, que l'historique de la cible strictement anterieur ou egal a
    cette origine."""
    out = {"random_walk": [], "random_walk_drift": [], "ar_p": []}
    for _, row in reference_forecasts.iterrows():
        origin, target_year, block = int(row["origin"]), int(row["target_year"]), row["block"]
        history = target_series.loc[target_series.index <= origin]
        if history.dropna().shape[0] < 3:
            continue
        y_true = target_series.get(target_year, np.nan)

        rw = random_walk_forecast(history, horizon)
        rwd = random_walk_drift_forecast(history, horizon)
        arp = ar_p_forecast(history, horizon, max_p)

        common = {"origin": origin, "target_year": target_year, "y_true": y_true, "block": block}
        out["random_walk"].append({**common, "y_pred": rw})
        out["random_walk_drift"].append({**common, "y_pred": rwd})
        out["ar_p"].append({**common, "y_pred": arp["forecast"], "p_retenu": arp["p"]})

    return {k: pd.DataFrame(v) for k, v in out.items()}
