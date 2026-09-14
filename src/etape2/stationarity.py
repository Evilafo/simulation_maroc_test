"""Stationnarite (ADF + KPSS), sur les series MAROC uniquement (jamais un
panel enchaine comme une serie artificielle). Regle de classification
combinee standard (cf. base methodologique 06_SERIES_TEMPORELLES.md) :

- ADF rejette H0 (racine unitaire) ET KPSS ne rejette pas H0 (stationnaire)
  au niveau -> I(0)
- ADF ne rejette pas au niveau MAIS rejette en difference 1, ET KPSS
  rejette au niveau mais ne rejette pas en difference 1 -> I(1)
- Sinon -> "indetermine/contradictoire" (les deux tests ne s'accordent pas,
  ou aucun des deux schemas ci-dessus ne s'applique) : signale, jamais
  force vers I(0) ou I(1) par defaut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def _adf(series: pd.Series, regression: str = "c") -> dict:
    s = series.dropna()
    stat, pval, lags, nobs, crit, _ = adfuller(s, regression=regression, autolag="AIC")
    return {"stat": stat, "pvalue": pval, "lags": lags, "nobs": nobs, "crit_1pct": crit["1%"], "crit_5pct": crit["5%"], "crit_10pct": crit["10%"]}


def _kpss(series: pd.Series, regression: str = "c") -> dict:
    s = series.dropna()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # p-value hors table (bornes) : attendu et documente ci-dessous
        stat, pval, lags, crit = kpss(s, regression=regression, nlags="auto")
    return {"stat": stat, "pvalue": pval, "lags": lags, "crit_10pct": crit["10%"], "crit_5pct": crit["5%"], "crit_1pct": crit["1%"]}


def classify_series(series: pd.Series, name: str, alpha: float = 0.05) -> dict:
    """Classifie une serie annuelle marocaine (ou tout autre pays isole).
    `series` doit etre indexee par annee, sans reechantillonnage panel."""
    level = series.dropna()
    diff1 = series.diff().dropna()

    adf_level = _adf(level)
    kpss_level = _kpss(level)
    adf_diff = _adf(diff1) if len(diff1) > 5 else None
    kpss_diff = _kpss(diff1) if len(diff1) > 5 else None

    adf_rejects_level = adf_level["pvalue"] < alpha
    kpss_rejects_level = kpss_level["pvalue"] < alpha
    adf_rejects_diff = adf_diff["pvalue"] < alpha if adf_diff else None
    kpss_rejects_diff = kpss_diff["pvalue"] < alpha if kpss_diff else None

    if adf_rejects_level and not kpss_rejects_level:
        order = "I(0)"
        confiance = "concordant"
    elif (not adf_rejects_level) and kpss_rejects_level and adf_rejects_diff and not kpss_rejects_diff:
        order = "I(1)"
        confiance = "concordant"
    elif adf_rejects_level and kpss_rejects_level:
        order = "indetermine"
        confiance = "contradictoire (ADF stationnaire, KPSS non-stationnaire au niveau)"
    elif (not adf_rejects_level) and (not kpss_rejects_level):
        order = "indetermine"
        confiance = "contradictoire (ADF non-stationnaire, KPSS stationnaire au niveau -- possible faible puissance)"
    else:
        order = "indetermine"
        confiance = "schema non standard, a examiner manuellement"

    return {
        "variable": name,
        "n_obs": int(level.shape[0]),
        "adf_stat_niveau": adf_level["stat"], "adf_pvalue_niveau": adf_level["pvalue"],
        "kpss_stat_niveau": kpss_level["stat"], "kpss_pvalue_niveau": kpss_level["pvalue"],
        "adf_stat_diff1": adf_diff["stat"] if adf_diff else np.nan,
        "adf_pvalue_diff1": adf_diff["pvalue"] if adf_diff else np.nan,
        "kpss_stat_diff1": kpss_diff["stat"] if kpss_diff else np.nan,
        "kpss_pvalue_diff1": kpss_diff["pvalue"] if kpss_diff else np.nan,
        "ordre_integration": order,
        "confiance": confiance,
        "note_kpss": "p-value KPSS bornee par la table de la statistique (< 0.01 ou > 0.10) : rapportee telle quelle par statsmodels, interpretee au seuil indique",
    }


def classify_all(df_country: pd.DataFrame, columns: list[str], alpha: float = 0.05) -> pd.DataFrame:
    rows = [classify_series(df_country[c], c, alpha=alpha) for c in columns if c in df_country.columns]
    return pd.DataFrame(rows)
