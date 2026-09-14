"""Cointegration (Johansen), admissibilite et estimation VAR/VECM.

Applique strictement la logique de decision de la demande (section 3) :
- systeme I(0) -> VAR en niveaux
- systeme I(1), rang nul -> VAR en differences
- systeme I(1), 0 < rang < K -> VECM candidat (sous reserve des diagnostics)
- melange I(0)/I(1), I(2), resultats ambigus ou rang K -> reexamen, PAS de
  VECM mecanique

Toutes les fonctions operent sur les donnees MAROC uniquement (systemes
economiquement motives et de petite taille, jamais 28-30 variables
ensemble).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen


def degrees_of_freedom(T: int, K: int, p: int, d: int, q: int = 0) -> dict:
    """T = observations disponibles avant perte de retards ; K = variables
    endogenes ; p = retards ; d = deterministes (1 si constante) ; q =
    exogenes. Parametres par equation = K*p + d + q (formule imposee par la
    demande)."""
    params_per_eq = K * p + d + q
    t_eff = T - p
    dof = t_eff - params_per_eq
    return {
        "T": T, "K": K, "p": p, "d": d, "q": q,
        "T_effectif": t_eff,
        "parametres_par_equation": params_per_eq,
        "degres_liberte_par_equation": dof,
        "defendable": dof >= 5,  # marge minimale arbitraire, documentee : au moins 5 dof residuels par equation
    }


def johansen_rank(data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1, alpha: str = "95%") -> dict:
    """Test de Johansen (trace). det_order : -1 sans constante, 0 constante
    restreinte, 1 constante+tendance (convention statsmodels). Retourne le
    rang retenu au seuil `alpha` et le detail complet trace/valeur propre
    maximale pour tracabilite."""
    idx_alpha = {"90%": 0, "95%": 1, "99%": 2}[alpha]
    result = coint_johansen(data.to_numpy(dtype=float), det_order, k_ar_diff)
    K = data.shape[1]
    trace_stats = result.lr1
    trace_crit = result.cvt[:, idx_alpha]
    maxeig_stats = result.lr2
    maxeig_crit = result.cvm[:, idx_alpha]

    rank = 0
    for r in range(K):
        if trace_stats[r] > trace_crit[r]:
            rank = r + 1
        else:
            break

    detail = pd.DataFrame({
        "H0_rang_<=": list(range(K)),
        "trace_stat": trace_stats,
        "trace_crit_95": result.cvt[:, 1],
        "maxeig_stat": maxeig_stats,
        "maxeig_crit_95": result.cvm[:, 1],
        "eigenvalue": result.eig,
    })
    return {"rang_retenu": rank, "detail": detail, "det_order": det_order, "k_ar_diff": k_ar_diff}


def decide_system_type(integration_orders: dict[str, str], rank_info: dict | None, K: int) -> dict:
    """Applique la logique de decision imposee par la demande. `integration_orders`
    mappe variable -> "I(0)"/"I(1)"/"indetermine" (ou autre)."""
    orders = set(integration_orders.values())

    if orders == {"I(0)"}:
        return {"type": "VAR_niveaux", "motif": "Toutes les variables sont I(0) : VAR en representation stationnaire."}

    if orders == {"I(1)"}:
        if rank_info is None:
            return {"type": "reexamen", "motif": "Toutes I(1) mais rang de cointegration non calcule."}
        r = rank_info["rang_retenu"]
        if r == 0:
            return {"type": "VAR_differences", "motif": f"Toutes I(1), rang de Johansen = 0 (aucune relation de cointegration) : VAR sur les series differenciees."}
        if 0 < r < K:
            return {"type": "VECM_candidat", "motif": f"Toutes I(1), rang de Johansen = {r} (0<rang<{K}) : VECM candidat, sous reserve des diagnostics residuels et de stabilite.", "rang": r}
        return {"type": "reexamen", "motif": f"Rang de Johansen = {r} = K (rang plein) : schema non standard pour un systeme I(1), pas de VECM mecanique."}

    return {"type": "reexamen", "motif": f"Ordres d'integration heterogenes ou indetermines ({integration_orders}) : la specification doit etre revue avant toute estimation VAR/VECM."}


def fit_small_var(data: pd.DataFrame, maxlags: int, trend: str = "c", ic: str = "aic") -> dict:
    """Estime un VAR de petite taille avec selection de retard par critere
    d'information (borne par maxlags), puis diagnostics : blancheur des
    residus (Portmanteau), normalite (Jarque-Bera multivarie), stabilite
    (racines du polynome caracteristique), heteroscedasticite (ARCH-LM
    univarie par equation)."""
    model = VAR(data)
    selected = model.select_order(maxlags=maxlags)
    p = getattr(selected, ic)
    p = max(p, 1)
    results = model.fit(maxlags=p, trend=trend)

    whiteness = results.test_whiteness(nlags=min(4, results.nobs // 2) if results.nobs > 8 else 2)
    normality = results.test_normality()
    arch_pvalues = {}
    for col in data.columns:
        resid_col = results.resid[col]
        try:
            _, pval, _, _ = het_arch(resid_col, nlags=min(2, max(1, len(resid_col) // 4)))
            arch_pvalues[col] = pval
        except Exception:
            arch_pvalues[col] = np.nan

    return {
        "p_retenu": p,
        "ic_utilise": ic,
        "trend": trend,
        "results": results,
        "roots": results.roots,
        "is_stable": results.is_stable(),
        "whiteness_stat": whiteness.test_statistic, "whiteness_pvalue": whiteness.pvalue,
        "normality_stat": normality.test_statistic, "normality_pvalue": normality.pvalue,
        "arch_lm_pvalues_par_equation": arch_pvalues,
        "dof": degrees_of_freedom(T=len(data), K=data.shape[1], p=p, d=1 if trend == "c" else 0),
    }


def fit_small_vecm(data: pd.DataFrame, k_ar_diff: int, coint_rank: int, deterministic: str = "co") -> dict:
    """Estime un VECM de petite taille (rang impose par le test de
    Johansen, jamais devine)."""
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)
    results = model.fit()

    whiteness = results.test_whiteness(nlags=min(4, len(data) // 4) if len(data) > 8 else 2)
    normality = results.test_normality()

    return {
        "k_ar_diff": k_ar_diff,
        "coint_rank": coint_rank,
        "deterministic": deterministic,
        "results": results,
        "whiteness_stat": whiteness.test_statistic, "whiteness_pvalue": whiteness.pvalue,
        "normality_stat": normality.test_statistic, "normality_pvalue": normality.pvalue,
        "dof": degrees_of_freedom(T=len(data), K=data.shape[1], p=k_ar_diff + 1, d=1),
    }
