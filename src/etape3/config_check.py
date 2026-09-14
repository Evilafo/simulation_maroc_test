"""Validation de la configuration Etape 3 au demarrage (consigne explicite).

Distingue les valeurs tracables a une prescription du cadrage (seuils,
equations) des valeurs par defaut proposees par ce projet. Leve une
exception explicite en cas d'incoherence -- jamais un avertissement muet.
"""
from __future__ import annotations

from typing import Any


class ConfigError(ValueError):
    pass


def validate_config_etape3(cfg3: dict[str, Any], cfg2: dict[str, Any]) -> list[str]:
    """Retourne la liste des verifications passees (pour journalisation) ou
    leve ConfigError au premier probleme bloquant."""
    checks_passed = []

    scope = cfg3["scope"]
    if scope["year_start"] != cfg2["scope"]["year_start"] or scope["year_end"] != cfg2["scope"]["year_end"]:
        raise ConfigError("scope.year_start/year_end de l'etape 3 doivent coincider avec l'etape 2 (meme perimetre historique).")
    checks_passed.append("perimetre historique coherent avec l'etape 2")

    if scope["projection_year_start"] != scope["year_end"] + 1:
        raise ConfigError("projection_year_start doit suivre immediatement year_end (pas de trou ni de chevauchement).")
    checks_passed.append("projection_year_start contigu a year_end")

    n_years_proj = scope["projection_year_end"] - scope["projection_year_start"] + 1
    if n_years_proj > scope["projection_horizon_max"]:
        raise ConfigError(f"Fenetre de projection ({n_years_proj} ans) depasse projection_horizon_max ({scope['projection_horizon_max']}).")
    checks_passed.append("horizon de projection <= plafond configure")

    donors = set(scope["donors_initial"]) | set(scope["donors_alternative"])
    if scope["focus_country"] in donors:
        raise ConfigError("focus_country (Maroc) ne doit jamais apparaitre parmi les donneurs.")
    checks_passed.append("Maroc exclu de ses propres donneurs")

    all_horizons = set(scope["horizons_tested_backtest"]) | set(scope["horizons_extrapolation"])
    if all_horizons != set(range(1, scope["projection_horizon_max"] + 1)):
        raise ConfigError("horizons_tested_backtest + horizons_extrapolation doivent couvrir exactement 1..projection_horizon_max sans trou ni recouvrement.")
    checks_passed.append("horizons testes + extrapolation couvrent exactement 1..horizon_max")

    elig = cfg3["donor_eligibility"]
    if elig["seuil_0_55_applicable"]:
        raise ConfigError("Le seuil 0.55 (Eq. 3.11 du cadrage) ne s'applique qu'au score complet S_c a 7 criteres, jamais a S_POC. seuil_0_55_applicable doit rester false tant que S1/S4/S5 ne sont pas calculables.")
    checks_passed.append("seuil 0.55 non applique a S_POC (garde-fou respecte)")

    for scenario in cfg3["weights"]["political_scenarios"]:
        w_sum = sum(scenario["weights"].values())
        if abs(w_sum - 1.0) > 1e-9:
            raise ConfigError(f"Scenario politique '{scenario['name']}' : somme des poids = {w_sum}, doit valoir 1.")
        if any(w < 0 for w in scenario["weights"].values()):
            raise ConfigError(f"Scenario politique '{scenario['name']}' : poids negatif interdit (Eq. 3.14 : omega_b >= 0).")
    checks_passed.append("scenarios politiques : poids >= 0 et somme = 1 (Eq. 3.14)")

    alphas = cfg3["scenarios"]["alphas"]
    if any(a < 0 or a > 1 for a in alphas):
        raise ConfigError("Tous les alpha_s doivent etre dans [0,1] (Eq. 3.15).")
    checks_passed.append("alpha_s dans [0,1] (Eq. 3.15)")

    if cfg3["scenarios"]["dynamic_adjustment_enabled"]:
        raise ConfigError("dynamic_adjustment_enabled=true : le 3e niveau (lissage/inertie) est hors perimetre du cadrage (Ch.3.5) et doit rester desactivable/desactive par defaut. Si active volontairement, documenter explicitement comme extension calibree separement.")
    checks_passed.append("ajustement dynamique (hors cadrage) desactive par defaut")

    mc = cfg3["monte_carlo"]
    if mc["n_draws_full"] < 10000:
        raise ConfigError("monte_carlo.n_draws_full doit etre >= 10000 pour la livraison finale (cadrage Ch.6.4).")
    checks_passed.append("n_draws_full >= 10000 (cadrage Ch.6.4)")

    return checks_passed
