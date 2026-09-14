"""Implementation directe des equations 3.14-3.15 du cadrage (Ch.3.5),
sans dynamique de substitution (pas de lissage/inertie -- 3e niveau
explicitement hors perimetre, cf. configs/config_etape3.yaml::scenarios.
dynamic_adjustment_enabled=false).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def benchmark_combine(donor_projections: pd.DataFrame, weights: dict[str, float], value_col: str = "valeur_projetee") -> pd.DataFrame:
    """Eq. 3.14 : X_bench_comb(t+h) = somme_b omega_b * X_b(t+h).
    `donor_projections` : lignes (pays, cible, horizon, annee_projetee, valeur_projetee).
    Verifie omega_b >= 0 et somme(omega_b) = 1 (a 1e-6 pres) avant de combiner."""
    w = pd.Series(weights)
    if (w < -1e-9).any():
        raise ValueError(f"Poids negatif interdit (Eq. 3.14, omega_b>=0) : {weights}")
    if abs(w.sum() - 1.0) > 1e-6:
        raise ValueError(f"Somme des poids doit valoir 1 (Eq. 3.14) : somme={w.sum()}")

    df = donor_projections[donor_projections["pays"].isin(w.index)].copy()
    df["poids"] = df["pays"].map(w)
    df["contribution"] = df["poids"] * df[value_col]

    grouped = (
        df.groupby(["cible", "horizon", "annee_projetee"])
        .agg(X_bench_comb=("contribution", "sum"), n_donneurs=("pays", "nunique"), poids_somme=("poids", "sum"))
        .reset_index()
    )
    return grouped


def scenario_trajectory(
    x_tend_mar: pd.DataFrame,
    x_bench_comb: pd.DataFrame,
    alpha: float,
    scenario_name: str,
    tend_col: str = "valeur_projetee",
    bench_col: str = "X_bench_comb",
) -> pd.DataFrame:
    """Eq. 3.15 : X_s(t+h) = (1-alpha)*X_tend_MAR(t+h) + alpha*X_bench_comb(t+h)."""
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha_s doit etre dans [0,1] (Eq. 3.15) : alpha={alpha}")

    merged = x_tend_mar.merge(x_bench_comb, on=["cible", "horizon", "annee_projetee"], how="inner", suffixes=("_tend", "_bench"))
    merged["alpha"] = alpha
    merged["scenario"] = scenario_name
    merged["X_scenario"] = (1 - alpha) * merged[tend_col] + alpha * merged[bench_col]
    return merged[["cible", "horizon", "annee_projetee", tend_col, bench_col, "alpha", "scenario", "X_scenario"]].rename(
        columns={tend_col: "X_tend_MAR", bench_col: "X_bench_comb"}
    )


def sanity_checks(x_tend_mar: pd.DataFrame, x_bench_comb: pd.DataFrame, single_donor_weights: dict[str, float], single_donor_projection: pd.DataFrame) -> pd.DataFrame:
    """Verifications explicitement demandees : alpha=0 -> X_tend_MAR exact ;
    alpha=1 -> X_bench_comb exact ; portefeuille a un seul donneur -> le
    benchmark combine reproduit exactement la trajectoire de ce donneur."""
    rows = []

    s0 = scenario_trajectory(x_tend_mar, x_bench_comb, 0.0, "test_alpha_0")
    ok0 = bool(np.allclose(s0["X_scenario"], s0["X_tend_MAR"], equal_nan=True))
    rows.append({"test": "alpha=0 => X_s = X_tend_MAR", "reussi": ok0, "ecart_max": float((s0["X_scenario"] - s0["X_tend_MAR"]).abs().max())})

    s1 = scenario_trajectory(x_tend_mar, x_bench_comb, 1.0, "test_alpha_1")
    ok1 = bool(np.allclose(s1["X_scenario"], s1["X_bench_comb"], equal_nan=True))
    rows.append({"test": "alpha=1 => X_s = X_bench_comb", "reussi": ok1, "ecart_max": float((s1["X_scenario"] - s1["X_bench_comb"]).abs().max())})

    bench_single = benchmark_combine(single_donor_projection, single_donor_weights)
    single_donor = list(single_donor_weights.keys())[0]
    ref = single_donor_projection[single_donor_projection["pays"] == single_donor]
    merged = bench_single.merge(ref, on=["cible", "horizon", "annee_projetee"])
    ok_single = bool(np.allclose(merged["X_bench_comb"], merged["valeur_projetee"], equal_nan=True))
    rows.append({"test": f"portefeuille a un seul donneur ({single_donor}) => benchmark = sa trajectoire", "reussi": ok_single, "ecart_max": float((merged["X_bench_comb"] - merged["valeur_projetee"]).abs().max()) if len(merged) else np.nan})

    try:
        benchmark_combine(single_donor_projection, {single_donor: 0.6})
        rows.append({"test": "poids ne sommant pas a 1 => rejet", "reussi": False, "ecart_max": np.nan})
    except ValueError:
        rows.append({"test": "poids ne sommant pas a 1 => rejet", "reussi": True, "ecart_max": 0.0})

    try:
        benchmark_combine(single_donor_projection, {single_donor: 1.2, "_dummy": -0.2})
        rows.append({"test": "poids negatif => rejet", "reussi": False, "ecart_max": np.nan})
    except ValueError:
        rows.append({"test": "poids negatif => rejet", "reussi": True, "ecart_max": 0.0})

    return pd.DataFrame(rows)
