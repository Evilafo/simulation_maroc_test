"""Etape 3, partie 4 : combinaison du benchmark (Eq. 3.14), trajectoires
scenarisees (Eq. 3.15) pour 2025-2034, scenarios politiques appairs a la
reference statistique, et verifications de coherence (alpha=0/1, poids,
portefeuille a un seul donneur).

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape3_part4_combination_scenarios.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src import io_utils
from src.etape3 import combination as comb

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> dict:
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    cfg3 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape3.yaml")
    logger = io_utils.setup_logging({"logging": cfg3["logging"] | {"log_file": cfg3["paths"]["log_file"]}})
    logger.info("=== Etape 3, partie 4 : combinaison et scenarios ===")

    tables_dir = PROJECT_ROOT / cfg3["paths"]["outputs_tables_dir"]
    focus = cfg3["scope"]["focus_country"]
    donors_initial = cfg3["scope"]["donors_initial"]

    projections = pd.read_csv(tables_dir / "e3_10_projections_tendancielles_2025_2034.csv")
    x_tend_mar = projections[projections["pays"] == focus][["cible", "horizon", "annee_projetee", "valeur_projetee", "horizon_teste_backtest"]].copy()

    common_weights = pd.read_csv(tables_dir / "e3_04_poids_statistiques_communs.csv")
    w_initial = dict(zip(
        common_weights[common_weights["portefeuille"] == "initial_cadrage"]["pays"],
        common_weights[common_weights["portefeuille"] == "initial_cadrage"]["poids_statistique_commun"],
    ))
    w_alt = dict(zip(
        common_weights[common_weights["portefeuille"] == "alternatif_8pays"]["pays"],
        common_weights[common_weights["portefeuille"] == "alternatif_8pays"]["poids_statistique_commun"],
    ))
    w_opt = dict(zip(
        common_weights[common_weights["portefeuille"] == "optimal_eq312"]["pays"],
        common_weights[common_weights["portefeuille"] == "optimal_eq312"]["poids_statistique_commun"],
    ))

    donor_projections = projections[projections["pays"] != focus][["pays", "cible", "horizon", "annee_projetee", "valeur_projetee"]]

    # ------------------------------------------------------------------
    # 1. Benchmark combine (Eq. 3.14) -- reference statistique + alternatif
    # ------------------------------------------------------------------
    bench_initial = comb.benchmark_combine(donor_projections, w_initial)
    bench_initial["portefeuille"] = "initial_cadrage_statistique"
    bench_alt = comb.benchmark_combine(donor_projections[donor_projections["pays"].isin(w_alt)], w_alt)
    bench_alt["portefeuille"] = "alternatif_8pays_statistique"
    bench_opt = comb.benchmark_combine(donor_projections[donor_projections["pays"].isin(w_opt)], w_opt)
    bench_opt["portefeuille"] = "optimal_eq312_statistique"

    bench_all = pd.concat([bench_initial, bench_alt, bench_opt], axis=0, ignore_index=True)
    bench_all.to_csv(tables_dir / "e3_30_benchmark_combine.csv", index=False)
    logger.info("Benchmark combine calcule pour %d portefeuilles statistiques", 3)

    # Scenarios politiques (mono-pays), toujours appaires a la reference statistique (garde-fou cadrage Ch.3.5.1)
    political_rows = []
    for scenario in cfg3["weights"]["political_scenarios"]:
        w_pol = {k: v for k, v in scenario["weights"].items() if v > 0}
        bp = comb.benchmark_combine(donor_projections[donor_projections["pays"].isin(w_pol)], w_pol)
        bp["portefeuille"] = f"politique_{scenario['name']}"
        political_rows.append(bp)
    bench_political = pd.concat(political_rows, axis=0, ignore_index=True)
    bench_political.to_csv(tables_dir / "e3_31_benchmark_scenarios_politiques.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Trajectoires scenarisees (Eq. 3.15), alpha configurables
    # ------------------------------------------------------------------
    alphas = cfg3["scenarios"]["alphas"]
    labels = cfg3["scenarios"]["alpha_labels"]
    scenario_rows = []
    for alpha in alphas:
        label = labels.get(str(alpha), f"alpha_{alpha}")
        traj = comb.scenario_trajectory(x_tend_mar, bench_initial, alpha, label)
        traj["portefeuille_reference"] = "initial_cadrage_statistique"
        scenario_rows.append(traj)
    scenarios_df = pd.concat(scenario_rows, axis=0, ignore_index=True)
    scenarios_df = scenarios_df.merge(
        x_tend_mar[["cible", "horizon", "horizon_teste_backtest"]].drop_duplicates(), on=["cible", "horizon"], how="left"
    )
    scenarios_df.to_csv(tables_dir / "e3_32_trajectoires_scenarisees.csv", index=False)
    logger.info("Trajectoires scenarisees calculees pour %d alpha", len(alphas))

    # Scenarios politiques appliques a l'Eq. 3.15 (alpha=1 par construction : ce sont des scenarios "tout benchmark" mono-pays)
    political_scenario_rows = []
    for scenario in cfg3["weights"]["political_scenarios"]:
        bp = bench_political[bench_political["portefeuille"] == f"politique_{scenario['name']}"]
        traj = comb.scenario_trajectory(x_tend_mar, bp, 1.0, scenario["name"])
        political_scenario_rows.append(traj)
    political_scenarios_df = pd.concat(political_scenario_rows, axis=0, ignore_index=True)
    political_scenarios_df.to_csv(tables_dir / "e3_33_scenarios_politiques_trajectoires.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Verifications de coherence explicitement demandees
    # ------------------------------------------------------------------
    single_donor_weights = {donors_initial[0]: 1.0}
    checks = comb.sanity_checks(x_tend_mar, bench_initial, single_donor_weights, donor_projections)
    checks.to_csv(tables_dir / "e3_34_verifications_coherence.csv", index=False)
    all_ok = bool(checks["reussi"].all())
    logger.info("Verifications de coherence : %s (%d/%d reussies)", "TOUTES OK" if all_ok else "ECHEC DETECTE", checks["reussi"].sum(), len(checks))
    if not all_ok:
        logger.error("Details des echecs :\n%s", checks[~checks["reussi"]].to_string())

    summary = {
        "n_lignes_benchmark_combine": len(bench_all),
        "n_scenarios_alpha": len(alphas),
        "n_scenarios_politiques": len(cfg3["weights"]["political_scenarios"]),
        "verifications_toutes_reussies": all_ok,
    }
    with (tables_dir / "e3_00_summary_part4.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Resume partie 4 : %s", summary)
    if not all_ok:
        raise RuntimeError("Des verifications de coherence ont echoue -- voir e3_34_verifications_coherence.csv")
    return summary


if __name__ == "__main__":
    main()
