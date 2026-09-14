"""Etape 3, partie 1 : validation de la configuration, grille d'eligibilite
des donneurs (S_POC), optimisation de portefeuille, poids statistiques
(communs + par cible) et scenarios politiques.

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape3_part1_donors_weights.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import pandas as pd

from src import io_utils
from src.etape2 import data_extended as dext
from src.etape3 import config_check, donors, weights

PROJECT_ROOT = Path(__file__).resolve().parent


def save(df: pd.DataFrame, name: str, tables_dir: Path) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False)


def main() -> dict:
    cfg1 = io_utils.load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    cfg3 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape3.yaml")
    logger = io_utils.setup_logging({"logging": cfg3["logging"] | {"log_file": cfg3["paths"]["log_file"]}})
    logger.info("=== Etape 3, partie 1 : donneurs, poids ===")

    checks = config_check.validate_config_etape3(cfg3, cfg2)
    for c in checks:
        logger.info("Validation config OK : %s", c)

    tables_dir = PROJECT_ROOT / cfg3["paths"]["outputs_tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    extended_df = dext.load_extended_work_base(cfg1, cfg2, logger)

    focus = cfg3["scope"]["focus_country"]
    donors_all = cfg3["scope"]["donors_initial"] + cfg3["scope"]["donors_alternative"]
    year_start, year_end = cfg3["scope"]["year_start"], cfg3["scope"]["year_end"]

    # ------------------------------------------------------------------
    # 1. Score S_POC (exploratoire)
    # ------------------------------------------------------------------
    score_table = donors.compute_score_poc(extended_df, focus, donors_all, year_start, year_end)
    save(score_table, "e3_01_score_poc", tables_dir)
    logger.info(
        "S_POC calcule pour %d donneurs. Composantes manquantes (non calculables) : %s. Seuil 0.55 applicable a S_POC : %s",
        len(score_table), score_table.attrs["composantes_manquantes"], score_table.attrs["seuil_0_55_applicable"],
    )

    # ------------------------------------------------------------------
    # 2. Redondance et optimisation de portefeuille (Eq. 3.12-3.13)
    # ------------------------------------------------------------------
    proxy_cols = ["VAAG", "VAIN", "VASE", "OUVERTURE_COM", "CROISSANCE_PIB", "FBCF"]
    rho = donors.redundancy_matrix(extended_df, donors_all, year_start, year_end, proxy_cols)
    save(rho.reset_index().rename(columns={"index": "pays"}), "e3_02_matrice_redondance", tables_dir)

    m = cfg3["donor_eligibility"]["portfolio_size_m"]
    lam = cfg3["donor_eligibility"]["diversity_lambda"]
    portfolio_opt = donors.optimize_portfolio(score_table, rho, m, lam)
    save(portfolio_opt["tous"], "e3_03_portefeuilles_compares", tables_dir)

    portfolio_initial = tuple(sorted(cfg3["scope"]["donors_initial"]))
    initial_row = portfolio_opt["tous"][portfolio_opt["tous"]["portefeuille"].apply(lambda p: tuple(sorted(p)) == portfolio_initial)]
    logger.info("Portefeuille initial cadrage %s : %s", portfolio_initial, initial_row.to_dict("records"))
    logger.info("Meilleur portefeuille (taille %d, lambda=%.2f) selon Eq.3.12 : %s", m, lam, portfolio_opt["meilleur"])

    # ------------------------------------------------------------------
    # 3. Poids statistiques : portefeuille initial, alternatif (8 pays), et optimal (Eq. 3.12)
    # ------------------------------------------------------------------
    targets = list(cfg3["targets"].keys())
    l2_reg = cfg3["weights"]["statistical"]["l2_regularization"]

    portfolios_to_weight = {
        "initial_cadrage": cfg3["scope"]["donors_initial"],
        "alternatif_8pays": donors_all,
        "optimal_eq312": list(portfolio_opt["meilleur"]["portefeuille"]),
    }

    weight_rows = []
    for pname, plist in portfolios_to_weight.items():
        res = weights.estimate_common_weights(extended_df, focus, plist, targets, year_start, year_end, l2_reg)
        for d, w in res["weights"].items():
            weight_rows.append({"portefeuille": pname, "pays": d, "poids_statistique_commun": w, "n_cibles_utilisees": res["n_targets_used"]})
    common_weights_table = pd.DataFrame(weight_rows)
    save(common_weights_table, "e3_04_poids_statistiques_communs", tables_dir)
    logger.info("Poids statistiques communs calcules pour %d portefeuilles", len(portfolios_to_weight))

    per_target = weights.estimate_per_target_weights(extended_df, focus, cfg3["scope"]["donors_initial"], targets, year_start, year_end, l2_reg)
    save(per_target, "e3_05_poids_statistiques_par_cible", tables_dir)

    coherence = weights.weight_coherence_across_targets(per_target, cfg3["scope"]["donors_initial"])
    save(coherence, "e3_06_coherence_poids_par_cible", tables_dir)
    logger.info("Coherence poids par cible : correlation moyenne = %.3f", coherence["correlation"].mean() if len(coherence) else float("nan"))

    political_rows = []
    for scenario in cfg3["weights"]["political_scenarios"]:
        w = weights.apply_political_scenario(cfg3["scope"]["donors_initial"], scenario["weights"])
        for d, wv in w.items():
            political_rows.append({"scenario": scenario["name"], "pays": d, "poids_politique": wv})
    save(pd.DataFrame(political_rows), "e3_07_poids_politiques", tables_dir)

    summary = {
        "n_donneurs_candidats": len(donors_all),
        "portefeuille_initial": cfg3["scope"]["donors_initial"],
        "meilleur_portefeuille_eq312": list(portfolio_opt["meilleur"]["portefeuille"]),
        "objectif_meilleur_portefeuille": portfolio_opt["meilleur"]["objectif"],
    }
    with (tables_dir / "e3_00_summary_part1.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Resume partie 1 : %s", summary)
    return summary


if __name__ == "__main__":
    main()
