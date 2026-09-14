"""Etape 3, partie 5 : simulation Monte Carlo jointe (cadrage Ch.6.4) pour
les scenarios alpha, avec controle de convergence, sensibilite a la
distribution des chocs, et bandes 68%/95% en unites economiques
originales.

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape3_part5_montecarlo.py [--fast]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src import io_utils
from src.etape3 import montecarlo as mc

PROJECT_ROOT = Path(__file__).resolve().parent


def main(fast_mode: bool = False) -> dict:
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    cfg3 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape3.yaml")
    logger = io_utils.setup_logging({"logging": cfg3["logging"] | {"log_file": cfg3["paths"]["log_file"]}})
    logger.info("=== Etape 3, partie 5 : Monte Carlo (%s) ===", "mode rapide -- NON livrable final" if fast_mode else "mode complet")

    tables_dir = PROJECT_ROOT / cfg3["paths"]["outputs_tables_dir"]
    targets = list(cfg2["targets"].keys())
    seed = cfg3["monte_carlo"]["random_seed"]
    n_draws = cfg3["monte_carlo"]["n_draws_fast"] if fast_mode else cfg3["monte_carlo"]["n_draws_full"]
    shrinkage = cfg3["monte_carlo"]["shrinkage_intensity"]
    levels = cfg3["monte_carlo"]["coverage_levels"]

    e2_forecasts = pd.read_csv(PROJECT_ROOT / cfg2["paths"]["outputs_tables_dir"] / "e2_10_previsions_walkforward.csv")
    scenarios_df = pd.read_csv(tables_dir / "e3_32_trajectoires_scenarisees.csv")

    # ------------------------------------------------------------------
    # 1. Covariance des residus (horizon 1, modele Elastic Net Maroc), avec
    #    shrinkage vers la diagonale (effectif limite : T~30 pour 7 cibles)
    # ------------------------------------------------------------------
    residuals_h1 = mc.historical_residual_matrix(e2_forecasts, targets, horizon=1)
    cov_shrunk, cov_empirique, n_obs_cov = mc.shrinkage_covariance(residuals_h1, shrinkage)
    cov_shrunk.to_csv(tables_dir / "e3_40_covariance_chocs_shrunk.csv")
    cov_empirique.to_csv(tables_dir / "e3_41_covariance_chocs_empirique.csv")
    logger.info("Covariance des chocs (h=1) estimee sur %d observations, shrinkage=%.2f", n_obs_cov, shrinkage)

    # ------------------------------------------------------------------
    # 2. Controle de convergence (avant la livraison finale)
    # ------------------------------------------------------------------
    central_h1 = {t: 0.0 for t in targets}  # verification de convergence des quantiles des CHOCS seuls (centre a 0), independante du scenario
    conv = mc.convergence_check(central_h1, cov_shrunk, cfg3["monte_carlo"]["convergence_check_draws"], "gaussienne", seed, residuals_h1)
    conv.to_csv(tables_dir / "e3_42_controle_convergence.csv", index=False)
    ecart_rel = conv.attrs.get("ecart_relatif_max_vs_n_max", float("nan"))
    logger.info("Controle de convergence : ecart relatif max des quantiles vs n_max = %.4f", ecart_rel)

    # ------------------------------------------------------------------
    # 3. Simulation jointe pour chaque (scenario alpha, horizon), en unites
    #    originales. Covariance etendue par horizon (Eq. randomwalk-like
    #    pour les horizons d'extrapolation 6-10, cf. montecarlo.py).
    # ------------------------------------------------------------------
    all_bands = []
    for (scenario, horizon), grp in scenarios_df.groupby(["scenario", "horizon"]):
        central = dict(zip(grp["cible"], grp["X_scenario"]))
        if len(central) < len(targets):
            continue
        cov_h = mc.extrapolated_covariance(cov_shrunk, base_horizon=1, target_horizon=horizon)

        for dist in cfg3["monte_carlo"]["shock_distributions"]:
            try:
                draws = mc.simulate_joint(central, cov_h, n_draws, dist, seed, residuals_h1)
            except ValueError as e:
                logger.warning("Simulation impossible (%s, h=%d, %s) : %s", scenario, horizon, dist, e)
                continue
            bands = mc.compute_bands(draws, levels)
            bands["scenario"], bands["horizon"], bands["distribution"] = scenario, horizon, dist
            bands["annee_projetee"] = grp["annee_projetee"].iloc[0]
            bands["horizon_teste_backtest"] = bool(grp["horizon_teste_backtest"].iloc[0]) if "horizon_teste_backtest" in grp.columns else horizon <= 5
            bands["n_tirages"] = n_draws
            all_bands.append(bands)

    bands_df = pd.concat(all_bands, axis=0, ignore_index=True)
    bands_df.to_csv(tables_dir / "e3_43_bandes_incertitude.csv", index=False)
    logger.info("Bandes d'incertitude calculees : %d lignes (scenarios x horizons x distributions x cibles)", len(bands_df))

    summary = {
        "mode": "rapide" if fast_mode else "complet",
        "n_draws": n_draws,
        "n_obs_covariance": n_obs_cov,
        "ecart_relatif_convergence": ecart_rel,
        "n_lignes_bandes": len(bands_df),
        "distributions_testees": cfg3["monte_carlo"]["shock_distributions"],
    }
    with (tables_dir / f"e3_00_summary_part5_{'fast' if fast_mode else 'full'}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Resume partie 5 : %s", summary)
    return summary


if __name__ == "__main__":
    main(fast_mode="--fast" in sys.argv)
