"""Etape 3, partie 6 : figures destinees au MEF (francais, lisibles),
donnees sous-jacentes exportees en CSV a cote de chaque figure.

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape3_part6_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd

from src import io_utils

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 9

PROJECT_ROOT = Path(__file__).resolve().parent


def _save(fig, data, fig_path, data_path):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        data.to_csv(data_path, index=False)


def plot_score_poc(score_poc: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = score_poc.sort_values("S_POC", ascending=True)
    ax.barh(ordered["pays"], ordered["S_POC"], color="#4C72B0")
    ax.set_xlabel("Score S_POC (exploratoire, 4 composantes sur 7)")
    ax.set_title("Score d'eligibilite partiel des pays donneurs (S_POC)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, ordered, figures_dir / "e3_score_poc.png", tables_dir / "e3_fig_score_poc_data.csv")


def plot_weights(weights_df: pd.DataFrame, portfolio: str, figures_dir: Path, tables_dir: Path) -> None:
    sub = weights_df[weights_df["portefeuille"] == portfolio].sort_values("poids_statistique_commun", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(sub["pays"], sub["poids_statistique_commun"], color="#55A868")
    ax.set_xlabel("Poids statistique (controle synthetique regularise)")
    ax.set_title(f"Poids statistiques communs -- portefeuille {portfolio}")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, sub, figures_dir / f"e3_poids_{portfolio}.png", tables_dir / f"e3_fig_poids_{portfolio}_data.csv")


def plot_scenario_fan(
    historical: pd.Series, scenarios_df: pd.DataFrame, target: str, unit: str, figures_dir: Path, tables_dir: Path
) -> None:
    sub = scenarios_df[scenarios_df["cible"] == target].sort_values(["scenario", "horizon"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(historical.index, historical.values, color="black", linewidth=1.3, label="Historique (1991-2024)")

    colors = {"tendance_nationale_seule": "#4C72B0", "convergence_faible": "#64B5CD", "convergence_moderee": "#8172B2", "convergence_forte": "#C44E52", "benchmark_pur": "#DD8452"}
    for scenario, grp in sub.groupby("scenario"):
        grp = grp.sort_values("horizon")
        years = grp["annee_projetee"]
        ax.plot(years, grp["X_scenario"], marker="o", markersize=3, label=scenario, color=colors.get(scenario))
        untested = grp[~grp["horizon_teste_backtest"]]
        if not untested.empty:
            ax.axvspan(untested["annee_projetee"].min() - 0.5, untested["annee_projetee"].max() + 0.5, color="grey", alpha=0.08)

    ax.axvline(2024.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"{target} -- historique et scenarios 2025-2034 ({unit})\n(zone grisee = extrapolation non testee en backtest, horizons 6-10)")
    ax.set_ylabel(unit)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save(fig, sub, figures_dir / f"e3_scenarios_{target}.png", tables_dir / f"e3_fig_scenarios_{target}_data.csv")


def plot_uncertainty_bands(bands_df: pd.DataFrame, target: str, scenario: str, unit: str, figures_dir: Path, tables_dir: Path) -> None:
    sub = bands_df[(bands_df["cible"] == target) & (bands_df["scenario"] == scenario) & (bands_df["distribution"] == "gaussienne")].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(sub["annee_projetee"], sub["borne_basse_95"], sub["borne_haute_95"], color="#4C72B0", alpha=0.2, label="Bande 95% (conditionnelle aux hypotheses)")
    ax.fill_between(sub["annee_projetee"], sub["borne_basse_68"], sub["borne_haute_68"], color="#4C72B0", alpha=0.4, label="Bande 68%")
    ax.plot(sub["annee_projetee"], sub["mediane"], color="#C44E52", marker="o", markersize=3, label="Mediane simulee")
    untested = sub[~sub["horizon_teste_backtest"]]
    if not untested.empty:
        ax.axvspan(untested["annee_projetee"].min() - 0.5, untested["annee_projetee"].max() + 0.5, color="grey", alpha=0.08)
    ax.set_title(f"{target} -- scenario {scenario}, bandes Monte Carlo (10000 tirages, {unit})\n(zone grisee = extrapolation non testee, horizons 6-10)")
    ax.set_ylabel(unit)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, sub, figures_dir / f"e3_bandes_{target}_{scenario}.png", tables_dir / f"e3_fig_bandes_{target}_{scenario}_data.csv")


def main() -> None:
    cfg1 = io_utils.load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    cfg3 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape3.yaml")
    logger = io_utils.setup_logging({"logging": cfg3["logging"] | {"log_file": cfg3["paths"]["log_file"]}})
    logger.info("=== Etape 3, partie 6 : figures ===")

    tables_dir = PROJECT_ROOT / cfg3["paths"]["outputs_tables_dir"]
    figures_dir = PROJECT_ROOT / cfg3["paths"]["outputs_figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    from src.etape2 import data_extended as dext
    extended_df = dext.load_extended_work_base(cfg1, cfg2, logger)
    mar = extended_df[extended_df["country_iso3"] == "MAR"].set_index("year").sort_index()

    score_poc = pd.read_csv(tables_dir / "e3_01_score_poc.csv")
    plot_score_poc(score_poc, figures_dir, tables_dir)

    weights_common = pd.read_csv(tables_dir / "e3_04_poids_statistiques_communs.csv")
    plot_weights(weights_common, "initial_cadrage", figures_dir, tables_dir)
    plot_weights(weights_common, "optimal_eq312", figures_dir, tables_dir)

    scenarios_df = pd.read_csv(tables_dir / "e3_32_trajectoires_scenarisees.csv")
    units = {t: spec.get("unit", "") for t, spec in cfg2["targets"].items()}
    for target in ["CROISSANCE_PIB", "DETTE_PUBLIQUE", "INFLATION_CPI"]:
        plot_scenario_fan(mar[target], scenarios_df, target, units.get(target, ""), figures_dir, tables_dir)

    bands_df = pd.read_csv(tables_dir / "e3_43_bandes_incertitude.csv")
    for target in ["CROISSANCE_PIB", "DETTE_PUBLIQUE"]:
        plot_uncertainty_bands(bands_df, target, "convergence_moderee", units.get(target, ""), figures_dir, tables_dir)

    logger.info("Figures generees dans %s", figures_dir)


if __name__ == "__main__":
    main()
