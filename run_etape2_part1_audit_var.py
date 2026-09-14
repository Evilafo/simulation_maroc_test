"""Etape 2, partie 1 : jeu de travail etendu, stationnarite (Maroc),
cointegration et systemes VAR/VECM candidats.

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape2_part1_audit_var.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*", category=UserWarning)

import pandas as pd

from src import io_utils
from src.etape2 import cointegration_var as cv
from src.etape2 import data_extended as dext
from src.etape2 import stationarity as stn

PROJECT_ROOT = Path(__file__).resolve().parent


def save(df: pd.DataFrame, name: str, tables_dir: Path) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False)


def main() -> dict:
    cfg1 = io_utils.load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    logger = io_utils.setup_logging({"logging": cfg2["logging"] | {"log_file": cfg2["paths"]["log_file"]}})
    logger.info("=== Etape 2, partie 1 : donnees, stationnarite, VAR/VECM ===")

    tables_dir = PROJECT_ROOT / cfg2["paths"]["outputs_tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    extended_df = dext.load_extended_work_base(cfg1, cfg2, logger)
    ok = dext.verify_against_etape1_work_base(extended_df, cfg1, logger)
    if not ok:
        raise RuntimeError("Le jeu etendu diverge du fichier de travail de l'etape 1 sur des colonnes partagees.")

    mar = extended_df[extended_df["country_iso3"] == "MAR"].set_index("year").sort_index()

    # ------------------------------------------------------------------
    # Stationnarite (Maroc uniquement)
    # ------------------------------------------------------------------
    all_vars_of_interest = sorted(set(
        list(cfg2["targets"].keys())
        + ["SOPR", "DETO", "RETO", "TCER", "IDEE", "TAAC", "POP_ACTIVE_TOTALE", "EMAG", "EMIN"]
    ))
    stationarity_table = stn.classify_all(mar, all_vars_of_interest)
    save(stationarity_table, "e2_01_stationnarite_maroc", tables_dir)
    logger.info("Stationnarite Maroc : \n%s", stationarity_table[["variable", "ordre_integration", "confiance"]].to_string())

    integration_map = dict(zip(stationarity_table["variable"], stationarity_table["ordre_integration"]))

    # ------------------------------------------------------------------
    # Systemes VAR/VECM candidats
    # ------------------------------------------------------------------
    var_results = []
    for system in cfg2["var_candidate_systems"]:
        name = system["name"]
        variables = system["variables"]
        K = len(variables)
        sys_integration = {v: integration_map.get(v, "inconnu") for v in variables}
        data = mar[variables].dropna()
        T = len(data)

        dof_p1 = cv.degrees_of_freedom(T=T, K=K, p=1, d=1)

        rank_info = None
        if set(sys_integration.values()) == {"I(1)"} and T > 10:
            try:
                rank_info = cv.johansen_rank(data, det_order=0, k_ar_diff=1, alpha="95%")
            except Exception as e:
                rank_info = None
                logger.warning("Johansen a echoue pour %s : %s", name, e)

        decision = cv.decide_system_type(sys_integration, rank_info, K)

        entry = {
            "systeme": name, "variables": ",".join(variables), "justification": system["justification"],
            "ordres_integration": json.dumps(sys_integration, ensure_ascii=False),
            "T_disponible": T, "dof_p1_par_equation": dof_p1["degres_liberte_par_equation"],
            "rang_johansen_95": rank_info["rang_retenu"] if rank_info else None,
            "decision": decision["type"], "motif_decision": decision["motif"],
        }

        estimation_note = "non estime"
        if decision["type"] in ("VAR_niveaux", "VAR_differences") and dof_p1["defendable"]:
            try:
                est_data = data if decision["type"] == "VAR_niveaux" else data.diff().dropna()
                fit = cv.fit_small_var(est_data, maxlags=cfg2["max_lags_var_search"])
                entry.update({
                    "p_retenu": fit["p_retenu"], "stable": fit["is_stable"],
                    "whiteness_pvalue": fit["whiteness_pvalue"], "normality_pvalue": fit["normality_pvalue"],
                    "dof_effectif_par_equation": fit["dof"]["degres_liberte_par_equation"],
                })
                estimation_note = "estime"
            except Exception as e:
                estimation_note = f"echec estimation VAR : {e}"
        elif decision["type"] == "VECM_candidat" and dof_p1["defendable"]:
            try:
                fit = cv.fit_small_vecm(data, k_ar_diff=1, coint_rank=decision["rang"], deterministic="co")
                entry.update({
                    "k_ar_diff": fit["k_ar_diff"], "coint_rank": fit["coint_rank"],
                    "whiteness_pvalue": fit["whiteness_pvalue"], "normality_pvalue": fit["normality_pvalue"],
                    "dof_effectif_par_equation": fit["dof"]["degres_liberte_par_equation"],
                })
                estimation_note = "estime"
            except Exception as e:
                estimation_note = f"echec estimation VECM : {e}"
        else:
            estimation_note = "rejete en amont (dof insuffisant ou reexamen requis) : voir motif_decision"

        entry["statut_estimation"] = estimation_note
        var_results.append(entry)
        logger.info("Systeme %s : decision=%s (%s)", name, decision["type"], estimation_note)

    var_table = pd.DataFrame(var_results)
    save(var_table, "e2_02_systemes_var_vecm", tables_dir)

    summary = {
        "n_variables_stationnarite_testees": len(stationarity_table),
        "n_systemes_var_candidats": len(var_results),
        "n_systemes_estimes": int((var_table["statut_estimation"] == "estime").sum()),
    }
    with (tables_dir / "e2_00_summary_part1.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Resume partie 1 : %s", summary)
    return summary


if __name__ == "__main__":
    main()
