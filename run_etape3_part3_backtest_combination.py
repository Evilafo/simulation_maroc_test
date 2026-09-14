"""Etape 3, partie 3 : backtest de la reference statistique de benchlearning
(Eq. 3.14) contre le modele national (etape 2) et les baselines, pour
h=1-5, avec reestimation des poids a chaque origine sur les seules
informations accessibles.

Principe respecte explicitement : les valeurs futures REALISEES des
donneurs ne sont jamais utilisees pour construire X_bench_comb(O+h) dans le
backtest -- seules leurs PROJECTIONS tendancielles (memes contraintes de
non-fuite que le modele national marocain de l'etape 2 : caracteristiques
et hyperparametres geles sur le bloc interne, coefficients reajustes par
origine) entrent dans la combinaison. Cette convention de gel reprend
exactement celle deja validee pour le modele national marocain en etape 2
(cf. docs/ETAPE2_PROTOCOLE_ET_CARACTERISTIQUES.md), appliquee ici aussi aux
donneurs pour la premiere fois.

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape3_part3_backtest_combination.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import numpy as np
import pandas as pd

from src import io_utils
from src.etape2 import data_extended as dext
from src.etape2 import evaluation as ev
from src.etape2 import features as feat
from src.etape2 import models as mdl
from src.etape3 import weights as wt

PROJECT_ROOT = Path(__file__).resolve().parent


def _prefix(colname: str) -> str:
    for marker in ["_lag", "_diff1", "_ma", "_std", "_logdiff1", "_log"]:
        idx = colname.find(marker)
        if idx > 0:
            return colname[:idx]
    return colname


def main() -> dict:
    cfg1 = io_utils.load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    cfg3 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape3.yaml")
    logger = io_utils.setup_logging({"logging": cfg3["logging"] | {"log_file": cfg3["paths"]["log_file"]}})
    logger.info("=== Etape 3, partie 3 : backtest de la combinaison benchlearning ===")

    tables_dir = PROJECT_ROOT / cfg3["paths"]["outputs_tables_dir"]

    extended_df = dext.load_extended_work_base(cfg1, cfg2, logger)
    focus = cfg3["scope"]["focus_country"]
    donors_initial = cfg3["scope"]["donors_initial"]
    all_countries = [focus] + donors_initial
    year_start, year_end = cfg3["scope"]["year_start"], cfg3["scope"]["year_end"]
    max_lags = cfg2["features"]["max_lags"]
    ma_windows = cfg2["features"]["moving_stat_windows"]
    redundancy_threshold = cfg2["selection"]["redundancy_corr_threshold"]
    alphas = cfg2["selection"]["elasticnet_alphas"]
    l1_ratios = cfg2["selection"]["elasticnet_l1_ratios"]
    internal_train_end = cfg2["temporal_protocol"]["internal_train_end"]
    final_start = cfg2["temporal_protocol"]["final_start"]
    final_end = cfg2["temporal_protocol"]["final_end"]
    horizons = cfg3["scope"]["horizons_tested_backtest"]
    seed = cfg2["project"]["random_seed"]
    l2_reg = cfg3["weights"]["statistical"]["l2_regularization"]

    all_mech_columns = sorted({c for spec in cfg2["target_mechanisms"].values() for c in spec["disponibles"]})
    feature_frames, target_series = {}, {}
    for c in all_countries:
        dfc = extended_df[extended_df["country_iso3"] == c]
        ff = feat.build_feature_frame(dfc, all_mech_columns, max_lags, ma_windows, year_start, year_end)
        feature_frames[c] = feat.drop_all_nan_columns(ff)
        base = feat.reindex_annual(dfc, year_start, year_end)
        target_series[c] = {t: base[t] for t in cfg2["targets"] if t in base.columns}

    columns_common = set(feature_frames[all_countries[0]].columns)
    for c in all_countries[1:]:
        columns_common &= set(feature_frames[c].columns)

    min_origin = year_start + max(max_lags, cfg2["temporal_protocol"]["min_train_years"] - 1)

    # ------------------------------------------------------------------
    # 1. Modeles de tendance des donneurs, walk-forward (memes garanties de
    #    non-fuite et de gel que le modele national de l'etape 2, reutilise
    #    tel quel via src/etape2/models.py -- pas de nouvelle logique).
    # ------------------------------------------------------------------
    donor_walkforward: dict[tuple[str, str, int], pd.DataFrame] = {}
    for target in cfg2["targets"]:
        mech = cfg2["target_mechanisms"][target]["disponibles"]
        candidate_columns = sorted(c for c in columns_common if _prefix(c) in mech)
        for d in donors_initial:
            for h in horizons:
                res = mdl.run_elasticnet_direct_h(
                    feature_frames[d], target_series[d][target], target, h, candidate_columns,
                    internal_train_end, final_start, final_end, min_origin, year_end - h,
                    redundancy_threshold, alphas, l1_ratios, seed,
                )
                if res is not None:
                    donor_walkforward[(d, target, h)] = res.forecasts.set_index("origin")
        logger.info("Modeles de tendance donneurs geles/executes pour la cible %s (%d donneurs x %d horizons)", target, len(donors_initial), len(horizons))

    # ------------------------------------------------------------------
    # 2. Combinaison ponderee par origine, poids reestimes sur l'historique
    #    accessible a chaque origine (jamais au-dela).
    # ------------------------------------------------------------------
    bench_rows = []
    weight_history = []
    for target in cfg2["targets"]:
        for h in horizons:
            origins_available = None
            for d in donors_initial:
                key = (d, target, h)
                if key not in donor_walkforward:
                    origins_available = set()
                    break
                s = set(donor_walkforward[key].index)
                origins_available = s if origins_available is None else (origins_available & s)
            if not origins_available:
                continue

            for origin in sorted(origins_available):
                w_res = wt.estimate_common_weights(extended_df, focus, donors_initial, list(cfg2["targets"].keys()), year_start, origin, l2_reg)
                w = w_res["weights"]
                if any(pd.isna(v) for v in w.values()):
                    continue
                weight_history.append({"cible": target, "horizon": h, "origin": origin, **w})

                donor_vals = {d: donor_walkforward[(d, target, h)].loc[origin, "y_pred"] for d in donors_initial}
                x_bench = sum(w[d] * donor_vals[d] for d in donors_initial)
                target_year = origin + h
                y_true = target_series[focus][target].get(target_year, np.nan)
                bench_rows.append({
                    "cible": target, "horizon": h, "origin": origin, "target_year": target_year,
                    "y_true": y_true, "y_pred": x_bench,
                    "block": "interne" if target_year <= internal_train_end else "finale",
                    "modele": "reference_benchlearning_statistique",
                })

    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(tables_dir / "e3_20_backtest_reference_benchlearning.csv", index=False)
    pd.DataFrame(weight_history).to_csv(tables_dir / "e3_22_historique_poids_reestimes.csv", index=False)
    logger.info("Backtest de la reference benchlearning : %d previsions calculees", len(bench_df))

    # ------------------------------------------------------------------
    # 3. Comparaison avec le modele national (etape 2) et les baselines
    # ------------------------------------------------------------------
    e2_forecasts = pd.read_csv(PROJECT_ROOT / cfg2["paths"]["outputs_tables_dir"] / "e2_10_previsions_walkforward.csv")

    eval_rows = []
    for target in cfg2["targets"]:
        family = cfg3["targets"][target]["family"]
        for h in horizons:
            bench_th = bench_df[(bench_df["cible"] == target) & (bench_df["horizon"] == h)]
            rw_th = e2_forecasts[(e2_forecasts["cible"] == target) & (e2_forecasts["horizon"] == h) & (e2_forecasts["modele"] == "baseline_random_walk")]
            national_th = e2_forecasts[(e2_forecasts["cible"] == target) & (e2_forecasts["horizon"] == h) & (e2_forecasts["modele"] == "elasticnet")]

            for block in ["interne", "finale"]:
                bench_block = bench_th[bench_th["block"] == block]
                rw_block = rw_th[rw_th["block"] == block]
                nat_block = national_th[national_th["block"] == block]
                if bench_block.empty or rw_block.empty:
                    continue
                res_vs_rw = ev.evaluate_forecasts(bench_block, rw_block, family, cfg3["evaluation"]["gain_threshold_variation_pct"], cfg3["evaluation"]["min_forecasts_for_conclusion"], cfg3["evaluation"]["dm_test_min_forecasts"], h)
                eval_rows.append({"cible": target, "horizon": h, "bloc": block, "comparaison": "benchlearning_vs_random_walk", **{k: v for k, v in res_vs_rw.items() if k != "diebold_mariano"}})
                if not nat_block.empty:
                    res_vs_nat = ev.evaluate_forecasts(bench_block, nat_block, family, cfg3["evaluation"]["gain_threshold_variation_pct"], cfg3["evaluation"]["min_forecasts_for_conclusion"], cfg3["evaluation"]["dm_test_min_forecasts"], h)
                    eval_rows.append({"cible": target, "horizon": h, "bloc": block, "comparaison": "benchlearning_vs_modele_national", **{k: v for k, v in res_vs_nat.items() if k != "diebold_mariano"}})

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(tables_dir / "e3_21_evaluation_benchlearning.csv", index=False)

    summary = {"n_previsions_benchlearning": len(bench_df), "n_evaluations": len(eval_df)}
    with (tables_dir / "e3_00_summary_part3.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Resume partie 3 : %s", summary)
    return summary


if __name__ == "__main__":
    main()
