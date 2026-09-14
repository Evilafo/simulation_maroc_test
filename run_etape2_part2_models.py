"""Etape 2, partie 2 : caracteristiques, Elastic Net (Maroc), panel pooled
(avec effets fixes + leave-one-country-out) et evaluation face aux
baselines, pour les 7 cibles x horizons 1..5.

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape2_part2_models.py
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import pandas as pd

from src import io_utils
from src.etape2 import baselines as bl
from src.etape2 import data_extended as dext
from src.etape2 import evaluation as ev
from src.etape2 import features as feat
from src.etape2 import models as mdl

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> dict:
    cfg1 = io_utils.load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    logger = io_utils.setup_logging({"logging": cfg2["logging"] | {"log_file": cfg2["paths"]["log_file"]}})
    logger.info("=== Etape 2, partie 2 : caracteristiques, modeles, evaluation ===")

    tables_dir = PROJECT_ROOT / cfg2["paths"]["outputs_tables_dir"]
    models_dir = PROJECT_ROOT / cfg2["paths"]["outputs_models_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    year_start, year_end = cfg2["scope"]["year_start"], cfg2["scope"]["year_end"]
    countries = cfg2["scope"]["countries"]
    focus = cfg2["scope"]["focus_country"]
    max_lags = cfg2["features"]["max_lags"]
    ma_windows = cfg2["features"]["moving_stat_windows"]
    redundancy_threshold = cfg2["selection"]["redundancy_corr_threshold"]
    alphas = cfg2["selection"]["elasticnet_alphas"]
    l1_ratios = cfg2["selection"]["elasticnet_l1_ratios"]
    internal_train_end = cfg2["temporal_protocol"]["internal_train_end"]
    final_start = cfg2["temporal_protocol"]["final_start"]
    final_end = cfg2["temporal_protocol"]["final_end"]
    min_train_years = cfg2["temporal_protocol"]["min_train_years"]
    horizons = cfg2["temporal_protocol"]["horizons"]
    ar_p_max = cfg2["models"]["ar_p_max"]
    gain_threshold = cfg2["evaluation"]["gain_threshold_variation_pct"]
    min_fc_conclusion = cfg2["evaluation"]["min_forecasts_for_conclusion"]
    dm_min = cfg2["evaluation"]["dm_test_min_forecasts"]

    extended_df = dext.load_extended_work_base(cfg1, cfg2, logger)
    min_origin = year_start + max_lags  # retards disponibles
    min_origin = max(min_origin, year_start + min_train_years - 1)

    all_columns = sorted({c for spec in cfg2["target_mechanisms"].values() for c in spec["disponibles"]})

    feature_frames: dict[str, pd.DataFrame] = {}
    target_series_by_country: dict[str, dict[str, pd.Series]] = {}
    for c in countries:
        df_c = extended_df[extended_df["country_iso3"] == c]
        ff = feat.build_feature_frame(df_c, all_columns, max_lags, ma_windows, year_start, year_end)
        feature_frames[c] = feat.drop_all_nan_columns(ff)
        base = feat.reindex_annual(df_c, year_start, year_end)
        target_series_by_country[c] = {t: base[t] for t in cfg2["targets"] if t in base.columns}

    logger.info("Caracteristiques construites pour %d pays (colonnes de base : %s)", len(countries), all_columns)

    all_forecast_rows = []
    all_eval_rows = []
    model_registry = {}

    for target, tspec in cfg2["targets"].items():
        candidate_prefixes = cfg2["target_mechanisms"][target]["disponibles"]

        def _prefix(colname: str) -> str:
            for suffix_marker in ["_lag", "_diff1", "_ma", "_std", "_logdiff1", "_log"]:
                idx = colname.find(suffix_marker)
                if idx > 0:
                    return colname[:idx]
            return colname

        # Intersection sur les 9 pays : necessaire pour le panel pooled, et
        # gage de comparabilite avec l'Elastic Net Maroc (memes colonnes).
        columns_common = set(feature_frames[countries[0]].columns)
        for c in countries[1:]:
            columns_common &= set(feature_frames[c].columns)
        candidate_columns = sorted(c for c in columns_common if _prefix(c) in candidate_prefixes)

        family = tspec["family"]
        target_series_focus = target_series_by_country[focus][target]

        for h in horizons:
            logger.info("--- Cible %s, horizon %d ---", target, h)

            elastic = mdl.run_elasticnet_direct_h(
                feature_frames[focus], target_series_focus, target, h, candidate_columns,
                internal_train_end, final_start, final_end, min_origin, year_end - h,
                redundancy_threshold, alphas, l1_ratios, cfg2["project"]["random_seed"],
            )
            if elastic is None:
                logger.warning("ElasticNet non calculable pour %s h=%d (donnees insuffisantes)", target, h)
                continue

            baselines_fc = bl.walk_forward_baselines(target_series_focus, elastic.forecasts, h, ar_p_max)

            target_series_per_country_this_target = {c: target_series_by_country[c][target] for c in countries if target in target_series_by_country[c]}

            panel_fe = mdl.run_panel_pooled_direct_h(
                feature_frames, target_series_per_country_this_target, target, h, candidate_columns,
                internal_train_end, final_start, final_end, min_origin, year_end - h,
                redundancy_threshold, alphas, l1_ratios, focus,
                exclude_country_from_training=None, use_country_dummies=True,
                random_state=cfg2["project"]["random_seed"],
            )
            panel_loco = mdl.run_panel_pooled_direct_h(
                feature_frames, target_series_per_country_this_target, target, h, candidate_columns,
                internal_train_end, final_start, final_end, min_origin, year_end - h,
                redundancy_threshold, alphas, l1_ratios, focus,
                exclude_country_from_training=focus, use_country_dummies=False,
                random_state=cfg2["project"]["random_seed"],
            )

            for model_name, res in [("elasticnet", elastic), ("panel_avec_effets_fixes", panel_fe), ("panel_leave_one_country_out", panel_loco)]:
                if res is None:
                    continue
                fc = res.forecasts.copy()
                fc["cible"], fc["horizon"], fc["modele"] = target, h, model_name
                all_forecast_rows.append(fc)

            for base_name, fc in baselines_fc.items():
                if fc.empty:
                    continue
                fc = fc.copy()
                fc["cible"], fc["horizon"], fc["modele"] = target, h, f"baseline_{base_name}"
                all_forecast_rows.append(fc)

            rw_fc = baselines_fc["random_walk"]
            for model_name, res in [("elasticnet", elastic), ("panel_avec_effets_fixes", panel_fe), ("panel_leave_one_country_out", panel_loco)]:
                if res is None or rw_fc.empty:
                    continue
                for block_label, block_name in [("interne", "interne"), ("finale", "finale")]:
                    fc_block = res.forecasts[res.forecasts["block"] == block_name]
                    rw_block = rw_fc[rw_fc["block"] == block_name]
                    if fc_block.empty or rw_block.empty:
                        continue
                    result = ev.evaluate_forecasts(
                        fc_block, rw_block, family, gain_threshold, min_fc_conclusion, dm_min, h,
                    )
                    all_eval_rows.append({
                        "cible": target, "horizon": h, "modele": model_name, "bloc": block_name,
                        "baseline_comparaison": "random_walk", **{k: v for k, v in result.items() if k != "diebold_mariano"},
                        "dm_pvalue_hln": result.get("diebold_mariano", {}).get("pvalue_hln_student_t"),
                        "dm_statut": result.get("diebold_mariano", {}).get("statut"),
                    })

            if elastic is not None:
                key = f"{target}_h{h}"
                model_registry[key] = {
                    "frozen_features": elastic.frozen_features,
                    "alpha": elastic.frozen_alpha, "l1_ratio": elastic.frozen_l1_ratio,
                    "tuning_detail": {k: v for k, v in elastic.tuning_detail.items() if k != "grid_results"},
                    "objets_deployables": {
                        block: {"origin": d["origin"], "target_year": d["target_year"], "features": d["features"]}
                        for block, d in elastic.deployable_models.items()
                    },
                }
                # Objets du modele (pickle, jamais ecrases : un fichier par
                # cible/horizon/bloc) -- necessaires pour rejouer le
                # backtest ou projeter en etape 3 sans reentrainer.
                for block, d in elastic.deployable_models.items():
                    pkl_path = models_dir / f"elasticnet_{target}_h{h}_{block}.pkl"
                    with pkl_path.open("wb") as f:
                        pickle.dump({
                            "target": target, "horizon": h, "block": block,
                            "origin": d["origin"], "target_year": d["target_year"],
                            "features": d["features"], "model": d["model"], "scaler": d["scaler"],
                            "alpha": elastic.frozen_alpha, "l1_ratio": elastic.frozen_l1_ratio,
                        }, f)

    forecasts_table = pd.concat(all_forecast_rows, axis=0, ignore_index=True) if all_forecast_rows else pd.DataFrame()
    forecasts_table.to_csv(tables_dir / "e2_10_previsions_walkforward.csv", index=False)

    eval_table = pd.DataFrame(all_eval_rows)
    eval_table.to_csv(tables_dir / "e2_11_evaluation_vs_baseline.csv", index=False)

    with (models_dir / "e2_model_registry.json").open("w", encoding="utf-8") as f:
        json.dump(model_registry, f, ensure_ascii=False, indent=2, default=str)

    summary = {
        "n_lignes_previsions": len(forecasts_table),
        "n_evaluations": len(eval_table),
        "n_specifications_gelees": len(model_registry),
    }
    with (tables_dir / "e2_00_summary_part2.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Resume partie 2 : %s", summary)
    return summary


if __name__ == "__main__":
    main()
