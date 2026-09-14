"""Etape 3, partie 2 : projections tendancielles 2025-2034 (Maroc + 8
donneurs, 7 cibles, horizons 1-10). Reutilise la specification gelee et
validee de l'etape 2 pour le Maroc aux horizons 1-5 ; nouveau reglage
(meme architecture) pour le Maroc aux horizons 6-10 (extrapolation
signalee) et pour tous les donneurs (jamais modelises en etape 2).

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape3_part2_trend_projections.py
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
from src.etape2 import features as feat
from src.etape3 import trend_projection as tp

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> dict:
    cfg1 = io_utils.load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    cfg3 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape3.yaml")
    logger = io_utils.setup_logging({"logging": cfg3["logging"] | {"log_file": cfg3["paths"]["log_file"]}})
    logger.info("=== Etape 3, partie 2 : projections tendancielles 2025-2034 ===")

    tables_dir = PROJECT_ROOT / cfg3["paths"]["outputs_tables_dir"]
    models_dir = PROJECT_ROOT / cfg3["paths"]["outputs_models_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    extended_df = dext.load_extended_work_base(cfg1, cfg2, logger)

    focus = cfg3["scope"]["focus_country"]
    donors_all = cfg3["scope"]["donors_initial"] + cfg3["scope"]["donors_alternative"]
    all_countries = [focus] + donors_all
    year_start, year_end = cfg3["scope"]["year_start"], cfg3["scope"]["year_end"]
    max_lags = cfg2["features"]["max_lags"]
    ma_windows = cfg2["features"]["moving_stat_windows"]
    redundancy_threshold = cfg2["selection"]["redundancy_corr_threshold"]
    alphas = cfg2["selection"]["elasticnet_alphas"]
    l1_ratios = cfg2["selection"]["elasticnet_l1_ratios"]
    horizons = list(range(1, cfg3["scope"]["projection_horizon_max"] + 1))
    horizons_tested = set(cfg3["scope"]["horizons_tested_backtest"])
    seed = cfg2["project"]["random_seed"]

    all_mech_columns = sorted({c for spec in cfg2["target_mechanisms"].values() for c in spec["disponibles"]})

    feature_frames, target_series = {}, {}
    for c in all_countries:
        dfc = extended_df[extended_df["country_iso3"] == c]
        ff = feat.build_feature_frame(dfc, all_mech_columns, max_lags, ma_windows, year_start, year_end)
        feature_frames[c] = feat.drop_all_nan_columns(ff)
        base = feat.reindex_annual(dfc, year_start, year_end)
        target_series[c] = {t: base[t] for t in cfg2["targets"] if t in base.columns}

    def _prefix(colname: str, mechs: list[str]) -> str:
        for marker in ["_lag", "_diff1", "_ma", "_std", "_logdiff1", "_log"]:
            idx = colname.find(marker)
            if idx > 0:
                return colname[:idx]
        return colname

    columns_common = set(feature_frames[all_countries[0]].columns)
    for c in all_countries[1:]:
        columns_common &= set(feature_frames[c].columns)

    min_origin = year_start + max(max_lags, cfg2["temporal_protocol"]["min_train_years"] - 1)

    all_rows = []
    n_frozen_used, n_new_tuned = 0, 0
    for target in cfg2["targets"]:
        mech = cfg2["target_mechanisms"][target]["disponibles"]
        candidate_columns = sorted(c for c in columns_common if _prefix(c, mech) in mech)

        for country in all_countries:
            frozen_specs = None
            if country == focus:
                frozen_specs = {}
                for h in horizons:
                    if h in horizons_tested:
                        spec = tp.load_frozen_mar_spec(models_dir, target, h)
                        if spec is not None:
                            frozen_specs[h] = spec

            proj = tp.project_country_all_horizons(
                feature_frames[country], target_series[country][target], country, target, horizons,
                candidate_columns, year_end, min_origin, redundancy_threshold, alphas, l1_ratios,
                frozen_specs_by_horizon=frozen_specs, random_state=seed,
            )
            if not proj.empty:
                proj["horizon_teste_backtest"] = proj["horizon"].isin(horizons_tested)
                all_rows.append(proj)
                n_frozen_used += (proj["specification"] == "etape2_fige").sum()
                n_new_tuned += (proj["specification"] == "etape3_nouveau").sum()

        logger.info("Cible %s : projections calculees pour %d pays", target, len(all_countries))

    projections = pd.concat(all_rows, axis=0, ignore_index=True)
    projections.to_csv(tables_dir / "e3_10_projections_tendancielles_2025_2034.csv", index=False)

    coverage = projections.groupby(["cible", "pays"])["horizon"].nunique().reset_index(name="n_horizons_disponibles")
    expected = len(horizons)
    missing = coverage[coverage["n_horizons_disponibles"] < expected]
    if not missing.empty:
        logger.warning("Couverture incomplete (horizons manquants) :\n%s", missing.to_string())
    coverage.to_csv(tables_dir / "e3_11_couverture_projections.csv", index=False)

    summary = {
        "n_lignes_projections": len(projections),
        "n_pays": len(all_countries),
        "n_cibles": len(cfg2["targets"]),
        "n_horizons": len(horizons),
        "n_specs_etape2_figees_reutilisees": int(n_frozen_used),
        "n_specs_nouvelles_etape3": int(n_new_tuned),
        "annees_projetees": [year_end + h for h in horizons],
    }
    with (tables_dir / "e3_00_summary_part2.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Resume partie 2 : %s", summary)
    return summary


if __name__ == "__main__":
    main()
