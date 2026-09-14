"""Point d'entree modulaire - Etape 1 (audit, preparation, deux ACP,
typologie des pays) - DATALAB MEF Maroc UC-S1.

Usage (Windows / VS Code / terminal) :
    "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape1.py

Conçu pour etre relance depuis un noyau/processus propre : ne depend
d'aucun etat externe autre que configs/config.yaml et les fichiers sources
qui y sont references.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

# Avertissement Windows/MKL connu et sans consequence pour des jeux de
# donnees de cette taille (n=9 ou n=278) : cf. docs/SUIVI_MODELISATION.md.
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)

from src import audit, clustering, io_utils, pca_profiles, pca_trajectories, plotting

PROJECT_ROOT = Path(__file__).resolve().parent


def save_table(df: pd.DataFrame, name: str, tables_dir: Path, reset_index: bool = False) -> Path:
    if reset_index:
        df = df.reset_index()
    path = tables_dir / f"{name}.csv"
    df.to_csv(path, index=not reset_index and df.index.name is not None)
    return path


def main() -> dict:
    cfg = io_utils.load_config()
    logger = io_utils.setup_logging(cfg)
    logger.info("=== Debut Etape 1 - UC-S1 MEF Maroc ===")
    env = io_utils.log_environment(logger)

    tables_dir = PROJECT_ROOT / cfg["paths"]["outputs_tables_dir"]
    figures_dir = PROJECT_ROOT / cfg["paths"]["outputs_figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    year_start = cfg["scope"]["year_start"]
    year_end = cfg["scope"]["year_end"]
    focus = cfg["scope"]["focus_country"]
    sens_start, sens_end = cfg["scope"]["sensitivity_common_window"]

    # ------------------------------------------------------------------
    # 1. Chargement des donnees et empreintes
    # ------------------------------------------------------------------
    work_df = io_utils.load_work_base(cfg, logger)
    source_df = io_utils.load_source_base(cfg, logger)
    registry = io_utils.load_variable_registry(cfg)

    fingerprints = {
        "work_base_mef": io_utils.sha256_of_file(io_utils.resolve_path(cfg, "work_base_mef")),
        "source_base_B_revised_wide": io_utils.sha256_of_file(io_utils.resolve_path(cfg, "source_base_B_revised_wide")),
    }
    logger.info("Empreintes SHA256 des sources : %s", fingerprints)

    countries_cfg = set(cfg["scope"]["countries"])
    countries_data = set(work_df["country_iso3"].unique())
    if countries_cfg != countries_data:
        logger.warning(
            "Ecart pays config vs donnees : config_only=%s, data_only=%s",
            countries_cfg - countries_data, countries_data - countries_cfg,
        )

    # ------------------------------------------------------------------
    # 2. Audit qualite + dictionnaire
    # ------------------------------------------------------------------
    logger.info("--- Audit qualite des donnees ---")
    year_gaps = audit.check_duplicates_and_year_gaps(work_df, year_start, year_end)
    save_table(year_gaps, "01_doublons_et_annees", tables_dir)

    states = audit.completeness_by_state(work_df)
    save_table(states, "02_completude_par_etat", tables_dir)
    n_complete = int(states["etat_complet"].sum())
    n_incomplete = int((~states["etat_complet"]).sum())
    logger.info("Etats complets: %d / incomplets: %d / total: %d", n_complete, n_incomplete, len(states))

    miss_by_ind = audit.missingness_by_indicator(work_df)
    save_table(miss_by_ind, "03_manquants_par_indicateur", tables_dir)

    common_window = audit.find_common_complete_window(work_df, year_start, year_end)
    logger.info("Fenetre commune complete detectee : %s", common_window)

    quality_checks = audit.constants_and_impossible_values(work_df, registry)
    save_table(quality_checks, "04_verifications_qualite", tables_dir)

    dict_check = audit.cross_check_targets_dictionary(cfg, work_df, registry)
    save_table(dict_check, "05_verification_dictionnaire_cibles", tables_dir)

    registry_vs_base = audit.variables_registry_vs_work_base(work_df, registry)
    save_table(registry_vs_base, "06_registre_vs_fichier_travail", tables_dir)

    identities = audit.exact_linear_identities(work_df)
    save_table(identities, "07_identites_lineaires_exactes", tables_dir)
    logger.info("Identites lineaires exactes detectees : %d", int(identities["exacte"].sum()) if len(identities) else 0)

    # ------------------------------------------------------------------
    # 3. ACP1 - profils moyens par pays
    # ------------------------------------------------------------------
    logger.info("--- ACP1 : profils moyens par pays (%d-%d) ---", year_start, year_end)
    acp1_result, acp1_profiles, acp1_coverage = pca_profiles.run_acp1(work_df, year_start, year_end)
    save_table(acp1_profiles, "10_acp1_profils_moyens", tables_dir, reset_index=True)
    save_table(acp1_coverage, "11_acp1_couverture_annees", tables_dir)
    save_table(acp1_result.eigenvalues, "12_acp1_valeurs_propres", tables_dir)

    acp1_scores = acp1_result.scores.reset_index().rename(columns={"index": "country_iso3"})
    save_table(acp1_scores, "13_acp1_coordonnees_individus", tables_dir)
    save_table(acp1_result.var_correlations.reset_index(), "14_acp1_correlations_variables", tables_dir)
    save_table(acp1_result.var_contributions.reset_index(), "15_acp1_contributions_variables", tables_dir)
    save_table(acp1_result.var_cos2.reset_index(), "16_acp1_cos2_variables", tables_dir)
    save_table(acp1_result.ind_contributions.reset_index().rename(columns={"index": "country_iso3"}), "17_acp1_contributions_individus", tables_dir)
    save_table(acp1_result.ind_cos2.reset_index().rename(columns={"index": "country_iso3"}), "18_acp1_cos2_individus", tables_dir)

    plotting.plot_scree(
        acp1_result.pca.explained_variance_ratio_,
        figures_dir / "acp1_eboulis.png", tables_dir / "acp1_eboulis_data.csv",
        "ACP1 - Profils moyens par pays - Eboulis des valeurs propres",
    )
    plotting.plot_individuals(
        acp1_scores, figures_dir / "acp1_individus_F1F2.png", tables_dir / "acp1_individus_F1F2_data.csv",
        "ACP1 - Plan des pays (F1-F2)", label_col="country_iso3",
        highlight=focus, highlight_col="country_iso3",
    )
    plotting.plot_correlation_circle(
        acp1_result.var_correlations, figures_dir / "acp1_cercle_correlations.png",
        tables_dir / "acp1_cercle_correlations_data.csv",
        "ACP1 - Cercle des correlations (F1-F2)",
    )

    # Sensibilite : fenetre commune complete (2000-2024)
    logger.info("--- ACP1 (sensibilite) : fenetre commune %d-%d ---", sens_start, sens_end)
    acp1_sens_result, acp1_sens_profiles, _ = pca_profiles.run_acp1(work_df, sens_start, sens_end)
    save_table(acp1_sens_result.eigenvalues, "19_acp1_sensibilite_fenetre_commune_valeurs_propres", tables_dir)
    acp1_sens_scores = acp1_sens_result.scores.reset_index().rename(columns={"index": "country_iso3"})
    save_table(acp1_sens_scores, "20_acp1_sensibilite_fenetre_commune_coordonnees", tables_dir)

    # Sensibilite : leave-one-country-out (stabilite de la variance expliquee de F1)
    logger.info("--- ACP1 (sensibilite) : leave-one-country-out ---")
    loco_results = pca_profiles.run_acp1_leave_one_country_out(work_df, year_start, year_end)
    loco_rows = []
    for excluded_country, res in loco_results.items():
        loco_rows.append({
            "pays_exclu": excluded_country,
            "variance_expliquee_F1_pct": res.eigenvalues.iloc[0]["variance_expliquee_pct"],
            "variance_expliquee_F2_pct": res.eigenvalues.iloc[1]["variance_expliquee_pct"] if len(res.eigenvalues) > 1 else None,
        })
    loco_df = pd.DataFrame(loco_rows)
    save_table(loco_df, "21_acp1_sensibilite_leave_one_country_out", tables_dir)

    # ------------------------------------------------------------------
    # 4. ACP2 - trajectoires annuelles
    # ------------------------------------------------------------------
    logger.info("--- ACP2 : trajectoires annuelles (%d-%d, cas complets) ---", year_start, year_end)
    acp2_result, acp2_scores_flat, acp2_incomplete = pca_trajectories.run_acp2(work_df, year_start, year_end)
    logger.info("ACP2 : %d etats complets utilises / %d exclus (incomplets)", acp2_result.n_obs, len(acp2_incomplete))
    save_table(acp2_incomplete, "30_acp2_etats_exclus_incomplets", tables_dir)
    save_table(acp2_result.eigenvalues, "31_acp2_valeurs_propres", tables_dir)
    save_table(acp2_scores_flat, "32_acp2_coordonnees_individus", tables_dir)
    save_table(acp2_result.var_correlations.reset_index(), "33_acp2_correlations_variables", tables_dir)
    save_table(acp2_result.var_contributions.reset_index(), "34_acp2_contributions_variables", tables_dir)
    save_table(acp2_result.var_cos2.reset_index(), "35_acp2_cos2_variables", tables_dir)

    plotting.plot_scree(
        acp2_result.pca.explained_variance_ratio_,
        figures_dir / "acp2_eboulis.png", tables_dir / "acp2_eboulis_data.csv",
        "ACP2 - Trajectoires annuelles - Eboulis des valeurs propres",
    )
    plotting.plot_individuals(
        acp2_scores_flat, figures_dir / "acp2_individus_F1F2_par_pays.png",
        tables_dir / "acp2_individus_F1F2_par_pays_data.csv",
        "ACP2 - Nuage pays-annee (F1-F2), couleur = pays",
        color_col="country_iso3",
    )
    plotting.plot_correlation_circle(
        acp2_result.var_correlations, figures_dir / "acp2_cercle_correlations.png",
        tables_dir / "acp2_cercle_correlations_data.csv",
        "ACP2 - Cercle des correlations (F1-F2)",
    )
    plotting.plot_country_trajectory(
        acp2_scores_flat, focus, figures_dir / f"acp2_trajectoire_{focus}.png",
        tables_dir / f"acp2_trajectoire_{focus}_data.csv",
        f"ACP2 - Trajectoire {focus} dans le plan F1-F2 ({year_start}-{year_end})",
    )
    for c in sorted(acp2_scores_flat["country_iso3"].unique()):
        if c == focus:
            continue
        plotting.plot_country_trajectory(
            acp2_scores_flat, c, figures_dir / f"acp2_trajectoire_{c}.png",
            tables_dir / f"acp2_trajectoire_{c}_data.csv",
            f"ACP2 - Trajectoire {c} dans le plan F1-F2 ({year_start}-{year_end})",
        )

    # ------------------------------------------------------------------
    # 5. Typologie des pays (clustering) sur ACP1 (espace pays) et ACP2
    #    (centroides de trajectoire par pays, pour comparaison)
    # ------------------------------------------------------------------
    logger.info("--- Typologie : K-means vs CAH, k=2,3 ---")
    k_values = cfg["clustering"]["k_values"]
    seed = cfg["clustering"]["random_state"]
    n_init = cfg["clustering"]["n_init"]

    comp_cols_acp1 = [c for c in acp1_scores.columns if c.startswith("F")][:2]
    acp1_scores_indexed = acp1_scores.set_index("country_iso3")
    cluster_acp1 = clustering.cluster_comparison(acp1_scores_indexed, comp_cols_acp1, k_values, seed, n_init)
    summary_acp1 = clustering.summarize_cluster_comparison(cluster_acp1, "ACP1_profils_pays_F1F2")
    save_table(summary_acp1, "40_typologie_acp1_synthese", tables_dir)
    for k, res in cluster_acp1.items():
        save_table(res["assignments"].reset_index().rename(columns={"index": "country_iso3"}), f"41_typologie_acp1_k{k}_affectations", tables_dir)

    # Centroide de trajectoire par pays dans l'espace ACP2 (moyenne F1,F2 sur
    # la periode), pour une typologie "au niveau pays" coherente avec ACP2.
    comp_cols_acp2 = [c for c in acp2_scores_flat.columns if c.startswith("F")][:2]
    acp2_country_centroids = acp2_scores_flat.groupby("country_iso3")[comp_cols_acp2].mean()
    cluster_acp2 = clustering.cluster_comparison(acp2_country_centroids, comp_cols_acp2, k_values, seed, n_init)
    summary_acp2 = clustering.summarize_cluster_comparison(cluster_acp2, "ACP2_centroides_trajectoires_F1F2")
    save_table(summary_acp2, "42_typologie_acp2_synthese", tables_dir)
    for k, res in cluster_acp2.items():
        save_table(res["assignments"].reset_index().rename(columns={"index": "country_iso3"}), f"43_typologie_acp2_k{k}_affectations", tables_dir)

    # ------------------------------------------------------------------
    # 6. Recapitulatif d'execution
    # ------------------------------------------------------------------
    run_summary = {
        "environnement": env,
        "empreintes_sha256": fingerprints,
        "fenetre_principale": [year_start, year_end],
        "fenetre_sensibilite_commune": [sens_start, sens_end],
        "fenetre_commune_complete_detectee": common_window,
        "n_etats_complets": n_complete,
        "n_etats_incomplets": n_incomplete,
        "n_pays": work_df["country_iso3"].nunique(),
        "n_indicateurs": len(io_utils.indicator_columns(work_df)),
        "acp1_n_individus": acp1_result.n_obs,
        "acp1_n_variables": len(acp1_result.variables),
        "acp2_n_individus_complets": acp2_result.n_obs,
        "acp2_n_etats_exclus": len(acp2_incomplete),
    }
    with (tables_dir / "00_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Recapitulatif d'execution : %s", json.dumps(run_summary, ensure_ascii=False, default=str))
    logger.info("=== Fin Etape 1 ===")
    return run_summary


if __name__ == "__main__":
    main()
