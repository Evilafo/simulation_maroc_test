"""Etape 2, partie 3 : synthese des verdicts par cible/horizon/modele
(statuts implemente/execute/evalue/valide/non concluant/bloque), a partir
de e2_11_evaluation_vs_baseline.csv.

Usage : "C:/Users/MR KITOHOU/anaconda3/python.exe" run_etape2_part3_synthese.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src import io_utils

PROJECT_ROOT = Path(__file__).resolve().parent


def verdict_row(row: pd.Series, family: str) -> dict:
    n = row["n_previsions"]
    if n < 4:
        return {"statut": "non_concluant", "verdict": f"n={n} previsions : effectif insuffisant"}

    dm_signif = bool(row.get("dm_pvalue_hln") is not None and not pd.isna(row.get("dm_pvalue_hln")) and row["dm_pvalue_hln"] < 0.10)

    if family == "variation":
        seuil_ok = bool(row.get("seuil_gain_atteint"))
        if seuil_ok and dm_signif:
            return {"statut": "evalue", "verdict": "gain >= seuil ET DM significatif (p<0.10) : resultat le plus solide obtenu a ce stade, non equivalent a une validation operationnelle"}
        if seuil_ok:
            return {"statut": "evalue", "verdict": "gain >= seuil observe, DM non significatif ou non calcule (n insuffisant) : amelioration observee non confirmee statistiquement"}
        return {"statut": "evalue", "verdict": "baseline non battue au seuil de 10% sur ce bloc"}

    rel_ok = bool(row.get("rmse_relative_modele") is not None and row.get("rmse_relative_baseline") is not None
                  and not pd.isna(row.get("rmse_relative_modele")) and row["rmse_relative_modele"] < row["rmse_relative_baseline"])
    no_drift = not bool(row.get("derive_systematique"))
    if rel_ok and no_drift and dm_signif:
        return {"statut": "evalue", "verdict": "RMSE relative < baseline, pas de derive systematique, DM significatif : resultat le plus solide obtenu a ce stade"}
    if rel_ok and no_drift:
        return {"statut": "evalue", "verdict": "RMSE relative < baseline et pas de derive, DM non significatif ou non calcule"}
    if rel_ok and not no_drift:
        return {"statut": "evalue", "verdict": "RMSE relative < baseline MAIS derive systematique detectee (biais > 0.5*RMSE) : a ne pas retenir tel quel"}
    return {"statut": "evalue", "verdict": "baseline non battue (RMSE relative) sur ce bloc"}


def main() -> None:
    cfg2 = io_utils.load_config(PROJECT_ROOT / "configs" / "config_etape2.yaml")
    tables_dir = PROJECT_ROOT / cfg2["paths"]["outputs_tables_dir"]

    eval_df = pd.read_csv(tables_dir / "e2_11_evaluation_vs_baseline.csv")
    families = {t: spec["family"] for t, spec in cfg2["targets"].items()}

    rows = []
    for _, r in eval_df.iterrows():
        fam = families[r["cible"]]
        v = verdict_row(r, fam)
        rows.append({**r.to_dict(), "famille": fam, **v})
    verdicts = pd.DataFrame(rows)
    verdicts.to_csv(tables_dir / "e2_12_verdicts.csv", index=False)

    # Meilleur modele candidat (hors baselines) par cible/horizon sur le
    # bloc final, selon la metrique pertinente a la famille.
    finale = verdicts[verdicts["bloc"] == "finale"].copy()

    def _score(row):
        if row["famille"] == "variation":
            return -row["gain_rmse_vs_baseline_pct"] if pd.notna(row.get("gain_rmse_vs_baseline_pct")) else np.inf
        return row["rmse_relative_modele"] if pd.notna(row.get("rmse_relative_modele")) else np.inf

    finale["score_tri"] = finale.apply(_score, axis=1)
    best = finale.sort_values("score_tri").groupby(["cible", "horizon"], as_index=False).first()
    best = best[["cible", "horizon", "famille", "modele", "n_previsions", "rmse_modele", "rmse_baseline",
                 "gain_rmse_vs_baseline_pct", "rmse_relative_modele", "rmse_relative_baseline",
                 "biais_moyen_modele", "derive_systematique", "dm_pvalue_hln", "statut", "verdict"]]
    best.to_csv(tables_dir / "e2_13_meilleur_modele_par_cible_horizon.csv", index=False)

    print(best.to_string())


if __name__ == "__main__":
    main()
