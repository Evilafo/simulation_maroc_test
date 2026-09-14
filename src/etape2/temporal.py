"""Protocole temporel : prevision directe par horizon, sans fuite.

Principe (cf. demande, section 1) : pour une origine O et un horizon h, le
modele utilise seulement l'information connue a O pour predire y_{O+h}.
Pour entrainer ce modele a l'origine O, seules les paires (o', y_{o'+h})
dont l'etiquette est deja connue a O (c'est-a-dire o'+h <= O) sont
utilisables. Ce n'est PAS la meme chose que "o' < O" des que h > 1 : les
origines les plus recentes ne peuvent pas encore servir a l'entrainement
pour un horizon eleve.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DirectDataset:
    """Table (origine, features..., annee_cible, cible) pour un horizon h
    donne, pret pour le walk-forward."""
    horizon: int
    frame: pd.DataFrame  # index = origin_year ; colonnes features + "target_year", "y"


def build_direct_dataset(
    feature_frame: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
) -> DirectDataset:
    """feature_frame : index=year, colonnes=features (deja calculees de
    maniere causale, cf. features.py). target_series : index=year, valeurs
    de la cible. Retourne une table avec, pour chaque origine O disposant de
    features completes, l'annee cible O+h et sa valeur (NaN si non
    observee -- ces lignes sont conservees mais exclues de l'evaluation en
    aval, jamais utilisees comme etiquette d'entrainement)."""
    rows = feature_frame.copy()
    rows["target_year"] = rows.index + horizon
    rows["y"] = rows["target_year"].map(target_series)
    return DirectDataset(horizon=horizon, frame=rows)


def training_rows(ds: DirectDataset, origin: int, min_origin: int) -> pd.DataFrame:
    """Lignes utilisables pour entrainer le modele qui produira la
    prevision a l'origine `origin` : etiquette connue avant ou a `origin`
    (target_year <= origin), origine >= min_origin (contrainte de retards
    disponibles), et etiquette reellement observee (pas de trou)."""
    f = ds.frame
    mask = (f["target_year"] <= origin) & (f.index >= min_origin) & f["y"].notna()
    return f.loc[mask]


def test_row(ds: DirectDataset, origin: int) -> pd.DataFrame:
    """La ligne (features seules, a l'origine) utilisee pour produire la
    prevision de l'annee origin+horizon."""
    f = ds.frame
    return f.loc[f.index == origin]


def evaluation_origins(
    ds: DirectDataset,
    internal_train_end: int,
    final_start: int,
    final_end: int,
    min_origin: int,
    max_origin: int,
) -> dict[str, list[int]]:
    """Partitionne les origines evaluables (target_year observee) en bloc
    interne (target_year <= internal_train_end) et bloc final (target_year
    dans [final_start, final_end]). Une origine n'est retenue que si elle
    dispose d'au moins une observation d'entrainement anterieure (verifie en
    aval, pas ici)."""
    f = ds.frame
    candidates = f[(f.index >= min_origin) & (f.index <= max_origin) & f["y"].notna()]
    internal = sorted(candidates[candidates["target_year"] <= internal_train_end].index.tolist())
    final = sorted(
        candidates[(candidates["target_year"] >= final_start) & (candidates["target_year"] <= final_end)].index.tolist()
    )
    return {"internal": internal, "final": final}
