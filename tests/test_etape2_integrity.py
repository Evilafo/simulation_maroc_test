"""Tests de non-regression pour l'Etape 2 (donnees etendues, protocole
temporel, absence de fuite, coherence des sorties).

Execution : "C:/Users/MR KITOHOU/anaconda3/python.exe" -m pytest tests/ -q
(depuis mef_maroc_ucs1/).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src import io_utils
from src.etape2 import data_extended as dext
from src.etape2 import features as feat
from src.etape2 import temporal as tmp


@pytest.fixture(scope="module")
def cfg1():
    return io_utils.load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")


@pytest.fixture(scope="module")
def cfg2():
    return io_utils.load_config(Path(__file__).resolve().parents[1] / "configs" / "config_etape2.yaml")


@pytest.fixture(scope="module")
def extended_df(cfg1, cfg2):
    return dext.load_extended_work_base(cfg1, cfg2)


def test_extended_dataset_shape(extended_df):
    assert extended_df.shape == (306, 32)  # 28 indicateurs etape1 + DETTE_PUBLIQUE + TCER + id + year


def test_extended_matches_etape1_on_shared_columns(extended_df, cfg1):
    assert dext.verify_against_etape1_work_base(extended_df, cfg1) is True


def test_morocco_has_no_missing_debt_or_tcer(extended_df):
    mar = extended_df[extended_df["country_iso3"] == "MAR"]
    assert mar["DETTE_PUBLIQUE"].isna().sum() == 0
    assert mar["TCER"].isna().sum() == 0


def test_direct_dataset_no_future_leakage():
    """Verifie que pour un horizon h, aucune ligne du jeu direct n'associe
    une origine O a une cible dont l'annee est <= O (la cible doit toujours
    etre strictement future par rapport a l'origine)."""
    idx = list(range(1991, 2025))
    feature_frame = pd.DataFrame({"x": range(len(idx))}, index=idx)
    target = pd.Series(range(len(idx)), index=idx)
    for h in [1, 3, 5]:
        ds = tmp.build_direct_dataset(feature_frame, target, h)
        assert (ds.frame["target_year"] == ds.frame.index + h).all()


def test_training_rows_never_include_future_labels():
    """A l'origine O, les lignes d'entrainement ne doivent jamais avoir une
    annee cible posterieure a O (sinon l'etiquette ne serait pas encore
    connue)."""
    idx = list(range(1991, 2025))
    feature_frame = pd.DataFrame({"x": range(len(idx))}, index=idx)
    target = pd.Series(range(len(idx)), index=idx)
    ds = tmp.build_direct_dataset(feature_frame, target, horizon=3)
    for origin in [2000, 2010, 2020]:
        train = tmp.training_rows(ds, origin, min_origin=1994)
        assert (train["target_year"] <= origin).all()


def test_feature_frame_is_causal_no_future_values():
    """Un retard/une moyenne mobile calcules a l'annee O ne doivent
    dependre d'aucune observation posterieure a O."""
    df_country = pd.DataFrame({"year": range(1991, 2025), "x": range(34)})
    ff = feat.build_feature_frame(df_country, ["x"], max_lags=2, ma_windows=[3], year_start=1991, year_end=2024)
    # x_lag1 a l'annee O doit valoir x a l'annee O-1
    for o in [2000, 2010]:
        assert ff.loc[o, "x_lag1"] == ff.loc[o - 1, "x_lag0"]
        assert ff.loc[o, "x_ma3"] == pytest.approx(ff.loc[o - 3:o - 1, "x_lag0"].mean())


def test_evaluation_table_family_metrics_populated():
    tables_dir = Path(__file__).resolve().parents[1] / "outputs" / "etape2" / "tables"
    path = tables_dir / "e2_11_evaluation_vs_baseline.csv"
    if not path.exists():
        pytest.skip("Etape 2 partie 2 non encore executee dans cet environnement")
    df = pd.read_csv(path)
    assert set(df["modele"].unique()) == {"elasticnet", "panel_avec_effets_fixes", "panel_leave_one_country_out"}
    variation = df[df["cible"].isin(["CROISSANCE_PIB", "INFLATION_CPI", "TACH"])]
    assert variation["gain_rmse_vs_baseline_pct"].notna().any()
    niveau = df[df["cible"].isin(["SOBG", "DETTE_PUBLIQUE", "BALANCE_COURANTE", "FBCF"])]
    assert niveau["rmse_relative_modele"].notna().any()
