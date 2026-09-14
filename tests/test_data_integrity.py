"""Tests de non-regression sur les donnees et le pipeline Etape 1.

Execution : "C:/Users/MR KITOHOU/anaconda3/python.exe" -m pytest tests/ -q
(depuis le dossier mef_maroc_ucs1/).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src import audit, io_utils, pca_profiles, pca_trajectories


@pytest.fixture(scope="module")
def cfg():
    return io_utils.load_config()


@pytest.fixture(scope="module")
def work_df(cfg):
    return io_utils.load_work_base(cfg)


def test_work_base_shape(work_df):
    assert work_df.shape == (306, 30), (
        "Le fichier de travail doit rester 306 lignes x 30 colonnes "
        "(country_iso3, year, 28 indicateurs) tel que specifie dans la demande."
    )


def test_no_duplicate_country_year(work_df):
    assert work_df.duplicated(subset=["country_iso3", "year"]).sum() == 0


def test_countries_match_expected(cfg, work_df):
    expected = set(cfg["scope"]["countries"])
    assert set(work_df["country_iso3"].unique()) == expected


def test_no_missing_years_1991_2024(work_df):
    for country, g in work_df.groupby("country_iso3"):
        years = set(g["year"].tolist())
        assert years == set(range(1991, 2025)), f"{country}: annees manquantes ou en trop"


def test_completeness_counts_match_spec(work_df):
    states = audit.completeness_by_state(work_df)
    assert int(states["etat_complet"].sum()) == 278
    assert int((~states["etat_complet"]).sum()) == 28
    assert int(states["n_manquants"].sum()) == 101


def test_dette_publique_and_tcer_excluded(work_df):
    assert "DETTE_PUBLIQUE" not in work_df.columns
    assert "TCER" not in work_df.columns


def test_exact_identities_hold(work_df):
    identities = audit.exact_linear_identities(work_df)
    exact_flags = dict(zip(identities["identite"], identities["exacte"]))
    assert exact_flags["SOBG = RETO - DETO"] is True
    assert exact_flags["OUVERTURE_COM = EXPO + IMPO"] is True
    assert exact_flags["EMAG + EMIN + EMSE = 100"] is True


def test_acp1_runs_and_has_expected_rank(work_df):
    result, profiles, _ = pca_profiles.run_acp1(work_df, 1991, 2024)
    assert profiles.shape[0] == 9  # 9 pays
    assert result.n_obs == 9
    assert len(result.eigenvalues) == 8  # rang centre max = 9-1


def test_acp2_uses_complete_cases_only(work_df):
    result, scores_flat, incomplete = pca_trajectories.run_acp2(work_df, 1991, 2024)
    assert result.n_obs == 278
    assert len(incomplete) == 28
    assert len(scores_flat) == 278
