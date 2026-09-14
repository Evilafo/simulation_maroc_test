"""Tests de non-regression pour l'Etape 3 (equations du cadrage, garde-fous
poids, coherence des scenarios).

Execution : "C:/Users/MR KITOHOU/anaconda3/python.exe" -m pytest tests/ -q
(depuis mef_maroc_ucs1/).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import io_utils
from src.etape3 import combination as comb
from src.etape3 import config_check


@pytest.fixture(scope="module")
def cfg2():
    return io_utils.load_config(Path(__file__).resolve().parents[1] / "configs" / "config_etape2.yaml")


@pytest.fixture(scope="module")
def cfg3():
    return io_utils.load_config(Path(__file__).resolve().parents[1] / "configs" / "config_etape3.yaml")


def test_config_etape3_validates(cfg2, cfg3):
    checks = config_check.validate_config_etape3(cfg3, cfg2)
    assert len(checks) >= 8


def test_config_rejects_seuil_applicable(cfg2, cfg3):
    bad = {**cfg3, "donor_eligibility": {**cfg3["donor_eligibility"], "seuil_0_55_applicable": True}}
    with pytest.raises(config_check.ConfigError):
        config_check.validate_config_etape3(bad, cfg2)


def test_config_rejects_dynamic_adjustment_enabled(cfg2, cfg3):
    bad = {**cfg3, "scenarios": {**cfg3["scenarios"], "dynamic_adjustment_enabled": True}}
    with pytest.raises(config_check.ConfigError):
        config_check.validate_config_etape3(bad, cfg2)


def _toy_data():
    horizons = [1, 2, 3]
    years = [2025, 2026, 2027]
    x_tend = pd.DataFrame({"cible": ["PIB"] * 3, "horizon": horizons, "annee_projetee": years, "valeur_projetee": [3.0, 3.2, 3.4], "horizon_teste_backtest": [True, True, False]})
    donor_proj = pd.DataFrame({
        "pays": ["VNM", "VNM", "VNM", "CRI", "CRI", "CRI"],
        "cible": ["PIB"] * 6,
        "horizon": horizons * 2,
        "annee_projetee": years * 2,
        "valeur_projetee": [6.0, 6.1, 6.2, 4.0, 4.1, 4.2],
    })
    return x_tend, donor_proj


def test_benchmark_combine_weights_must_sum_to_one():
    _, donor_proj = _toy_data()
    with pytest.raises(ValueError):
        comb.benchmark_combine(donor_proj, {"VNM": 0.5, "CRI": 0.3})


def test_benchmark_combine_rejects_negative_weight():
    _, donor_proj = _toy_data()
    with pytest.raises(ValueError):
        comb.benchmark_combine(donor_proj, {"VNM": 1.2, "CRI": -0.2})


def test_benchmark_combine_matches_manual_computation():
    _, donor_proj = _toy_data()
    result = comb.benchmark_combine(donor_proj, {"VNM": 0.5, "CRI": 0.5})
    expected_h1 = 0.5 * 6.0 + 0.5 * 4.0
    assert result[result["horizon"] == 1]["X_bench_comb"].iloc[0] == pytest.approx(expected_h1)


def test_scenario_alpha_zero_equals_tendance():
    x_tend, donor_proj = _toy_data()
    bench = comb.benchmark_combine(donor_proj, {"VNM": 0.5, "CRI": 0.5})
    scen = comb.scenario_trajectory(x_tend, bench, alpha=0.0, scenario_name="test")
    assert np.allclose(scen["X_scenario"], scen["X_tend_MAR"])


def test_scenario_alpha_one_equals_benchmark():
    x_tend, donor_proj = _toy_data()
    bench = comb.benchmark_combine(donor_proj, {"VNM": 0.5, "CRI": 0.5})
    scen = comb.scenario_trajectory(x_tend, bench, alpha=1.0, scenario_name="test")
    assert np.allclose(scen["X_scenario"], scen["X_bench_comb"])


def test_scenario_rejects_alpha_out_of_bounds():
    x_tend, donor_proj = _toy_data()
    bench = comb.benchmark_combine(donor_proj, {"VNM": 1.0})
    with pytest.raises(ValueError):
        comb.scenario_trajectory(x_tend, bench, alpha=1.5, scenario_name="bad")


def test_single_donor_portfolio_reproduces_its_own_trajectory():
    _, donor_proj = _toy_data()
    bench = comb.benchmark_combine(donor_proj, {"VNM": 1.0})
    ref = donor_proj[donor_proj["pays"] == "VNM"]
    merged = bench.merge(ref, on=["cible", "horizon", "annee_projetee"])
    assert np.allclose(merged["X_bench_comb"], merged["valeur_projetee"])


def test_sanity_checks_all_pass_on_toy_data():
    x_tend, donor_proj = _toy_data()
    bench = comb.benchmark_combine(donor_proj, {"VNM": 0.5, "CRI": 0.5})
    checks = comb.sanity_checks(x_tend, bench, {"VNM": 1.0}, donor_proj)
    assert checks["reussi"].all(), checks[~checks["reussi"]].to_string()
