"""Ingredient 09 tests: radiative physics against hand calculations and a published number."""
import numpy as np
import pytest

from conftest import deps

P = deps.get("09")


@pytest.fixture(scope="module")
def band():
    return P.Band()


def test_planck_against_hand_value():
    nu = P.C / 15e-6
    x = P.H * nu / (P.KB * 300.0)
    assert P.planck_nu(15.0, 300.0) == pytest.approx(2 * P.H * nu ** 3 / P.C ** 2 / np.expm1(x))
    # Rayleigh-Jeans limit (h nu << kT): B_nu -> 2 nu^2 k T / c^2
    assert P.planck_nu(15.0, 1e6) == pytest.approx(2 * nu ** 2 * P.KB * 1e6 / P.C ** 2, rel=1e-3)


def test_band_pivot(band):
    assert 14.5 < band.pivot_um < 15.6


def test_brightness_temperature_round_trip(band):
    for T in (250.0, 700.0, 3000.0):
        assert band.brightness_temperature(band.planck(T)) == pytest.approx(T, rel=1e-4)


def test_band_average_lies_between_the_edges(band):
    T = 700.0
    edges = P.planck_nu(np.array([13.5, 16.6]), T)
    assert edges.min() < band.planck(T) < edges.max()


def test_bare_rock_temperature_gj3929b():
    """Xue et al. 2025 quote 737 +/- 14 K for T_eff = 3369 K, a/Rs = 17.05."""
    assert P.dayside_temperature(3369.0, 17.05) == pytest.approx(737.0, abs=2.0)
    # full redistribution: (2/3) -> (1/4), so T drops by (3/8)^(1/4)
    assert P.dayside_temperature(3369.0, 17.05, 0.0, 1.0) == pytest.approx(737.3 * (3 / 8) ** 0.25, abs=2.0)
    # a reflective surface is cooler
    assert P.dayside_temperature(3369.0, 17.05, 0.3) < P.dayside_temperature(3369.0, 17.05)


def test_depth_temperature_inverse(band):
    i_star = band.planck(3000.0)
    d = P.depth_from_temperature(band, 737.0, 0.0318, i_star)
    assert 100e-6 < d < 200e-6                      # a GJ 3929 b-like bare rock is ~150 ppm deep
    assert P.temperature_from_depth(band, d, 0.0318, i_star) == pytest.approx(737.0, rel=1e-4)
    assert np.isnan(P.temperature_from_depth(band, -1e-5, 0.0318, i_star))


def test_verdict_prefers_the_true_scenario(band):
    rng = np.random.default_rng(0)
    teff = rng.normal(3369, 50, 5000)
    a_rs = rng.normal(17.05, 0.4, 5000)
    rp = rng.normal(0.0318, 0.0007, 5000)
    i_star = P.stellar_intensity(band, teff)
    pred = {name: 1e6 * P.depth_from_temperature(band, P.dayside_temperature(teff, a_rs, 0.0, eps), rp, i_star)
            for name, eps in (("rock", 0.0), ("atmosphere", 1.0))}
    assert pred["rock"].mean() > 1.4 * pred["atmosphere"].mean()
    for truth, other in (("rock", "atmosphere"), ("atmosphere", "rock")):
        samples = rng.normal(pred[truth].mean(), 15.0, 20000)       # a measurement centred on the truth
        comp = P.compare_scenarios(samples, pred, reference=other)
        assert comp[truth]["bayes_factor_vs_ref"] > 10
        assert abs(comp[truth]["tension_sigma"]) < 1
        assert comp[other]["bayes_factor_vs_ref"] == pytest.approx(1.0)
        assert "evidence for" in P.interpret_bayes_factor(comp[truth]["bayes_factor_vs_ref"])


def test_bayes_factor_words():
    assert P.interpret_bayes_factor(1.5) == "no real preference"
    assert P.interpret_bayes_factor(np.exp(3)) == "moderate evidence for"
    assert P.interpret_bayes_factor(np.exp(-6)) == "strong evidence against"
    assert P.interpret_bayes_factor(0.0) == "excluded"
