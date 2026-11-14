"""Ingredient 09: radiative physics against hand calculations and literature numbers."""
import numpy as np
import pytest

from rockyeye import s09_physics as ph


@pytest.fixture(scope="module")
def band():
    return ph.Band()


def test_planck_against_hand_value():
    # B_nu(15 um, 300 K): x = h nu / kT = 3.1976..., standard formula
    nu = ph.C / 15e-6
    x = ph.H * nu / (ph.KB * 300.0)
    assert ph.planck_nu(15.0, 300.0) == pytest.approx(2 * ph.H * nu ** 3 / ph.C ** 2 / np.expm1(x))
    # Rayleigh-Jeans limit at high temperature
    assert ph.planck_nu(15.0, 1e6) == pytest.approx(2 * nu ** 2 * ph.KB * 1e6 / ph.C ** 2, rel=1e-3)


def test_band_is_f1500w(band):
    assert 14.5 < band.pivot_um < 15.6


def test_brightness_temperature_round_trip(band):
    for T in (250.0, 700.0, 3000.0):
        assert band.brightness_temperature(band.planck(T)) == pytest.approx(T, rel=1e-4)


def test_bare_rock_temperature_gj3929b():
    """Xue et al. 2025 quote 737 +/- 14 K for T_eff = 3369 K, a/Rs = 17.05."""
    assert ph.dayside_temperature(3369.0, 17.05, 0.0, 0.0) == pytest.approx(737.0, abs=2.0)
    # full redistribution: f = 1/4 instead of 2/3 -> factor (3/8)^(1/4)
    assert ph.dayside_temperature(3369.0, 17.05, 0.0, 1.0) == pytest.approx(737.3 * (3 / 8) ** 0.25, abs=2.0)


def test_depth_temperature_inverse(band):
    i_star = band.planck(3000.0)
    d = ph.depth_from_temperature(band, 737.0, 0.0318, i_star)
    assert 100e-6 < d < 200e-6                       # a GJ 3929 b-like bare rock is ~150 ppm deep
    assert ph.temperature_from_depth(band, d, 0.0318, i_star) == pytest.approx(737.0, rel=1e-4)
    assert np.isnan(ph.temperature_from_depth(band, -1e-5, 0.0318, i_star))


def test_flux_to_intensity_matches_blackbody(band):
    """A blackbody star's flux density at distance d converts back to its own intensity."""
    T, r, d = 3000.0, 0.3, 10.0
    f_mjy = np.pi * band.planck(T) * (r * ph.RSUN / (d * ph.PC)) ** 2 / 1e-29
    i = ph.stellar_intensity(band, "flux", flux_mjy=f_mjy, rstar_rsun=r, dist_pc=d)
    assert band.brightness_temperature(i) == pytest.approx(T, rel=1e-4)


def test_verdict_prefers_the_true_scenario(band):
    rng = np.random.default_rng(0)
    system = {"teff": 3369.0, "teff_err": 50.0, "a_rs": 17.05, "a_rs_err": 0.4, "rp_rs": 0.0318, "rp_rs_err": 0.0007}
    draws = ph.draw_system(system, 5000, rng)
    i_star = ph.stellar_intensity(band, "blackbody", teff=draws["teff"])
    scen = {"rock": {"albedo": 0.0, "eps": 0.0}, "atmosphere": {"albedo": 0.0, "eps": 1.0}}
    pred = ph.scenario_depths(band, draws, i_star, scen)
    assert pred["rock"].mean() > 1.4 * pred["atmosphere"].mean()
    for truth, other in (("rock", "atmosphere"), ("atmosphere", "rock")):
        samples = rng.normal(pred[truth].mean(), 15.0, 20000)       # a measurement centred on the truth
        comp = ph.compare_scenarios(samples, pred, reference=other)
        assert comp[truth]["bayes_factor_vs_ref"] > 10
        assert abs(comp[truth]["tension_sigma"]) < 1
        assert "for" in ph.interpret_bayes_factor(comp[truth]["bayes_factor_vs_ref"])
