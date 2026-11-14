"""Ingredient 08: injected depths come back, at pixel level and at light-curve level."""
import numpy as np
import pytest

from conftest import PRIORS
from rockyeye import s08_injection as inj
from rockyeye.s06_fit import EclipseFit


def test_pixel_injection_scales_only_the_star():
    rng = np.random.default_rng(0)
    sci = rng.normal(100.0, 1.0, (50, 9, 9)).astype(np.float32)
    sci[:, 4, 4] += 1000.0
    bg = np.full(50, 100.0)
    shape = np.ones(50); shape[20:30] = 0.0
    out = inj.inject_into_stamps({"sci": sci}, bg, shape, 0.01)["sci"]
    assert np.allclose(out[:20], sci[:20])                      # untouched out of eclipse
    star_before = (sci[25] - 100).sum(); star_after = (out[25] - 100).sum()
    assert star_after / star_before == pytest.approx(0.99, rel=1e-5)


def test_fast_estimator_matches_truth(three_visits, orbit):
    fit = EclipseFit(three_visits, orbit, PRIORS)
    theta0 = [np.array([np.log(0.4)])] * 3
    best = inj.fast_joint_depth(fit, [v.flux for v in three_visits], theta0, np.linspace(-60, 60, 61))
    assert abs(best["depth"] - 150) < 3 * best["err"]
    assert abs(best["dt"] - (-20)) <= 4


def test_injection_grid_is_unbiased_and_calibrated(three_visits, orbit):
    fit = EclipseFit(three_visits, orbit, PRIORS)
    p = fit.max_likelihood()["p"]
    res = inj.run_injection_grid(fit, p, [0, 150, 300], n_trials=60, dt_grid_min=np.array([0.0]),
                                 scan_timing=False, seed=1)
    for depth, r in res["by_depth"].items():
        assert abs(r["bias"]) < 4 * r["bias_err"] + 1.0, (depth, r["bias"], r["bias_err"])
        assert 0.6 < r["error_ratio"] < 1.6
    assert res["by_depth"][300.0]["detection_rate"] > 0.95
    assert res["by_depth"][0.0]["detection_rate"] < 0.1


def test_visit_without_eclipse_gets_zero_weight(three_visits, orbit):
    """Split visits (GJ 3929 b eclipse 4a/4b): at some trial timings one half has no
    in-eclipse points. That used to raise 'Singular matrix'; it must give no information instead."""
    fit = EclipseFit(three_visits, orbit, PRIORS)
    theta = np.array([np.log(0.4)])
    v = three_visits[0]
    d, e, chi2 = inj.linear_depth(fit, 0, v.t.max() + 1.0, theta, v.flux)
    assert d == 0.0 and np.isinf(e) and np.isfinite(chi2)
    # a joint fit where visit 0's eclipse is pushed a day away still recovers the depth from the others
    t_base = list(fit.t_pred); t_base[0] += 1.0
    best = inj.fast_joint_depth(fit, [vv.flux for vv in three_visits], [theta] * 3,
                                np.linspace(-60, 60, 61), t_base=t_base)
    assert np.isinf(best["per_visit_err"][0]) and abs(best["depth"] - 150) < 4 * best["err"]
