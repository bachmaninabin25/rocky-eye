"""Ingredient 06 tests: the likelihood is right, and the fit recovers a KNOWN depth.

Truth in the synthetic data (made with batman, see conftest.py):
    depth 150 ppm, timing offset -20 min, noise 250 ppm, ramp tau 0.4 h.
"""
import numpy as np
import pytest

from conftest import GJ, PRIORS, deps, synthetic_light_curve

F = deps.get("06")
ORB = deps.get("04").Orbit(**GJ)


@pytest.fixture(scope="module")
def visits(three_light_curves):
    return [F.make_visit(f"v{k}", lc, "exp+x") for k, lc in enumerate(three_light_curves)]


def test_parameter_layout(visits):
    fit = F.EclipseFit(visits, ORB, PRIORS)
    assert fit.names[:2] == ["depth_ppm", "dt_min"]
    assert fit.names[2:4] == ["log_sigma:v0", "log_tau:v0"]
    assert fit.ndim == 2 + 3 * 2


def test_eclipse_time_offset_is_in_minutes(visits):
    fit = F.EclipseFit(visits, ORB, PRIORS)
    p = np.array([150, -20] + [np.log(250e-6), np.log(0.4)] * 3)
    assert fit.t_ecl(p, 1) == pytest.approx(fit.t_pred[1] - 20 / 1440)


def test_design_matrix(visits):
    fit = F.EclipseFit(visits, ORB, PRIORS)
    p = np.array([150, -20] + [np.log(250e-6), np.log(0.4)] * 3)
    M = fit.design(p, 0)
    assert M.shape == (visits[0].n, 4)                      # const, slope, ramp, x
    p0 = p.copy(); p0[0] = 0.0
    assert np.allclose(fit.design(p0, 0)[:, 0], 1.0)        # no eclipse -> ones column untouched
    assert M[:, 0].min() == pytest.approx(1.0, abs=1e-9)                # mid-eclipse: planet hidden
    assert M[:, 0].max() == pytest.approx(1 + 150e-6)       # out of eclipse: 1 + depth


def test_prior_bounds(visits):
    fit = F.EclipseFit(visits, ORB, PRIORS)
    p = np.array([150, -20] + [np.log(250e-6), np.log(0.4)] * 3)
    assert np.isfinite(fit.log_prob(p))
    for idx, val in ((0, 5000), (1, 90), (2, np.log(10.0)), (3, np.log(100.0))):
        bad = p.copy(); bad[idx] = val
        assert fit.log_prob(bad) == -np.inf, f"parameter {fit.names[idx]}={val} should be outside the prior"


def test_exp2_timescales_must_be_ordered(three_light_curves):
    v = F.make_visit("v", three_light_curves[0], "exp2")
    fit = F.EclipseFit([v], ORB, PRIORS)
    good = np.array([150, -20, np.log(250e-6), np.log(0.2), np.log(2.0)])
    bad = np.array([150, -20, np.log(250e-6), np.log(2.0), np.log(0.2)])
    assert np.isfinite(fit.log_prior(good)) and fit.log_prior(bad) == -np.inf


def test_marginal_likelihood_matches_brute_force():
    """Analytic marginalisation over the linear coefficient == numerical integration over it."""
    lc = synthetic_light_curve(amp=0.0, slope=0.0, cx=0.0, cadence_s=400, seed=5)
    v = F.make_visit("v", lc, "const")
    fit = F.EclipseFit([v], ORB, PRIORS)
    ls = np.log(250e-6)

    def brute(depth):
        E = fit.design(np.array([depth, -20.0, ls]), 0)[:, 0]
        c = np.linspace(0.999, 1.001, 20001)
        r = v.flux[None, :] - c[:, None] * E[None, :]
        ll = -0.5 * np.sum(r ** 2, axis=1) / np.exp(2 * ls) - v.n * (ls + 0.5 * np.log(2 * np.pi))
        return np.log(np.trapezoid(np.exp(ll - ll.max()), c)) + ll.max()

    a = fit.log_like(np.array([150.0, -20.0, ls])) - fit.log_like(np.array([0.0, -20.0, ls]))
    assert a == pytest.approx(brute(150.0) - brute(0.0), abs=1e-3)


def test_max_likelihood_and_bic(visits):
    fit = F.EclipseFit(visits, ORB, PRIORS)
    ml = fit.max_likelihood()
    assert ml["p"][0] == pytest.approx(150, abs=60)
    assert ml["p"][1] == pytest.approx(-20, abs=3)
    assert all(abs(s - 250) < 30 for s in ml["sigma_ppm"])
    assert ml["k"] == 2 + 3 + 3 + 3 * 4                      # depth, dt, 3 taus, 3 sigmas, 3x4 linear
    flat = fit.max_likelihood(depth_fixed=0.0)
    assert flat["k"] == ml["k"] - 2                          # depth and timing removed
    assert flat["bic"] - ml["bic"] > 10                      # the eclipse is strongly preferred


@pytest.mark.slow
def test_mcmc_recovers_depth(visits):
    fit = F.EclipseFit(visits, ORB, PRIORS)
    res = fit.run_mcmc(fit.max_likelihood()["p"], n_walkers=32, n_steps=3000)
    assert abs(res["depth_median"] - 150) < 2 * res["depth_std"]
    assert 10 < res["depth_std"] < 40
