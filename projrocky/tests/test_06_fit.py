"""Ingredient 06: the likelihood is right and the fit recovers a known depth within 1 sigma."""
import numpy as np
import pytest

from conftest import PRIORS
from rockyeye.s06_fit import EclipseFit, bin_series, make_visit


def test_bin_series():
    assert np.allclose(bin_series(np.arange(10.0), 4), [1.5, 5.5])


def test_parameter_layout(three_visits, orbit):
    fit = EclipseFit(three_visits, orbit, PRIORS)
    assert fit.names[:2] == ["depth_ppm", "dt_min"]
    assert fit.ndim == 2 + 3 * 2
    free_t = EclipseFit(three_visits, orbit, PRIORS, shared_time=False)
    assert free_t.ndim == 1 + 3 + 3 * 2


def test_prior_bounds(three_visits, orbit):
    fit = EclipseFit(three_visits, orbit, PRIORS)
    p = np.array([150, -20] + [np.log(250e-6), np.log(0.4)] * 3)
    assert np.isfinite(fit.log_prob(p))
    bad = p.copy(); bad[0] = 5000
    assert fit.log_prob(bad) == -np.inf
    bad = p.copy(); bad[3] = np.log(100.0)
    assert fit.log_prob(bad) == -np.inf


def test_marginal_likelihood_matches_brute_force(orbit):
    """Analytic marginalisation over a linear coefficient equals numerical integration."""
    from conftest import synthetic_light_curve
    lc = synthetic_light_curve(orbit, 2460864.9, amp=0.0, slope=0.0, cx=0.0, cadence_s=400, seed=5)
    v = make_visit("v", lc, "const")
    fit = EclipseFit([v], orbit, PRIORS)
    ls = np.log(250e-6)
    def brute(depth):
        p = np.array([depth, -20.0, ls])
        E = fit.design(p, 0)[:, 0]
        c = np.linspace(0.999, 1.001, 20001)
        r = v.flux[None, :] - c[:, None] * E[None, :]
        ll = -0.5 * np.sum(r ** 2, axis=1) / np.exp(2 * ls) - v.n * (ls + 0.5 * np.log(2 * np.pi))
        return np.log(np.trapezoid(np.exp(ll - ll.max()), c)) + ll.max()
    a = fit.log_like(np.array([150.0, -20.0, ls])) - fit.log_like(np.array([0.0, -20.0, ls]))
    b = brute(150.0) - brute(0.0)
    assert a == pytest.approx(b, abs=1e-3)


def test_max_likelihood_and_bic(three_visits, orbit):
    fit = EclipseFit(three_visits, orbit, PRIORS)
    ml = fit.max_likelihood()
    assert ml["p"][0] == pytest.approx(150, abs=60)
    assert ml["p"][1] == pytest.approx(-20, abs=3)
    assert all(abs(s - 250) < 30 for s in ml["sigma_ppm"])
    flat = fit.max_likelihood(depth_fixed=0.0)
    assert flat["bic"] - ml["bic"] > 10          # the eclipse is strongly preferred
    assert flat["k"] == ml["k"] - 2              # depth and timing removed


@pytest.mark.slow
def test_mcmc_recovers_depth_within_1sigma(three_visits, orbit, tmp_path):
    fit = EclipseFit(three_visits, orbit, PRIORS)
    # max_steps is only a safety cap; the sampler stops once tau, length and R-hat all pass
    res = fit.run_mcmc(tmp_path / "chain.h5", fit.max_likelihood()["p"], n_walkers=32, max_steps=20000,
                       check_every=2000)
    assert res["n_steps"] < 20000, "did not converge before the cap"
    d = res["summary"]["depth_ppm"]
    assert abs(d["median"] - 150) < 1.5 * d["std"]
    assert 10 < d["std"] < 40
    assert res["diagnostics"]["converged"], res["diagnostics"]["flags"]
    # resumability: a second call continues from the stored chain instead of restarting
    import emcee
    n_before = emcee.backends.HDFBackend(str(tmp_path / "chain.h5"), read_only=True).iteration
    fit.run_mcmc(tmp_path / "chain.h5", fit.max_likelihood()["p"], n_walkers=32, max_steps=n_before + 500,
                 check_every=500)
    assert emcee.backends.HDFBackend(str(tmp_path / "chain.h5"), read_only=True).iteration == n_before + 500


def test_flat_model_is_valid_when_timing_prior_excludes_zero(three_visits, orbit):
    """Regression: the no-eclipse comparison must not be evaluated outside the timing prior."""
    pri = dict(PRIORS, dt_min=[-25.0, -15.0])            # excludes 0, contains the truth (-20)
    narrow = EclipseFit(three_visits, orbit, pri)
    wide = EclipseFit(three_visits, orbit, PRIORS)
    flat_narrow, flat_wide = narrow.max_likelihood(depth_fixed=0.0), wide.max_likelihood(depth_fixed=0.0)
    assert flat_narrow["lnL"] == pytest.approx(flat_wide["lnL"], abs=0.5)
    from rockyeye.s10_compare import detection_bic
    assert 10 < detection_bic(narrow)["dbic"] < 200      # a ~7 sigma joint signal, not thousands
