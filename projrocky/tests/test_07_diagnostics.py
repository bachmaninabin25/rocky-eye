"""Ingredient 07: good chains pass, deliberately bad chains are flagged."""
import numpy as np

from rockyeye.s07_diagnostics import diagnose, integrated_autocorr_time, split_rhat


def ar1(n_steps, n_walkers, rho, rng, offsets=None):
    """AR(1) chains with autocorrelation rho; tau = (1+rho)/(1-rho)."""
    x = np.zeros((n_steps, n_walkers, 1))
    x[0] = rng.normal(size=(n_walkers, 1))
    for i in range(1, n_steps):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=(n_walkers, 1))
    if offsets is not None:
        x += np.asarray(offsets)[None, :, None]
    return x


def test_autocorrelation_time_of_ar1():
    rng = np.random.default_rng(0)
    tau = integrated_autocorr_time(ar1(20000, 16, 0.9, rng))[0]
    assert abs(tau - 19.0) < 3.0


def test_good_chain_converges():
    rng = np.random.default_rng(1)
    d = diagnose(ar1(8000, 32, 0.8, rng), ["x"], acceptance=np.full(32, 0.3))
    assert d["converged"], d["flags"]
    assert d["rhat"][0] < 1.01


def test_flags_walkers_that_disagree():
    rng = np.random.default_rng(2)
    chain = ar1(8000, 32, 0.8, rng, offsets=np.where(np.arange(32) < 16, 0.0, 3.0))
    d = diagnose(chain, ["x"])
    assert not d["converged"]
    assert split_rhat(chain)[0] > 1.1
    assert any("R-hat" in f for f in d["flags"])


def test_flags_chain_that_is_too_short():
    rng = np.random.default_rng(3)
    d = diagnose(ar1(600, 16, 0.99, rng), ["x"])          # tau ~ 200 but only 600 steps
    assert not d["converged"]
    assert any("tau long" in f for f in d["flags"])


def test_flags_drifting_chain_and_stuck_walkers():
    rng = np.random.default_rng(4)
    chain = ar1(8000, 32, 0.5, rng) + np.linspace(0, 4, 8000)[:, None, None]   # still burning in
    acc = np.full(32, 0.3); acc[:3] = 0.0
    d = diagnose(chain, ["x"], acceptance=acc)
    assert not d["converged"]
    assert any("stuck" in f for f in d["flags"])
