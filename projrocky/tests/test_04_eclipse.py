"""Ingredient 04: the batman wrapper matches analytic test cases."""
import numpy as np
import pytest

from rockyeye.s04_eclipse import Orbit, eclipse_flux, eclipse_shape, eclipse_shape_analytic


def test_shape_limits(orbit):
    te = orbit.predicted_eclipse_time(2461000.0)
    assert eclipse_shape(np.array([te]), te, orbit)[0] == pytest.approx(0.0, abs=1e-12)
    assert eclipse_shape(np.array([te + 0.2]), te, orbit)[0] == pytest.approx(1.0, abs=1e-12)
    f = eclipse_flux(np.array([te, te + 0.2]), 200e-6, te, orbit)
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(1.0002)


def test_matches_analytic_geometry(orbit):
    te = orbit.predicted_eclipse_time(2461000.0)
    t = te + np.linspace(-0.1, 0.1, 5001)
    assert np.abs(eclipse_shape(t, te, orbit) - eclipse_shape_analytic(t, te, orbit)).max() < 1e-6


def test_durations_match_contact_formulae(orbit):
    te = orbit.predicted_eclipse_time(2461000.0)
    t = te + np.linspace(-0.1, 0.1, 200001)
    s = eclipse_shape(t, te, orbit)
    dt = t[1] - t[0]
    assert np.ptp(t[s < 1]) == pytest.approx(orbit.duration_full(), abs=3 * dt)
    assert np.ptp(t[s == 0]) == pytest.approx(orbit.duration_total(), abs=3 * dt)


def test_predicted_time_is_half_phase(orbit):
    te = orbit.predicted_eclipse_time(2460864.95)
    phase = ((te - orbit.t0) / orbit.period) % 1
    assert phase == pytest.approx(0.5, abs=1e-9)
    assert abs(te - 2460864.95) < orbit.period / 2


def test_transit_is_not_an_eclipse(orbit):
    """The planet is in front of the star at transit: nothing is hidden."""
    t = orbit.t0 + np.linspace(-0.05, 0.05, 11)
    te = orbit.t0 + orbit.period / 2
    assert np.allclose(eclipse_shape_analytic(t, te, orbit), 1.0)
    assert np.allclose(eclipse_shape(t, te, orbit), 1.0)
