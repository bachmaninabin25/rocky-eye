"""Ingredient 04 tests. Your analytic eclipse must match batman to 1e-6."""
import numpy as np
import pytest

from conftest import GJ, batman_shape, deps

E = deps.get("04")
ORB = E.Orbit(**GJ)


def test_overlap_area_cases():
    d = np.array([0.0, 0.5, 0.95, 1.05, 3.0])
    a = E.circle_overlap_area(1.0, 0.1, d)
    assert a[0] == pytest.approx(np.pi * 0.01)       # fully inside
    assert a[1] == pytest.approx(np.pi * 0.01)
    assert a[4] == 0.0                                # far apart
    assert 0 < a[3] < a[2]                            # partial overlap shrinks with distance


def test_overlap_area_half_way():
    # small circle centred exactly on the big circle's edge: about half of it overlaps
    a = E.circle_overlap_area(1.0, 0.01, np.array([1.0]))[0]
    assert a / (np.pi * 1e-4) == pytest.approx(0.5, abs=0.01)


def test_predicted_time_is_half_phase():
    te = ORB.predicted_eclipse_time(2460864.95)
    assert ((te - ORB.t0) / ORB.period) % 1 == pytest.approx(0.5, abs=1e-9)
    assert abs(te - 2460864.95) < ORB.period / 2


def test_shape_limits():
    te = ORB.predicted_eclipse_time(2461000.0)
    assert E.eclipse_shape(np.array([te]), te, ORB)[0] == pytest.approx(0.0, abs=1e-12)
    assert E.eclipse_shape(np.array([te + 0.2]), te, ORB)[0] == pytest.approx(1.0, abs=1e-12)
    f = E.eclipse_flux(np.array([te, te + 0.2]), 200e-6, te, ORB)
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(1.0002)


def test_matches_batman(ref_orbit):
    te = ORB.predicted_eclipse_time(2461000.0)
    t = te + np.linspace(-0.1, 0.1, 5001)
    assert np.abs(E.eclipse_shape(t, te, ORB) - batman_shape(t, te, ref_orbit)).max() < 1e-6


def test_durations_match_the_shape():
    te = ORB.predicted_eclipse_time(2461000.0)
    t = te + np.linspace(-0.1, 0.1, 200001)
    s = E.eclipse_shape(t, te, ORB)
    dt = t[1] - t[0]
    assert np.ptp(t[s < 1]) == pytest.approx(ORB.duration_full(), abs=3 * dt)
    assert np.ptp(t[s == 0]) == pytest.approx(ORB.duration_total(), abs=3 * dt)


def test_transit_is_not_an_eclipse():
    """At transit the planet is IN FRONT of the star: nothing of the planet is hidden."""
    t = ORB.t0 + np.linspace(-0.05, 0.05, 11)
    assert np.allclose(E.eclipse_shape(t, ORB.t0 + ORB.period / 2, ORB), 1.0)
