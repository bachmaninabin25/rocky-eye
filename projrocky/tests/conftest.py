"""Shared synthetic data for the tests. Nothing here needs the network or real JWST files."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockyeye.s04_eclipse import Orbit, eclipse_shape  # noqa: E402
from rockyeye.s06_fit import make_visit  # noqa: E402

GJ = Orbit(period=2.6162644, t0=2460452.8997, a_rs=17.05, inc=89.3, rp_rs=0.0318)


@pytest.fixture(scope="session")
def orbit():
    return GJ


def synthetic_light_curve(orbit, t_mid, depth_ppm=150.0, dt_min=-20.0, sigma=250e-6, tau_h=0.4,
                          amp=8e-4, slope=2e-4, cx=5e-5, cadence_s=40.0, seed=0):
    """A light curve with a known eclipse, an exponential ramp, a slope and centroid sensitivity."""
    rng = np.random.default_rng(seed)
    te = orbit.predicted_eclipse_time(t_mid)
    t = te + np.arange(-0.11, 0.12, cadence_s / 86400.0)
    th = (t - t[0]) * 24
    x = rng.normal(size=t.size)
    astro = 1 + depth_ppm * 1e-6 * eclipse_shape(t, te + dt_min / 1440.0, orbit)
    syst = 1 + slope * (th - 3) / 6 + amp * np.exp(-th / tau_h) + cx * x
    flux = astro * syst + rng.normal(0, sigma, t.size)
    return {"time": t, "flux": flux, "x": x, "y": rng.normal(size=t.size), "sx": rng.normal(size=t.size),
            "sy": rng.normal(size=t.size), "bg": rng.normal(size=t.size)}


@pytest.fixture(scope="session")
def three_visits(orbit):
    return [make_visit(f"v{k}", synthetic_light_curve(orbit, 2460864.9 + 23.5 * k, seed=k), "exp+x")
            for k in range(3)]


PRIORS = {"depth_ppm": [-500, 1000], "dt_min": [-60, 60], "tau_hours": [0.02, 6]}
