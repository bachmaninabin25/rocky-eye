"""Shared test setup for codecode/. Synthetic data only: no network, no JWST files.

Every test asks for an ingredient through ``deps.get("NN")``, which returns YOUR
module if NN is in ROCKY_MINE, otherwise the reference. ``python run.py check 04``
sets ROCKY_MINE=04 for you.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _deps as deps  # noqa: E402
from rockyeye.s04_eclipse import Orbit as RefOrbit, eclipse_shape as batman_shape  # noqa: E402

# GJ 3929 b-like geometry (the real numbers live in projrocky/configs/gj3929b.yaml)
GJ = dict(period=2.6162644, t0=2460452.8997, a_rs=17.05, inc=89.3, rp_rs=0.0318)
PRIORS = {"depth_ppm": [-500, 1000], "dt_min": [-60, 60], "tau_hours": [0.02, 6]}


@pytest.fixture(scope="session")
def ref_orbit():
    return RefOrbit(**GJ)


def synthetic_light_curve(depth_ppm=150.0, dt_min=-20.0, sigma=250e-6, tau_h=0.4, amp=8e-4,
                          slope=2e-4, cx=5e-5, cadence_s=40.0, t_mid=2460864.9, seed=0):
    """A light curve with a KNOWN eclipse, an exponential ramp, a slope and centroid sensitivity.

    The truth is made with batman (independent of anything you write), so a fit
    that recovers ~150 ppm and ~-20 min is genuinely right.
    """
    orbit = RefOrbit(**GJ)
    rng = np.random.default_rng(seed)
    te = orbit.predicted_eclipse_time(t_mid)
    t = te + np.arange(-0.11, 0.12, cadence_s / 86400.0)
    th = (t - t[0]) * 24
    x = rng.normal(size=t.size)
    astro = 1 + depth_ppm * 1e-6 * batman_shape(t, te + dt_min / 1440.0, orbit)
    syst = 1 + slope * (th - 3) / 6 + amp * np.exp(-th / tau_h) + cx * x
    flux = astro * syst + rng.normal(0, sigma, t.size)
    return {"time": t, "flux": flux, "x": x, "y": rng.normal(size=t.size)}


@pytest.fixture(scope="session")
def three_light_curves():
    return [synthetic_light_curve(t_mid=2460864.9 + 23.5 * k, seed=k) for k in range(3)]
