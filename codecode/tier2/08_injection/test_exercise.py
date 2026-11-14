import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _load import exercise  # noqa: E402
from rockyeye.s04_eclipse import Orbit, eclipse_shape  # noqa: E402
from rockyeye.s08_injection import inject_into_stamps as ref_inject  # noqa: E402

X = exercise(__file__)
ORB = Orbit(period=2.6162644, t0=2460452.8997, a_rs=17.05, inc=89.3, rp_rs=0.0318)


def test_inject_matches_reference():
    rng = np.random.default_rng(0)
    sci = rng.normal(100, 5, (50, 9, 9)).astype(np.float32)
    bg = rng.normal(10, 1, 50)
    shape = np.linspace(1, 0, 50)
    mine = X.inject_into_stamps(sci, bg, shape, 1e-3)
    ref = ref_inject({"sci": sci}, bg, shape, 1e-3)["sci"]
    assert np.allclose(mine, ref)
    assert np.allclose(mine[0], sci[0])                     # out of eclipse: untouched


def test_linear_depth_recovers_injected_depths():
    rng = np.random.default_rng(1)
    te = ORB.predicted_eclipse_time(2460864.9)
    t = te + np.arange(-0.11, 0.12, 40 / 86400)
    th = (t - t[0]) * 24
    B = np.column_stack([np.ones_like(t), (th - th.mean()) / np.ptp(th)])
    s = eclipse_shape(t, te, ORB) - 1.0
    found, errs = [], []
    for depth in (0.0, 100.0, 300.0):
        for k in range(20):
            flux = (1 + depth * 1e-6 * (s + 1)) * (1 + 2e-4 * B[:, 1]) + rng.normal(0, 250e-6, t.size)
            d, e = X.linear_depth(B, s, flux)
            found.append(d - depth)
            errs.append(e)
    found, errs = np.array(found), np.array(errs)
    assert abs(found.mean()) < 3 * errs.mean() / np.sqrt(len(found))      # unbiased
    assert np.std(found) == pytest.approx(errs.mean(), rel=0.35)           # honest error bars
