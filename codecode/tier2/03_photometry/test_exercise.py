import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _load import exercise  # noqa: E402
from rockyeye.s03_photometry import centroid_and_width as ref_cw  # noqa: E402

X = exercise(__file__)


def gaussian_cube(n=20, size=41, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(float)
    xs = 20.3 + 0.05 * rng.normal(size=n)
    ys = 19.6 + 0.05 * rng.normal(size=n)
    cube = np.array([1000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2)) + 10 for x, y in zip(xs, ys)])
    return cube + rng.normal(0, 0.5, cube.shape), np.full(n, 10.0), xs, ys


def test_finds_the_star():
    cube, bg, xs, ys = gaussian_cube()
    x, y, sx, sy = X.centroid_and_width(cube, bg, 20, 20)
    assert np.allclose(x, xs, atol=0.02) and np.allclose(y, ys, atol=0.02)
    assert np.allclose(sx, 1.5, atol=0.1) and np.allclose(sy, 1.5, atol=0.1)


def test_matches_reference():
    cube, bg, _, _ = gaussian_cube(seed=1)
    for a, b in zip(X.centroid_and_width(cube, bg, 20.4, 19.7, box=4), ref_cw(cube, bg, 20.4, 19.7, box=4)):
        assert np.allclose(a, b)
