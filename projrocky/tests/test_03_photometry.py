"""Ingredient 03: exact aperture weights and flux within 0.1% of an independent reference."""
import numpy as np
import pytest

from rockyeye import s03_photometry as ph


def test_weights_sum_to_circle_area():
    for r in (0.3, 1.0, 4.0, 7.77):
        w = ph.aperture_weights((41, 41), 20.3, 19.6, r)
        assert w.sum() == pytest.approx(np.pi * r * r, rel=1e-12)
        assert w.min() >= -1e-12 and w.max() <= 1 + 1e-12


def test_weights_match_photutils():
    photutils = pytest.importorskip("photutils.aperture")
    rng = np.random.default_rng(1)
    for _ in range(30):
        x0, y0 = rng.uniform(15, 25, 2)
        r = rng.uniform(1, 12)
        ref = photutils.CircularAperture((x0, y0), r).to_mask(method="exact").to_image((41, 41))
        assert np.abs(ph.aperture_weights((41, 41), x0, y0, r) - ref).max() < 1e-9


def _fake_stamps(n=400, half=40, flux=5e4, sky=200.0, seed=2, drift=0.02):
    rng = np.random.default_rng(seed)
    size = 2 * half + 1
    yy, xx = np.mgrid[:size, :size]
    x = half + 0.3 + drift * np.sin(np.arange(n) / 30)
    y = half - 0.2 + drift * np.cos(np.arange(n) / 45)
    sig = 1.9
    cube = np.empty((n, size, size), np.float32)
    for i in range(n):
        psf = np.exp(-0.5 * ((xx - x[i]) ** 2 + (yy - y[i]) ** 2) / sig ** 2) / (2 * np.pi * sig ** 2)
        cube[i] = flux * psf + sky
    noise = rng.normal(0, 1.0, cube.shape).astype(np.float32)
    return {"time": 2460000 + np.arange(n) * 10 / 86400, "sci": cube + noise, "err": np.ones_like(cube),
            "bad": np.zeros(cube.shape, bool), "origin": np.array([10, 20]), "centre": np.array([10 + half, 20 + half]),
            "segment_starts": np.array([0])}, x, y, flux, sky


PHOT = {"aperture_radii": [4.0, 6.0, 10.0], "annulus": [16.0, 30.0], "centroid_box": 5, "outlier_sigma": 5.0}


def test_flux_within_0p1_percent_of_truth_and_photutils():
    stamps, x, y, flux, sky = _fake_stamps()
    lc = ph.extract_light_curve(stamps, dict(PHOT, radius=10.0))
    assert np.median(lc["flux_dn_s"]) == pytest.approx(flux, rel=1e-3)     # r=10 px holds >99.99% of a sigma=1.9 PSF
    photutils = pytest.importorskip("photutils.aperture")
    i = 123
    ap = photutils.CircularAperture((lc["x"][i] - 10, lc["y"][i] - 20), 10.0)
    ref = photutils.aperture_photometry(stamps["sci"][i] - lc["bg"][i], ap, method="exact")["aperture_sum"][0]
    assert lc["flux_dn_s"][i] == pytest.approx(ref, rel=1e-3)


def test_centroid_and_background_recovered():
    stamps, x, y, flux, sky = _fake_stamps()
    lc = ph.extract_light_curve(stamps, PHOT)
    assert np.median(lc["bg"]) == pytest.approx(sky, abs=0.05)
    # the fixed-window centroid tracks the true motion (slope ~1, tiny scatter)
    assert np.corrcoef(lc["x"] - 10, x)[0, 1] > 0.98
    assert np.std((lc["x"] - 10) - x) < 0.01


def test_cosmic_rays_and_bad_pixels_are_repaired():
    stamps, *_ = _fake_stamps()
    clean = ph.extract_light_curve(stamps, dict(PHOT, radius=6.0))["flux_dn_s"]
    dirty = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in stamps.items()}
    dirty["sci"][50, 40, 41] += 5000.0            # cosmic ray inside the aperture
    dirty["bad"][:, 42, 39] = True                # a permanently dead pixel inside the aperture
    dirty["sci"][:, 42, 39] = np.nan
    out = ph.extract_light_curve(dirty, dict(PHOT, radius=6.0))["flux_dn_s"]
    assert abs(out[50] - clean[50]) / clean[50] < 2e-3
    assert np.all(np.isfinite(out))
