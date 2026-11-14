"""Synthetic stamps with a known eclipse: an end-to-end test bench for stages 03-10.

Not a detector simulator. It writes the same stamp files stage 02 produces
(81x81 px cut-outs per integration), built from a diffraction-like PSF, sky,
photon-like noise, a settling ramp, slow drift, pointing jitter, cosmic rays and
a few bad pixels, with an eclipse of known depth and time in the star's flux.
Noise and ramp sizes are matched to what we measured on a real calibrated
segment of the simulated LHS 1140 b data (about 650 ppm per 13.5 s integration,
a ramp of a few thousand ppm at the start).

Because the truth is known exactly, running the pipeline on these stamps
verifies photometry, systematics, fitting, injection and physics together.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.special import j1

from . import s01_fetch, s02_calibrate
from .common import log, mark_cached
from .s04_eclipse import Orbit, eclipse_shape


def airy_psf(size: int, x0: float, y0: float, first_null_px: float = 5.26, oversample: int = 3) -> np.ndarray:
    """Airy pattern normalised to unit total flux on the stamp, integrated over pixels by oversampling."""
    n = size * oversample
    c = (np.arange(n) + 0.5) / oversample - 0.5
    xx, yy = np.meshgrid(c - x0, c - y0)
    r = np.hypot(xx, yy) * 3.8317 / first_null_px
    r = np.where(r == 0, 1e-9, r)
    psf = (2 * j1(r) / r) ** 2
    psf = psf.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    return psf / psf.sum()


def make_visit(visit: str, k: int, syn: dict, orbit: Orbit, half: int, rng) -> dict:
    """One synthetic visit: stamp cube plus the truth that went into it."""
    n = syn["n_integrations"]
    cadence = syn["cadence_s"] / 86400.0
    t_ecl = orbit.t0 + orbit.period * (syn["first_epoch"] + k + 0.5) + syn["truth"]["dt_min"] / 1440.0
    # the eclipse is not centred in the window, differently for each visit
    start = t_ecl - (0.42 + 0.12 * rng.random()) * n * cadence
    t = start + (np.arange(n) + 0.5) * cadence
    th = (t - t[0]) * 24.0
    size = 2 * half + 1
    astro = 1.0 + syn["truth"]["depth_ppm"] * 1e-6 * eclipse_shape(t, t_ecl, orbit)
    amp = syn["ramp_amp"] * (0.6 + 0.8 * rng.random())
    tau = syn["ramp_tau_h"] * (0.6 + 0.8 * rng.random())
    ramp = 1.0 - amp * np.exp(-th / tau) + syn["slope_per_h"] * rng.normal() * (th - th.mean())
    # pointing: white jitter plus a slow random walk; flux depends weakly on x (pixel sensitivity)
    walk = np.cumsum(rng.normal(0, syn["jitter_px"] / 30, (n, 2)), axis=0)
    xy = rng.normal(0, syn["jitter_px"], (n, 2)) + walk - walk.mean(axis=0)
    x = half + 0.21 + xy[:, 0]
    y = half + 0.39 + xy[:, 1]
    sens = 1.0 + syn["x_sensitivity"] * (xy[:, 0] / syn["jitter_px"])
    star = syn["star_flux"] * astro * ramp * sens
    sky = syn["sky"] * (1.0 + 0.01 * np.exp(-th / 1.0))
    sci = np.empty((n, size, size), np.float32)
    psf0 = airy_psf(size, half + 0.21, half + 0.39)
    gx, gy = np.gradient(psf0, axis=1), np.gradient(psf0, axis=0)
    for i in range(n):                           # first-order shift of the PSF: accurate for milli-pixel jitter
        psf = psf0 - xy[i, 0] * gx - xy[i, 1] * gy
        sci[i] = star[i] * psf + sky[i]
    var = syn["noise_gain"] * np.clip(sci, 1e-3, None)
    sci = sci + rng.normal(0, 1, sci.shape).astype(np.float32) * np.sqrt(var)
    hits = rng.random(sci.shape) < syn["cosmic_ray_rate"]
    sci[hits] += rng.uniform(50, 2000, hits.sum()).astype(np.float32)
    bad = np.zeros(sci.shape, bool)
    for _ in range(syn["n_bad_pixels"]):
        by, bx = rng.integers(0, size, 2)
        bad[:, by, bx] = True
    sci[bad] = np.nan
    return {"time": t, "sci": sci, "err": np.sqrt(var).astype(np.float32), "bad": bad,
            "origin": np.array([100, 100]), "centre": np.array([100.0 + half, 100.0 + half]),
            "median_frame": np.nanmedian(sci[:50], axis=0).astype(np.float32),
            "unit": "MJy/sr", "pixar_sr": 2.8606e-13,
            "truth": {"t_ecl": float(t_ecl), "ramp_amp": float(amp), "ramp_tau_h": float(tau)}}


def write_stamps(cfg: dict, force: bool = False) -> dict:
    """Generate every visit and store it as stage-02 stamp files (split into segments)."""
    syn = cfg["source"]
    orbit = Orbit.from_config(cfg["system"])
    key = s02_calibrate.stage_key(cfg)
    truth = {}
    for k, visit in enumerate(cfg["visits"]):
        files = s01_fetch.list_files(cfg, [visit])
        if not force and all(s02_calibrate.is_cached(s02_calibrate.stamp_path(cfg, f), key) for f in files):
            continue
        rng = np.random.default_rng(syn["seed"] + 1000 * k)
        v = make_visit(visit, k, syn, orbit, cfg["stamp"]["half_size"], rng)
        truth[visit] = v.pop("truth")
        edges = np.linspace(0, len(v["time"]), len(files) + 1).astype(int)
        for rf, a, b in zip(files, edges[:-1], edges[1:]):
            out = s02_calibrate.stamp_path(cfg, rf)
            part = {kk: (vv[a:b] if kk in ("time", "sci", "err", "bad") else vv) for kk, vv in v.items()}
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **part, filename=rf.filename, visit=visit, segment=rf.segment)
            os.replace(tmp, out)
            mark_cached(out, key)
        log.info("synthetic %s: %d integrations, eclipse at BJD %.5f", visit, len(v["time"]), truth[visit]["t_ecl"])
    return truth
