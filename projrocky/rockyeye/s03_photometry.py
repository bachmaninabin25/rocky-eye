"""Ingredient 03 — aperture photometry: stamps -> light curve.

For every integration we (1) repair bad pixels, (2) estimate the sky from a
ring around the star, (3) find the star's centre and width, and (4) add up the
sky-subtracted flux inside a circular aperture with exact pixel-overlap
weights. Done for a grid of aperture radii; the radius with the lowest
high-frequency scatter is kept (a criterion that cannot see the eclipse depth,
so choosing it does not bias the answer).

Written from scratch (no photutils) — photutils is only used in the tests as
an independent reference.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import median_filter


# --------------------------------------------------- exact circular aperture ---

def _quadrant_area(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Area of {0<=x'<=x, 0<=y'<=y} inside a circle of radius r at the origin (x, y >= 0)."""
    x = np.minimum(x, r)
    y = np.minimum(y, r)
    inside = x * x + y * y <= r * r
    xa = np.sqrt(np.maximum(r * r - y * y, 0.0))      # where the arc meets the top edge

    def seg(u):                                        # integral of sqrt(r^2 - u^2) du
        return 0.5 * (u * np.sqrt(np.maximum(r * r - u * u, 0.0)) + r * r * np.arcsin(np.clip(u / r, -1, 1)))

    arc = xa * y + seg(x) - seg(xa)
    return np.where(inside, x * y, arc)


def _signed_area(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    return np.sign(x) * np.sign(y) * _quadrant_area(np.abs(x), np.abs(y), r)


def aperture_weights(shape: tuple[int, int], x0: float, y0: float, r: float) -> np.ndarray:
    """Exact fraction of every pixel that lies inside a circle of radius r at (x0, y0).

    Pixel (i, j) covers [j-0.5, j+0.5] x [i-0.5, i+0.5], so integer coordinates
    are pixel centres. The overlap of a circle with a rectangle follows from the
    area function A(x, y) by inclusion-exclusion over the four corners.
    """
    ny, nx = shape
    xl = np.arange(nx) - 0.5 - x0
    yl = np.arange(ny) - 0.5 - y0
    X0, Y0 = np.meshgrid(xl, yl)
    X1, Y1 = X0 + 1.0, Y0 + 1.0
    return (_signed_area(X1, Y1, r) - _signed_area(X0, Y1, r)
            - _signed_area(X1, Y0, r) + _signed_area(X0, Y0, r))


# ------------------------------------------------------------- cleaning ---

def clean_cube(sci: np.ndarray, bad: np.ndarray, sigma: float = 5.0, window: int = 11):
    """Replace bad pixels and cosmic-ray hits with each pixel's own running median in time.

    Returns (cleaned cube, number of temporal outliers replaced per integration).
    A pixel that is bad in every integration has no history; it is filled from
    its spatial neighbours instead.
    """
    cube = np.array(sci, dtype=np.float32)
    cube[bad] = np.nan
    with warnings.catch_warnings():                      # all-NaN (dead) pixels are handled below
        warnings.simplefilter("ignore", RuntimeWarning)
        pixel_median = np.nanmedian(cube, axis=0)
        filled = np.where(np.isfinite(cube), cube, pixel_median[None])
        smooth = median_filter(np.nan_to_num(filled), size=(window, 1, 1), mode="nearest")
        resid = filled - smooth
        mad = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0)
    mad = np.where(mad > 0, mad, np.inf)
    outlier = np.abs(resid) > sigma * mad[None]
    replace = outlier | ~np.isfinite(cube)
    out = np.where(replace, smooth, cube)
    dead = ~np.isfinite(pixel_median)                    # never-valid pixels
    for iy, ix in zip(*np.where(dead)):
        nb = out[:, max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].copy()
        nb[:, min(iy, 1), min(ix, 1)] = np.nan
        out[:, iy, ix] = np.nanmean(nb.reshape(len(out), -1), axis=1)
    return out, outlier.sum(axis=(1, 2))


def annulus_background(cube: np.ndarray, x0: float, y0: float, r_in: float, r_out: float,
                       clip: float = 3.0):
    """Sigma-clipped mean sky level per integration in a ring r_in < r < r_out.

    Returns (background per pixel, its standard error) for each integration.
    """
    ny, nx = cube.shape[1:]
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.hypot(xx - x0, yy - y0)
    ring = cube[:, (rr >= r_in) & (rr < r_out)].astype(float)
    for _ in range(2):
        med = np.nanmedian(ring, axis=1, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(ring - med), axis=1, keepdims=True)
        ring = np.where(np.abs(ring - med) > clip * mad, np.nan, ring)
    n = np.sum(np.isfinite(ring), axis=1)
    return np.nanmean(ring, axis=1), np.nanstd(ring, axis=1) / np.sqrt(np.maximum(n, 1))


def centroid_and_width(cube: np.ndarray, bg: np.ndarray, x0: float, y0: float, box: int = 5):
    """Flux-weighted centre (x, y) and RMS widths (sx, sy) in a fixed window around (x0, y0).

    The window is the same for all integrations so that it cannot hop between
    pixels and add its own noise.
    """
    cx, cy = int(round(x0)), int(round(y0))
    sub = cube[:, cy - box:cy + box + 1, cx - box:cx + box + 1].astype(float) - bg[:, None, None]
    ys = np.arange(cy - box, cy + box + 1, dtype=float)
    xs = np.arange(cx - box, cx + box + 1, dtype=float)
    tot = sub.sum(axis=(1, 2))
    px = sub.sum(axis=1)            # profile along x
    py = sub.sum(axis=2)            # profile along y
    x = (px * xs).sum(axis=1) / tot
    y = (py * ys).sum(axis=1) / tot
    sx = np.sqrt(np.abs((px * (xs[None] - x[:, None]) ** 2).sum(axis=1) / tot))
    sy = np.sqrt(np.abs((py * (ys[None] - y[:, None]) ** 2).sum(axis=1) / tot))
    return x, y, sx, sy


# ------------------------------------------------------------- photometry ---

def aperture_sum(cube: np.ndarray, var: np.ndarray | None, bg: np.ndarray,
                 x: np.ndarray, y: np.ndarray, r: float):
    """Sky-subtracted flux in a circular aperture, centred per integration."""
    n = len(cube)
    flux = np.empty(n)
    ferr = np.full(n, np.nan)
    shape = cube.shape[1:]
    for i in range(n):
        w = aperture_weights(shape, x[i], y[i], r)
        flux[i] = np.sum(w * cube[i]) - bg[i] * w.sum()
        if var is not None:
            ferr[i] = np.sqrt(np.sum(w * w * var[i]))
    return flux, ferr


def highpass_scatter(flux: np.ndarray, window: int = 101):
    """Robust point-to-point scatter (fraction of the flux) after removing a running median.

    Returns (sigma, boolean mask of points within 5 sigma).
    """
    f = flux / np.nanmedian(flux)
    resid = f - median_filter(f, size=window, mode="nearest")
    sig = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    return float(sig), np.abs(resid) < 5 * sig


def extract_light_curve(stamps: dict, phot: dict) -> dict:
    """Full photometry of one visit. ``stamps`` comes from ingredient 02."""
    sci, bad = stamps["sci"], stamps["bad"]
    cube, n_out = clean_cube(sci, bad, phot.get("outlier_sigma", 5.0))
    var = np.where(np.isfinite(stamps["err"]), stamps["err"].astype(np.float32) ** 2, 0.0)
    ox, oy = stamps["origin"]
    x0, y0 = stamps["centre"][0] - ox, stamps["centre"][1] - oy     # stamp coordinates
    r_in, r_out = phot["annulus"]
    box = phot.get("centroid_box", 5)
    # two passes: refine the reference position, then measure everything about it
    bg, bg_err = annulus_background(cube, x0, y0, r_in, r_out)
    x, y, sx, sy = centroid_and_width(cube, bg, x0, y0, box)
    x0, y0 = float(np.nanmedian(x)), float(np.nanmedian(y))
    bg, bg_err = annulus_background(cube, x0, y0, r_in, r_out)
    x, y, sx, sy = centroid_and_width(cube, bg, x0, y0, box)
    if phot.get("aperture_centre", "frame") == "fixed":
        xa, ya = np.full_like(x, x0), np.full_like(y, y0)
    else:
        xa, ya = x, y

    radii = list(phot["aperture_radii"])
    fluxes, errs, scatters = [], [], []
    for r in radii:
        f, e = aperture_sum(cube, var, bg, xa, ya, r)
        fluxes.append(f)
        errs.append(e)
        scatters.append(highpass_scatter(f)[0])
    best = int(np.argmin(scatters)) if phot.get("radius") is None else radii.index(phot["radius"])
    flux = fluxes[best]
    sig, good = highpass_scatter(flux)
    med = np.nanmedian(flux)
    return {
        "time": stamps["time"], "flux": flux / med, "flux_err": errs[best] / med,
        "flux_dn_s": flux, "bg": bg, "bg_err": bg_err,
        "x": x + ox, "y": y + oy, "sx": sx, "sy": sy, "good": good,
        "n_outlier_pixels": n_out,
        "radii": np.array(radii), "scatter_ppm": np.array(scatters) * 1e6,
        "flux_all": np.array(fluxes) / np.nanmedian(fluxes, axis=1)[:, None],
        "best_radius": radii[best], "scatter_best_ppm": sig * 1e6,
        "median_err_ppm": float(np.nanmedian(errs[best] / med) * 1e6),
        "segment_starts": stamps.get("segment_starts", np.array([0])),
    }


def absolute_flux_mjy(stamps: dict, cal: dict) -> dict:
    """Star's flux density in mJy from flux-calibrated frames (stage 2 with the photom step on).

    Uses the calibration-program recipe: a fixed aperture and sky ring for which
    the aperture correction to "infinite" radius is tabulated (CRDS apcorr file).
        F [mJy] = sum(aperture, sky-subtracted) [MJy/sr] * pixel area [sr] * 1e9 * apcorr
    Returns the median over integrations, its scatter, and the adopted systematic error.
    """
    if not str(stamps.get("unit", "")).startswith("MJy"):
        raise ValueError(f"frames are in '{stamps.get('unit')}', not MJy/sr: rerun stage 2 with photom enabled")
    cube, _ = clean_cube(stamps["sci"], stamps["bad"])
    ox, oy = stamps["origin"]
    x0, y0 = stamps["centre"][0] - ox, stamps["centre"][1] - oy
    bg, _ = annulus_background(cube, x0, y0, cal["skyin"], cal["skyout"])
    x, y, _, _ = centroid_and_width(cube, bg, x0, y0, 5)
    x0, y0 = float(np.nanmedian(x)), float(np.nanmedian(y))
    bg, _ = annulus_background(cube, x0, y0, cal["skyin"], cal["skyout"])
    step = max(1, len(cube) // 400)                        # a few hundred integrations are plenty
    idx = np.arange(0, len(cube), step)
    f, _ = aperture_sum(cube[idx], None, bg[idx], np.full(len(idx), x0), np.full(len(idx), y0), cal["radius"])
    mjy = f * stamps["pixar_sr"] * 1e9 * cal["apcorr"]
    med = float(np.median(mjy))
    return {"flux_mjy": med, "scatter_mjy": float(np.std(mjy)),
            "systematic_mjy": med * cal.get("systematic_frac", 0.03), "n": int(len(idx))}


def save_light_curve(path, lc: dict) -> None:
    np.savez_compressed(path, **lc)


def load_light_curve(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
