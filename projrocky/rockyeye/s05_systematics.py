"""Ingredient 05 — systematics models.

MIRI time series are never flat: the detector "settles" for the first hour or
so (a ramp), the flux drifts slowly, and the measured flux follows the star's
position on the pixels, the PSF width and the background level.

Every systematics model here has the form

    S(t) = sum_k  c_k * B_k(t; theta)

a *linear* combination of basis vectors B_k. The coefficients c_k (baseline
level, slope, ramp amplitude, decorrelation weights) enter linearly; only a few
parameters theta (ramp timescales) enter non-linearly. That split is the key
design decision of the fit: linear coefficients are solved exactly by least
squares inside the likelihood (ingredient 06), so the sampler only has to
explore theta, the eclipse parameters and the noise level.

A model is named by a string such as ``"exp+xy"``:
    baseline:  const | linear | poly2 | poly3 | exp | exp2      (before the +)
    extras:    x, y (centroid), w (PSF width), bg (background)  (after the +;
               ``xy`` = x and y, ``xyw``, ``xywbg`` likewise)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINES = {
    # name: (number of linear columns, names of non-linear parameters)
    "const": (1, ()),
    "linear": (2, ()),
    "poly2": (3, ()),
    "poly3": (4, ()),
    "exp": (3, ("log_tau",)),
    "exp2": (4, ("log_tau", "log_tau2")),
}

AUX_KEYS = {"x": ("x",), "y": ("y",), "w": ("sx", "sy"), "bg": ("bg",)}


@dataclass(frozen=True)
class SystematicsModel:
    baseline: str
    extras: tuple[str, ...] = ()

    @classmethod
    def parse(cls, name: str) -> "SystematicsModel":
        base, _, extra = name.partition("+")
        if base not in BASELINES:
            raise ValueError(f"unknown baseline '{base}' (choose from {sorted(BASELINES)})")
        extras, rest = [], extra
        for key in ("bg", "x", "y", "w"):          # 'bg' first so its letters are not misread
            if key in rest:
                extras.append(key)
                rest = rest.replace(key, "", 1)
        if rest:
            raise ValueError(f"cannot parse decorrelation terms '{extra}' in '{name}'")
        order = ["x", "y", "w", "bg"]
        return cls(base, tuple(sorted(extras, key=order.index)))

    @property
    def name(self) -> str:
        return self.baseline + ("+" + "".join(self.extras) if self.extras else "")

    @property
    def nonlinear_names(self) -> tuple[str, ...]:
        return BASELINES[self.baseline][1]

    @property
    def aux_columns(self) -> tuple[str, ...]:
        return tuple(c for e in self.extras for c in AUX_KEYS[e])

    @property
    def n_linear(self) -> int:
        return BASELINES[self.baseline][0] + len(self.aux_columns)

    @property
    def n_nonlinear(self) -> int:
        return len(self.nonlinear_names)

    def linear_names(self) -> list[str]:
        base = {"const": ["c0"], "linear": ["c0", "slope"], "poly2": ["c0", "slope", "quad"],
                "poly3": ["c0", "slope", "quad", "cubic"], "exp": ["c0", "slope", "amp"],
                "exp2": ["c0", "slope", "amp", "amp2"]}[self.baseline]
        return base + [f"c_{c}" for c in self.aux_columns]

    def basis(self, th: np.ndarray, aux: dict, theta=()) -> np.ndarray:
        """Design matrix B with shape (n_points, n_linear).

        ``th`` is time in hours since the first integration *of the visit*
        (so ramps always start at 0). ``aux`` holds standardised decorrelation
        vectors (zero mean, unit variance). ``theta`` holds the non-linear
        parameters in the order of ``nonlinear_names``.
        """
        tc = (th - th.mean()) / max(np.ptp(th), 1e-12)   # centred, unit-range time
        cols = [np.ones_like(th)]
        if self.baseline != "const":
            cols.append(tc)
        if self.baseline in ("poly2", "poly3"):
            cols.append(tc ** 2)
        if self.baseline == "poly3":
            cols.append(tc ** 3)
        if self.baseline in ("exp", "exp2"):
            cols.append(np.exp(-th / np.exp(theta[0])))
        if self.baseline == "exp2":
            cols.append(np.exp(-th / np.exp(theta[1])))
        for c in self.aux_columns:
            cols.append(aux[c])
        return np.column_stack(cols)


def standardise(v: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance copy of a decorrelation vector (robust to outliers)."""
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(v) or 1.0
    return np.clip((v - med) / mad, -8, 8)


def solve_linear(y: np.ndarray, M: np.ndarray):
    """Least-squares coefficients for y ~ M c.

    Returns (c, chi2, logdet) where chi2 = |y - M c|^2 and
    logdet = log det(M^T M), both needed by the marginal likelihood.
    """
    q, r = np.linalg.qr(M)
    c = np.linalg.solve(r, q.T @ y)
    resid = y - M @ c
    logdet = 2.0 * np.sum(np.log(np.abs(np.diag(r))))
    return c, float(resid @ resid), float(logdet)


# --------------------------------------------------------- noise diagnostics ---

def binned_rms(resid: np.ndarray, max_bin: int | None = None):
    """RMS of the residuals after averaging in bins of n points ("Allan" plot).

    White noise falls as 1/sqrt(n). A curve that flattens above that line means
    correlated (red) noise is left in the residuals.
    """
    resid = np.asarray(resid, dtype=float)
    n = len(resid)
    max_bin = max_bin or n // 8
    sizes = np.unique(np.round(np.geomspace(1, max(max_bin, 2), 30)).astype(int))
    rms = []
    for s in sizes:
        nb = n // s
        rms.append(np.std(resid[: nb * s].reshape(nb, s).mean(axis=1), ddof=1) if nb > 1 else np.nan)
    rms = np.array(rms)
    expected = rms[0] / np.sqrt(sizes) * np.sqrt((n // sizes) / np.maximum(n // sizes - 1, 1))
    return sizes, rms, expected


def red_noise_beta(resid: np.ndarray, cadence_s: float, timescale_s: float) -> float:
    """Ratio of measured to white-noise-expected RMS at a given timescale (>= 1 means red noise).

    Evaluated around ``timescale_s`` (we use the eclipse ingress/egress to full
    duration scale), following the "beta" factor of Winn et al. (2008).
    """
    sizes, rms, expected = binned_rms(resid)
    target = max(1.0, timescale_s / cadence_s)
    sel = (sizes >= target / 2) & (sizes <= target * 2) & np.isfinite(rms)
    if not sel.any():
        return 1.0
    return float(max(1.0, np.median(rms[sel] / expected[sel])))
