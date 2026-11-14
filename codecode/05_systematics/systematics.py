"""Ingredient 05 (YOUR version): models of the detector's noise ("systematics").

Read first: learntopic/day3 (why MIRI light curves drift and ramp) and
learncode/05_systematics.md.

The idea that makes the whole fit fast: every systematics model is a LINEAR
combination of basis columns,

    S(t) = c_0 * 1 + c_1 * t + c_2 * exp(-t / tau) + c_3 * x(t) + ...

The c's enter linearly, so for any fixed tau they can be solved exactly with
least squares (no sampler needed). Only tau is non-linear.

Model names:  "<baseline>+<extras>"
    baseline: const | linear | poly2 | exp | exp2
    extras:   x, y (star position), w (PSF width -> columns sx, sy), bg (background)
    e.g. "exp+xy" = constant, slope, exponential ramp, x and y decorrelation.

Check yourself:   python run.py check 05
Target size:      ~80 lines of your own code
"""
from __future__ import annotations

import numpy as np

# name -> (number of linear columns, names of the non-linear parameters)   (given)
BASELINES = {
    "const": (1, ()),
    "linear": (2, ()),
    "poly2": (3, ()),
    "exp": (3, ("log_tau",)),
    "exp2": (4, ("log_tau", "log_tau2")),
}
# extra -> the aux columns it adds   (given)
AUX_KEYS = {"x": ("x",), "y": ("y",), "w": ("sx", "sy"), "bg": ("bg",)}
EXTRA_ORDER = ["x", "y", "w", "bg"]


def parse_model(name: str) -> tuple[str, tuple[str, ...]]:
    """"exp+xybg" -> ("exp", ("x", "y", "bg")).

    TODO:
      - split on the first "+": baseline before, extras string after (may be empty)
      - unknown baseline -> raise ValueError
      - read the extras string: look for "bg" FIRST (otherwise its letters could be
        misread), then "x", "y", "w"; anything left over -> raise ValueError
      - return the extras sorted in EXTRA_ORDER
    """
    raise NotImplementedError


def aux_columns(name: str) -> tuple[str, ...]:
    """Aux column names the model uses, e.g. "exp+xw" -> ("x", "sx", "sy").  TODO: 2 lines."""
    raise NotImplementedError


def n_linear(name: str) -> int:
    """Number of linear coefficients = baseline columns + aux columns.  TODO: 2 lines."""
    raise NotImplementedError


def nonlinear_names(name: str) -> tuple[str, ...]:
    """Names of the non-linear parameters, e.g. ("log_tau",).  TODO: 1 line."""
    raise NotImplementedError


def basis(name: str, th: np.ndarray, aux: dict, theta=()) -> np.ndarray:
    """Design matrix B, shape (n_points, n_linear(name)). Column order matters!

    th:    time in HOURS since the first point of the visit (ramps start at 0)
    aux:   dict of standardised vectors, e.g. {"x": ..., "y": ...}
    theta: non-linear parameters in the order of nonlinear_names(name)

    TODO, columns in this order:
      1. ones
      2. (not for "const") centred, unit-range time: (th - mean) / ptp(th)
      3. (poly2) that time squared
      4. (exp, exp2) exp(-th / exp(theta[0]))        <- theta stores log(tau)
      5. (exp2)      exp(-th / exp(theta[1]))
      6. one column per aux_columns(name), in that order
    Stack with np.column_stack.
    """
    raise NotImplementedError


def standardise(v) -> np.ndarray:
    """Zero-centre and scale a decorrelation vector ROBUSTLY, then clip to [-8, 8].

    TODO: subtract the median, divide by 1.4826 * MAD (median absolute
    deviation; the 1.4826 makes it equal the std for Gaussian data). If the MAD
    is 0 or not finite, fall back to the std (or 1.0). Use nan-aware numpy.
    Why robust: one cosmic-ray hit should not rescale the whole vector.
    """
    raise NotImplementedError


def solve_linear(y: np.ndarray, M: np.ndarray):
    """Least squares y ~ M c.  Return (c, chi2, logdet).

    chi2   = |y - M c|^2
    logdet = log det(M^T M)      (needed by the marginal likelihood in 06)

    TODO: use a QR decomposition (np.linalg.qr), NOT the normal equations
    (M^T M is badly conditioned). With M = QR:
        c = solve(R, Q^T y)
        logdet = 2 * sum(log|diag(R)|)
    """
    raise NotImplementedError


def binned_rms(resid: np.ndarray, max_bin: int | None = None):
    """RMS of the residuals after averaging in bins of n points (the "Allan" curve).

    Returns (sizes, rms, expected):
      sizes    = unique ints from np.geomspace(1, max_bin, 30), max_bin default len//8
      rms[j]   = std (ddof=1) of the bin means for bin size sizes[j]; NaN if < 2 bins
      expected = rms[0] / sqrt(sizes) * sqrt(nb / (nb - 1)),  nb = len // sizes
                 (pure white noise; the sqrt factor corrects for few bins)
    TODO
    """
    raise NotImplementedError


def red_noise_beta(resid: np.ndarray, cadence_s: float, timescale_s: float) -> float:
    """Measured / white-noise RMS around a timescale; >= 1, and >1 means red noise.  (given)"""
    sizes, rms, expected = binned_rms(resid)
    target = max(1.0, timescale_s / cadence_s)
    sel = (sizes >= target / 2) & (sizes <= target * 2) & np.isfinite(rms)
    if not sel.any():
        return 1.0
    return float(max(1.0, np.median(rms[sel] / expected[sel])))
