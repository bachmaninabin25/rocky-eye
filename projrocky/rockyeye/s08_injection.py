"""Ingredient 08 — injection-recovery.

"Would we have seen an eclipse of depth D if it were there, and would we have
measured it correctly?" The only honest way to answer is to plant fake
eclipses in the data and run the same machinery on them.

Two levels:

* **Pixel level** (``inject_into_stamps``): the star's flux in the calibrated
  frames is dimmed by the eclipse model, then photometry and the fit are rerun.
  This exercises everything after stage 2 and is the validation gate on the
  simulated data.
* **Light-curve level** (``run_injection_grid``): hundreds of trials per depth,
  each with a different noise realisation made by cyclically shifting the real
  residuals ("prayer bead"), which preserves their correlated structure.
  Each trial is fitted with a fast linearised estimator.

The fast estimator exploits the fact that for depths << 1,

    flux = (1 + d*s(t)) * (B c)  ~=  B c + (d*c0) * s(t)

so the eclipse shape s(t) is just one more column of the linear design matrix,
and depth and its formal error drop out of one least-squares solve. Only the
ramp timescale and the eclipse time are scanned non-linearly.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from .s06_fit import EclipseFit, Visit


def inject_into_stamps(stamps: dict, bg: np.ndarray, shape: np.ndarray, depth: float) -> dict:
    """Return a copy of the stamps with an eclipse of fractional ``depth`` planted in the star.

    Only the star (frame minus sky) is scaled: star -> star * (1 + depth*(shape-1)),
    i.e. unchanged out of eclipse and dimmer by ``depth`` in eclipse.
    """
    out = dict(stamps)
    scale = (1.0 + depth * (shape - 1.0)).astype(np.float32)[:, None, None]
    sky = bg.astype(np.float32)[:, None, None]
    out["sci"] = sky + (stamps["sci"] - sky) * scale
    return out


def linear_depth(fit: EclipseFit, i: int, t_ecl: float, theta: np.ndarray, flux: np.ndarray):
    """Depth (ppm), formal error (ppm) and chi2 for visit i at fixed timing and ramp timescales."""
    v = fit.visits[i]
    B = v.model.basis(v.th, v.aux, theta)
    for tb in v.breaks:
        B = np.column_stack([B, (v.t > tb).astype(float)])
    s = fit.shape(v.t, t_ecl) - 1.0                 # 0 out of eclipse, -1 in eclipse
    if not np.any(s < 0):
        # The eclipse misses this visit entirely (e.g. one half of a split visit at this
        # trial timing): no depth information. Return infinite error so it gets zero weight.
        c, *_ = np.linalg.lstsq(B, flux, rcond=None)
        resid = flux - B @ c
        return 0.0, np.inf, float(resid @ resid)
    M = np.column_stack([B, s])
    q, r = np.linalg.qr(M)
    c = np.linalg.solve(r, q.T @ flux)
    resid = flux - M @ c
    chi2 = float(resid @ resid)
    sigma2 = chi2 / (len(flux) - M.shape[1])
    rinv = np.linalg.inv(r)
    cov = sigma2 * rinv @ rinv.T
    # baseline level at mid-eclipse normalises the amplitude into a depth
    base = float(np.median(B @ c[:-1]))
    return 1e6 * c[-1] / base, 1e6 * np.sqrt(cov[-1, -1]) / base, chi2


def fast_visit_fit(fit: EclipseFit, i: int, flux: np.ndarray, t_ecl: float, theta0: np.ndarray):
    """Optimise the visit's ramp timescales at fixed eclipse time; return (depth, err, chi2)."""
    v = fit.visits[i]
    if v.model.n_nonlinear == 0:
        return linear_depth(fit, i, t_ecl, theta0, flux)
    lo, hi = fit.log_tau_lim

    def obj(th):
        th = np.atleast_1d(th)
        if np.any(th < lo) or np.any(th > hi) or (len(th) == 2 and th[1] <= th[0]):
            return 1e30
        return linear_depth(fit, i, t_ecl, th, flux)[2]

    if v.model.n_nonlinear == 1:
        res = optimize.minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
        theta = np.atleast_1d(res.x)
        # the bounded search can miss the basin: also try the starting value
        if obj(theta0) < res.fun:
            theta = np.atleast_1d(theta0)
    else:
        theta = optimize.minimize(obj, theta0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-12}).x
    return linear_depth(fit, i, t_ecl, theta, flux)


def fast_joint_depth(fit: EclipseFit, fluxes: list[np.ndarray], theta0: list[np.ndarray],
                     dt_grid_min: np.ndarray, t_base: list[float] | None = None):
    """Joint depth over visits with a shared timing offset scanned on a grid.

    For each trial offset the per-visit depths are combined by inverse-variance
    weighting; the offset with the lowest total chi2 wins. Scanning the timing
    matters: letting the fit choose the time biases weak signals upward, and the
    injection test must include that effect because the real fit has it too.
    """
    t_base = t_base or fit.t_pred
    best = None
    for dt in np.atleast_1d(dt_grid_min):
        d, e, chi2 = zip(*[fast_visit_fit(fit, i, fluxes[i], t_base[i] + dt / 1440.0, theta0[i])
                           for i in range(len(fit.visits))])
        d, e = np.array(d), np.array(e)
        w = 1 / e ** 2
        if w.sum() == 0:                        # eclipse misses every visit at this offset
            continue
        depth, err = float(np.sum(w * d) / w.sum()), float(1 / np.sqrt(w.sum()))
        # -2 ln L (profiled over each visit's noise) plus the penalty for forcing one shared depth
        stat = float(np.sum([fit.visits[i].n * np.log(chi2[i] / fit.visits[i].n) for i in range(len(d))])
                     + np.sum(w * (d - depth) ** 2))
        if best is None or stat < best["stat"]:
            best = {"depth": depth, "err": err, "dt": float(dt), "stat": stat,
                    "per_visit": d.tolist(), "per_visit_err": e.tolist()}
    return best


def run_injection_grid(fit: EclipseFit, p_best: np.ndarray, depths_ppm, n_trials: int,
                       dt_grid_min: np.ndarray, detection_sigma: float = 3.0, seed: int = 7,
                       scan_timing: bool = True) -> dict:
    """Injection-recovery at light-curve level.

    For each trial: take the fitted systematics model, add cyclically shifted
    residuals (a new noise realisation with the same colour), multiply by an
    eclipse of the chosen depth at the true (fitted) time, and refit.
    """
    rng = np.random.default_rng(seed)
    parts = fit.chi2_parts(p_best)
    theta0 = [p_best[sl][1:] for sl in fit._slices]
    t_true = [fit.t_ecl(p_best, i) for i in range(len(fit.visits))]
    dt_true = (t_true[0] - fit.t_pred[0]) * 1440.0
    grid = np.atleast_1d(dt_grid_min) if scan_timing else np.array([dt_true])
    results = {}
    for depth in depths_ppm:
        rec, err, dts = [], [], []
        for _ in range(n_trials):
            fluxes = []
            for i, v in enumerate(fit.visits):
                shift = rng.integers(1, v.n - 1)
                sign = rng.choice([-1.0, 1.0])
                noise = sign * np.roll(parts[i]["resid"], shift)
                s = fit.shape(v.t, t_true[i])
                fluxes.append(parts[i]["systematics"] * (1 + depth * 1e-6 * s) + noise)
            best = fast_joint_depth(fit, fluxes, theta0, grid)
            rec.append(best["depth"])
            err.append(best["err"])
            dts.append(best["dt"])
        rec, err = np.array(rec), np.array(err)
        results[float(depth)] = {
            "recovered_mean": float(rec.mean()), "recovered_std": float(rec.std(ddof=1)),
            "bias": float(rec.mean() - depth), "bias_err": float(rec.std(ddof=1) / np.sqrt(n_trials)),
            "formal_err_median": float(np.median(err)),
            "error_ratio": float(rec.std(ddof=1) / np.median(err)),
            "detection_rate": float(np.mean(rec / rec.std(ddof=1) > detection_sigma)),
            "recovered": rec.tolist(), "dt": dts,
        }
    # sensitivity: empirical scatter of the null trials sets the detection threshold
    null = results.get(0.0)
    summary = {"dt_true_min": dt_true, "n_trials": n_trials}
    if null:
        s0 = null["recovered_std"]
        summary["null_scatter_ppm"] = s0
        summary["threshold_ppm"] = detection_sigma * s0
        for d, r in results.items():
            r["detection_rate"] = float(np.mean(np.array(r["recovered"]) > detection_sigma * s0))
    return {"by_depth": results, "summary": summary}
