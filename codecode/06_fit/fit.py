"""Ingredient 06 (YOUR version): the likelihood and the MCMC fit. The heart of the project.

Read first: learntopic/day2 (Bayes, likelihoods, MCMC) and learncode/06_fit.md.

Model for one visit:

    flux(t) = E(t) * [B(t; theta) @ c] + noise,      E(t) = 1 + depth * shape(t - t_ecl)
    noise   ~ Normal(0, sigma^2), independent, one sigma per visit

The linear coefficients c are INTEGRATED OUT analytically (flat priors), which
leaves the marginal log-likelihood per visit:

    ln L = -(N - K) * (ln sigma + 0.5 ln 2pi) - chi2_min / (2 sigma^2) - 0.5 ln det(M^T M)

with M = E[:, None] * B (the eclipse-modulated design matrix, K columns) and
chi2_min the least-squares residual. So the sampler only explores:

    p = [depth_ppm, dt_min,  log_sigma_v1, <ramp params of v1>,  log_sigma_v2, ...]
         shared by all visits   ^------------- per visit ------------^

Uses YOUR or the reference ingredient 04 (eclipse_shape) and 05 (basis,
solve_linear), whichever ROCKY_MINE selects.

Check yourself:   python run.py check 06        (the MCMC test takes ~2 minutes)
Target size:      ~150 lines of your own code
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _deps  # noqa: E402

ECL = _deps.get("04")    # ECL.eclipse_shape(t, t_ecl, orbit)
SYS = _deps.get("05")    # SYS.basis, SYS.solve_linear, SYS.standardise, SYS.nonlinear_names

LOG2PI = np.log(2 * np.pi)


# ----------------------------------------------------------------- data (given) ---

@dataclass
class Visit:
    """One eclipse observation, ready to fit."""

    name: str
    t: np.ndarray          # BJD_TDB, days
    flux: np.ndarray       # normalised to median 1
    aux: dict              # standardised decorrelation vectors
    model: str             # systematics model name, e.g. "exp+x"
    th: np.ndarray = field(init=False)

    def __post_init__(self):
        self.th = (self.t - self.t[0]) * 24.0       # hours since the first point

    @property
    def n(self) -> int:
        return len(self.t)


def make_visit(name: str, lc: dict, model: str) -> Visit:
    """Light-curve dict (time, flux, optional x, y, sx, sy, bg) -> Visit.  (given)"""
    t = np.asarray(lc["time"], float)
    f = np.asarray(lc["flux"], float)
    aux = {k: SYS.standardise(lc[k]) for k in ("x", "y", "sx", "sy", "bg") if k in lc}
    return Visit(name, t, f / np.median(f), aux, model)


# ---------------------------------------------------------------------- the fit ---

class EclipseFit:
    """Posterior for one or more visits sharing one eclipse depth and one timing offset."""

    def __init__(self, visits: list[Visit], orbit, priors: dict):
        # (given) bookkeeping: prior ranges and the parameter layout
        self.visits = visits
        self.orbit = orbit
        self.depth_lim = tuple(priors.get("depth_ppm", (-1000.0, 2000.0)))
        self.dt_lim = tuple(priors.get("dt_min", (-90.0, 90.0)))
        self.log_tau_lim = tuple(np.log(priors.get("tau_hours", (0.02, 6.0))))
        self.log_sigma_lim = (np.log(1e-6), np.log(1e-1))
        self.t_pred = [orbit.predicted_eclipse_time(0.5 * (v.t[0] + v.t[-1])) for v in visits]
        self.names = ["depth_ppm", "dt_min"]
        self.slices = []                       # where each visit's block lives in p
        for v in visits:
            start = len(self.names)
            self.names.append(f"log_sigma:{v.name}")
            self.names += [f"{n}:{v.name}" for n in SYS.nonlinear_names(v.model)]
            self.slices.append(slice(start, len(self.names)))
        self.ndim = len(self.names)

    # -- pieces -----------------------------------------------------------------
    def t_ecl(self, p: np.ndarray, i: int) -> float:
        """Mid-eclipse time of visit i: predicted time + p[1] MINUTES (convert to days!).  TODO"""
        raise NotImplementedError

    def design(self, p: np.ndarray, i: int) -> np.ndarray:
        """M for visit i: the systematics basis multiplied row-wise by E(t).

        TODO:
          theta = the ramp params of visit i  (its block p[self.slices[i]] minus log_sigma)
          B = SYS.basis(v.model, v.th, v.aux, theta)
          E = 1 + depth_ppm * 1e-6 * ECL.eclipse_shape(v.t, self.t_ecl(p, i), self.orbit)
          return E[:, None] * B
        """
        raise NotImplementedError

    def log_prior(self, p: np.ndarray) -> float:
        """0 inside the box, -inf outside.

        TODO: depth in depth_lim; dt in dt_lim; for every visit: log_sigma in
        log_sigma_lim, each log_tau in log_tau_lim, and for two-timescale models
        (exp2) require log_tau2 > log_tau (otherwise the two ramps can swap
        places and the posterior has two identical copies).
        """
        raise NotImplementedError

    def log_like(self, p: np.ndarray) -> float:
        """Sum over visits of the marginal log-likelihood in the module docstring.

        TODO: for each visit: M = design, (c, chi2, logdet) = SYS.solve_linear(flux, M),
        K = M.shape[1]. Catch np.linalg.LinAlgError -> -inf. Return -inf if not finite.
        """
        raise NotImplementedError

    def log_prob(self, p: np.ndarray) -> float:
        """Log posterior = log prior + log likelihood (skip the likelihood if the prior is -inf).  (given)"""
        lp = self.log_prior(p)
        return lp + self.log_like(p) if np.isfinite(lp) else -np.inf

    # -- best fit + BIC -------------------------------------------------------------
    def max_likelihood(self, depth_fixed: float | None = None, seed: int = 1) -> dict:
        """Global best fit. Returns {"p", "lnL", "k", "n", "bic", "sigma_ppm"}.

        sigma is PROFILED: for fixed everything-else the best sigma^2 is chi2/N,
        and plugging it in turns -2 lnL into  sum_v  N_v * ln(chi2_v / N_v) + const.

        TODO:
          1. free parameters = depth, dt and all ramp params (NOT log_sigma).
             depth_fixed=0 means "no eclipse": drop depth AND dt (timing is
             meaningless without an eclipse; park dt at the middle of its prior).
          2. minimise  sum_v N_v ln(chi2_v / N_v)  with
             scipy.optimize.differential_evolution(obj, bounds, seed=seed, tol=1e-8,
             popsize=20, maxiter=400, polish=True). Return 1e30 outside the prior.
          3. at the best point set each log_sigma = 0.5 ln(chi2_v / N_v) and
             lnL = sum_v -0.5 N_v (ln 2pi + ln(chi2_v / N_v) + 1)
          4. k = (#free params) + (#visits, one sigma each) + sum of n_linear(model)
             (the linear c's count as parameters too!)
             bic = k ln(N_total) - 2 lnL
          5. sigma_ppm = [sqrt(chi2_v / N_v) * 1e6 for each visit]
        """
        raise NotImplementedError

    # -- MCMC -------------------------------------------------------------------------
    def run_mcmc(self, p0: np.ndarray, n_walkers: int = 32, n_steps: int = 3000, seed: int = 42) -> dict:
        """Sample the posterior with emcee. Returns {"depth_median", "depth_std", "flat", "names", "n_steps"}.

        TODO:
          1. n_walkers = max(n_walkers, 2*ndim + 2)
          2. start walkers in a small ball around p0: scale 5 for depth, 1 for dt,
             0.02 for everything else; re-draw any walker that starts outside the prior
          3. emcee.EnsembleSampler(n_walkers, ndim, self.log_prob,
                moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
             and run n_steps
          4. throw away the first third (burn-in): get_chain(discard=n_steps//3, flat=True)
          5. depth_median = median of column 0, depth_std = std (ddof=1)
        """
        raise NotImplementedError
