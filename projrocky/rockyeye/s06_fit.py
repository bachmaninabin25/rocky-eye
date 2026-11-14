"""Ingredient 06 — likelihood and MCMC fit.

Model for one visit (see ingredients 04 and 05):

    flux(t) = [1 + depth * shape(t - t_ecl)] * [B(t; theta) @ c] + noise

    noise ~ Normal(0, sigma^2), independent per point, one sigma per visit.

The linear coefficients ``c`` are integrated out analytically with flat priors,
which leaves the *marginal* likelihood

    ln L = -(N-K)/2 ln(2 pi sigma^2) - chi2_min / (2 sigma^2) - 1/2 ln det(M^T M)

with M = shape-modulated design matrix and chi2_min the least-squares residual
sum. The sampler therefore explores only:

    shared:    depth_ppm, dt_min (eclipse-time offset from phase 0.5, in minutes)
    per visit: log_sigma, and the ramp timescales of that visit's model

A joint fit of several visits multiplies their likelihoods and shares the depth
(and, by default, the timing offset). A single-visit fit is the same code with
one visit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import optimize

from .common import log
from .s04_eclipse import Orbit, eclipse_shape
from .s05_systematics import SystematicsModel, solve_linear, standardise
from .s07_diagnostics import diagnose, integrated_autocorr_time, split_rhat

LOG2PI = np.log(2 * np.pi)


# -------------------------------------------------------------------- data ---

@dataclass
class Visit:
    """One eclipse observation, ready to fit."""

    name: str
    t: np.ndarray                 # BJD_TDB
    flux: np.ndarray              # normalised to a median of 1
    aux: dict                     # standardised decorrelation vectors: x, y, sx, sy, bg
    model: SystematicsModel
    breaks: tuple = ()            # times where the baseline may jump (split exposures)
    th: np.ndarray = field(init=False)

    def __post_init__(self):
        self.th = (self.t - self.t[0]) * 24.0

    @property
    def n(self) -> int:
        return len(self.t)

    def with_model(self, model: SystematicsModel | str) -> "Visit":
        m = SystematicsModel.parse(model) if isinstance(model, str) else model
        return Visit(self.name, self.t, self.flux, self.aux, m, self.breaks)


def bin_series(x: np.ndarray, n: int) -> np.ndarray:
    """Average consecutive groups of n points (the ragged tail is dropped)."""
    if n <= 1:
        return np.asarray(x, dtype=float)
    nb = len(x) // n
    return np.asarray(x[: nb * n], dtype=float).reshape(nb, n).mean(axis=1)


def make_visit(name: str, lc: dict, model: str, trim_start: int = 0, bin_size: int = 1,
               mask: np.ndarray | None = None, break_gap_s: float = 300.0) -> Visit:
    """Build a Visit from a light-curve dict (ingredient 03 output).

    Order matters: drop flagged points, cut the settling integrations, bin,
    normalise, then standardise the decorrelation vectors.
    """
    good = np.isfinite(lc["flux"]) & (lc.get("good", np.ones(len(lc["flux"]), bool)))
    if mask is not None:
        good &= mask
    good[:trim_start] = False
    t_full = np.asarray(lc["time"])[good]
    t = bin_series(t_full, bin_size)
    f = bin_series(np.asarray(lc["flux"])[good], bin_size)
    aux = {k: standardise(bin_series(np.asarray(lc[k])[good], bin_size))
           for k in ("x", "y", "sx", "sy", "bg") if k in lc}
    # a gap much longer than the cadence marks a new exposure -> allow a flux offset
    dt = np.diff(t) * 86400.0
    cadence = np.median(dt)
    breaks = tuple(0.5 * (t[i] + t[i + 1]) for i in np.where(dt > max(break_gap_s, 5 * cadence))[0])
    return Visit(name, t, f / np.median(f), aux, SystematicsModel.parse(model), breaks)


# ----------------------------------------------------------------- problem ---

class ShapeTable:
    """Eclipse shape tabulated against (t - t_ecl), so the sampler never calls batman.

    With fixed geometry the shape depends only on the time from mid-eclipse.
    We evaluate batman once on a fine grid and interpolate afterwards.
    """

    def __init__(self, orbit: Orbit, step_s: float = 1.0):
        half = 0.75 * orbit.duration_full() + 10.0 / 1440.0
        self.grid = np.arange(-half, half + step_s / 86400.0, step_s / 86400.0)
        self.shape = eclipse_shape(self.grid, 0.0, orbit)

    def __call__(self, t: np.ndarray, t_ecl: float) -> np.ndarray:
        return np.interp(t - t_ecl, self.grid, self.shape, left=1.0, right=1.0)


class EclipseFit:
    """Posterior for one or more visits sharing an eclipse depth."""

    def __init__(self, visits: list[Visit], orbit: Orbit, priors: dict, shared_time: bool = True):
        self.visits = visits
        self.orbit = orbit
        self.shared_time = shared_time or len(visits) == 1
        self.shape = ShapeTable(orbit)
        self.depth_lim = tuple(priors.get("depth_ppm", (-1000.0, 2000.0)))
        self.dt_lim = tuple(priors.get("dt_min", (-90.0, 90.0)))
        self.log_tau_lim = tuple(np.log(priors.get("tau_hours", (0.02, 6.0))))
        self.log_sigma_lim = (np.log(1e-6), np.log(1e-1))
        # predicted mid-eclipse (circular orbit) for each visit
        self.t_pred = [orbit.predicted_eclipse_time(0.5 * (v.t[0] + v.t[-1])) for v in visits]

        self.names = ["depth_ppm"]
        self.names += ["dt_min"] if self.shared_time else [f"dt_min:{v.name}" for v in visits]
        self._slices = []
        for v in visits:
            start = len(self.names)
            self.names.append(f"log_sigma:{v.name}")
            self.names += [f"{n}:{v.name}" for n in v.model.nonlinear_names]
            self._slices.append(slice(start, len(self.names)))
        self.ndim = len(self.names)

    # -- pieces ---------------------------------------------------------------
    def t_ecl(self, p: np.ndarray, i: int) -> float:
        dt = p[1] if self.shared_time else p[1 + i]
        return self.t_pred[i] + dt / 1440.0

    def design(self, p: np.ndarray, i: int) -> np.ndarray:
        """Full design matrix for visit i: eclipse-modulated systematics basis."""
        v = self.visits[i]
        theta = p[self._slices[i]][1:]
        B = v.model.basis(v.th, v.aux, theta)
        for tb in v.breaks:                      # flux offset after each exposure break
            B = np.column_stack([B, (v.t > tb).astype(float)])
        E = 1.0 + p[0] * 1e-6 * self.shape(v.t, self.t_ecl(p, i))
        return E[:, None] * B

    def n_linear(self, i: int) -> int:
        return self.visits[i].model.n_linear + len(self.visits[i].breaks)

    def log_prior(self, p: np.ndarray) -> float:
        if not self.depth_lim[0] <= p[0] <= self.depth_lim[1]:
            return -np.inf
        n_dt = 1 if self.shared_time else len(self.visits)
        if np.any(p[1:1 + n_dt] < self.dt_lim[0]) or np.any(p[1:1 + n_dt] > self.dt_lim[1]):
            return -np.inf
        for v, sl in zip(self.visits, self._slices):
            q = p[sl]
            if not self.log_sigma_lim[0] <= q[0] <= self.log_sigma_lim[1]:
                return -np.inf
            taus = q[1:]
            if np.any(taus < self.log_tau_lim[0]) or np.any(taus > self.log_tau_lim[1]):
                return -np.inf
            if len(taus) == 2 and taus[1] <= taus[0]:     # exp2: order the two timescales
                return -np.inf
        return 0.0

    def log_like(self, p: np.ndarray) -> float:
        total = 0.0
        for i, v in enumerate(self.visits):
            M = self.design(p, i)
            try:
                _, chi2, logdet = solve_linear(v.flux, M)
            except np.linalg.LinAlgError:
                return -np.inf
            log_sigma = p[self._slices[i]][0]
            dof = v.n - M.shape[1]
            total += (-dof * (log_sigma + 0.5 * LOG2PI)
                      - 0.5 * chi2 * np.exp(-2 * log_sigma) - 0.5 * logdet)
        return total if np.isfinite(total) else -np.inf

    def log_prob(self, p: np.ndarray) -> float:
        lp = self.log_prior(p)
        return lp + self.log_like(p) if np.isfinite(lp) else -np.inf

    __call__ = log_prob

    # -- point estimates ------------------------------------------------------
    def chi2_parts(self, p: np.ndarray):
        """Per-visit (chi2, coefficients, model, systematics-only model)."""
        out = []
        for i, v in enumerate(self.visits):
            M = self.design(p, i)
            c, chi2, _ = solve_linear(v.flux, M)
            full = M @ c
            E = 1.0 + p[0] * 1e-6 * self.shape(v.t, self.t_ecl(p, i))
            out.append({"chi2": chi2, "coeffs": c, "model": full, "systematics": full / E,
                        "astro": E, "resid": v.flux - full})
        return out

    def max_likelihood(self, depth_fixed: float | None = None, seed: int = 1) -> dict:
        """Global maximum of the (profile) likelihood; also returns the BIC.

        sigma is profiled analytically (sigma^2 = chi2/N per visit), so the
        optimiser only sees depth, timing and ramp timescales. With
        ``depth_fixed=0`` this gives the no-eclipse comparison model.
        """
        nl_idx = [0] + list(range(1, 1 + (1 if self.shared_time else len(self.visits))))
        for sl in self._slices:
            nl_idx += list(range(sl.start + 1, sl.stop))
        if depth_fixed is not None:
            nl_idx = [j for j in nl_idx if j != 0]
            if depth_fixed == 0:                      # timing is meaningless without an eclipse
                nl_idx = [j for j in nl_idx if not self.names[j].startswith("dt_min")]
        bounds = []
        for j in nl_idx:
            nm = self.names[j]
            bounds.append(self.depth_lim if nm == "depth_ppm" else
                          self.dt_lim if nm.startswith("dt_min") else self.log_tau_lim)

        def expand(x):
            p = np.zeros(self.ndim)
            p[0] = depth_fixed if depth_fixed is not None else 0.0
            # parameters not being optimised must still sit inside their prior: with no
            # eclipse the timing is irrelevant, but an out-of-prior value would make every
            # trial point invalid (this happened when the timing prior excluded 0)
            for j, nm in enumerate(self.names):
                if nm.startswith("dt_min"):
                    p[j] = 0.5 * (self.dt_lim[0] + self.dt_lim[1])
            p[nl_idx] = x
            for sl in self._slices:
                p[sl.start] = -7.0
            return p

        def objective(x):
            p = expand(x)
            if not np.isfinite(self.log_prior(p)):
                return 1e30
            return sum(v.n * np.log(part["chi2"] / v.n)
                       for v, part in zip(self.visits, self.chi2_parts(p)))

        if nl_idx:
            res = optimize.differential_evolution(objective, bounds, seed=seed, tol=1e-8,
                                                  popsize=20, maxiter=400, polish=True)
            p = expand(res.x)
        else:
            p = expand(np.array([]))
        parts = self.chi2_parts(p)
        n_tot, lnl = 0, 0.0
        for i, (v, part) in enumerate(zip(self.visits, parts)):
            s2 = part["chi2"] / v.n
            p[self._slices[i].start] = 0.5 * np.log(s2)
            lnl += -0.5 * v.n * (LOG2PI + np.log(s2) + 1.0)
            n_tot += v.n
        k = len(nl_idx) + len(self.visits) + sum(self.n_linear(i) for i in range(len(self.visits)))
        return {"p": p, "lnL": lnl, "k": k, "n": n_tot, "bic": k * np.log(n_tot) - 2 * lnl,
                "aic": 2 * k - 2 * lnl, "parts": parts,
                "sigma_ppm": [float(np.sqrt(pt["chi2"] / v.n) * 1e6) for v, pt in zip(self.visits, parts)]}

    # -- sampling -------------------------------------------------------------
    def run_mcmc(self, backend_path: str | Path, p0: np.ndarray, n_walkers: int = 64,
                 max_steps: int = 40000, check_every: int = 2000, tau_multiple: float = 50.0,
                 seed: int = 42, progress: bool = False, max_rhat: float = 1.01) -> dict:
        """Run emcee until converged or ``max_steps``; safe to kill and rerun.

        Every step is written to an HDF5 backend. On restart the sampler picks
        up from the last stored step, so an interrupted overnight run loses at
        most the step in progress.
        """
        import emcee

        backend_path = Path(backend_path)
        backend_path.parent.mkdir(parents=True, exist_ok=True)
        backend = emcee.backends.HDFBackend(str(backend_path))
        n_walkers = max(n_walkers, 2 * self.ndim + 2)
        fresh = True
        try:
            fresh = backend.iteration == 0 or backend.shape != (n_walkers, self.ndim)
        except (OSError, AttributeError, KeyError):
            fresh = True
        rng = np.random.default_rng(seed)
        if fresh:
            backend.reset(n_walkers, self.ndim)
            scale = np.array([5.0 if n.startswith("depth") else 1.0 if n.startswith("dt")
                              else 0.02 for n in self.names])
            start = p0 + scale * rng.standard_normal((n_walkers, self.ndim))
            for k in range(n_walkers):                 # make sure every walker starts inside the prior
                tries = 0
                while not np.isfinite(self.log_prior(start[k])) and tries < 200:
                    start[k] = p0 + 0.2 * scale * rng.standard_normal(self.ndim)
                    tries += 1
                if not np.isfinite(self.log_prior(start[k])):
                    start[k] = p0
        else:
            start = None
            log.info("resuming chain %s at step %d", backend_path.name, backend.iteration)
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        sampler = emcee.EnsembleSampler(n_walkers, self.ndim, self.log_prob, moves=moves, backend=backend)
        def check(old_tau):
            """Converged once the chain is long in units of tau and tau has stopped growing."""
            n = sampler.iteration
            burn = n // 3
            chain = sampler.get_chain(discard=burn)
            tau = integrated_autocorr_time(chain)
            rhat = float(np.nanmax(split_rhat(chain)))
            enough = np.all((n - burn) > tau_multiple * tau)
            # the estimate of tau creeps upward while a chain is too short; require the
            # slowest parameter's tau to have changed by under 10% since the last check
            stable = abs(old_tau - tau.max()) / tau.max() < 0.10
            # stop only on the same criteria the convergence report applies (s07)
            mixed = rhat <= max_rhat
            log.info("mcmc %s: step %d, max tau %.0f, (N-burn)/tau_max %.0f, max R-hat %.3f", backend_path.stem, n,
                     tau.max(), (n - burn) / tau.max(), rhat)
            return bool(enough and stable and mixed), tau.max()

        old_tau = np.inf
        state = start
        while sampler.iteration < max_steps:
            n_do = min(check_every, max_steps - sampler.iteration)
            state = sampler.run_mcmc(state, n_do, progress=progress)
            state = None                                # afterwards continue from the backend
            done, old_tau = check(old_tau)
            if done:
                break
        return self.summarise(sampler)

    def summarise(self, sampler) -> dict:
        n = sampler.iteration
        burn = n // 3
        chain = sampler.get_chain(discard=burn)
        diag = diagnose(chain, self.names, sampler.acceptance_fraction)
        thin = max(1, int(np.max(diag["tau"]) / 4))
        flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
        lnp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
        q = np.percentile(flat, [15.865, 50, 84.135], axis=0)
        summary = {nm: {"median": float(q[1, i]), "minus": float(q[1, i] - q[0, i]),
                        "plus": float(q[2, i] - q[1, i]), "mean": float(flat[:, i].mean()),
                        "std": float(flat[:, i].std(ddof=1))} for i, nm in enumerate(self.names)}
        return {"names": self.names, "summary": summary, "diagnostics": diag, "flat": flat,
                "lnp": lnp, "map": flat[np.argmax(lnp)], "n_steps": n, "burn": burn, "thin": thin}
