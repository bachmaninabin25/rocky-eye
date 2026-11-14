"""Reference for ingredient 06, mapped onto the lean API of 06_fit/fit.py.

projrocky's EclipseFit has more features (exposure breaks, free timing per
visit, a resumable HDF5 chain). This adapter exposes only what the lean
version has, so the same tests run against both.
"""
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rockyeye import s06_fit as _f
from rockyeye.s05_systematics import SystematicsModel, standardise

LOG2PI = _f.LOG2PI


@dataclass
class Visit:
    name: str
    t: np.ndarray
    flux: np.ndarray
    aux: dict
    model: str
    th: np.ndarray = field(init=False)

    def __post_init__(self):
        self.th = (self.t - self.t[0]) * 24.0

    @property
    def n(self):
        return len(self.t)


def make_visit(name, lc, model):
    t = np.asarray(lc["time"], float)
    f = np.asarray(lc["flux"], float)
    aux = {k: standardise(lc[k]) for k in ("x", "y", "sx", "sy", "bg") if k in lc}
    return Visit(name, t, f / np.median(f), aux, model)


def _to_ref(v):
    return _f.Visit(v.name, v.t, v.flux, v.aux, SystematicsModel.parse(v.model))


class EclipseFit:
    def __init__(self, visits, orbit, priors):
        self.visits = visits
        self._fit = _f.EclipseFit([_to_ref(v) for v in visits], orbit, priors, shared_time=True)
        self.names = self._fit.names
        self.ndim = self._fit.ndim
        self.t_pred = self._fit.t_pred
        self.slices = self._fit._slices

    def t_ecl(self, p, i):
        return self._fit.t_ecl(p, i)

    def design(self, p, i):
        return self._fit.design(p, i)

    def log_prior(self, p):
        return self._fit.log_prior(p)

    def log_like(self, p):
        return self._fit.log_like(p)

    def log_prob(self, p):
        return self._fit.log_prob(p)

    def max_likelihood(self, depth_fixed=None, seed=1):
        r = self._fit.max_likelihood(depth_fixed=depth_fixed, seed=seed)
        return {k: r[k] for k in ("p", "lnL", "k", "n", "bic", "sigma_ppm")}

    def run_mcmc(self, p0, n_walkers=32, n_steps=3000, seed=42):
        with tempfile.TemporaryDirectory() as d:
            r = self._fit.run_mcmc(Path(d) / "chain.h5", p0, n_walkers=n_walkers, max_steps=n_steps,
                                   check_every=min(1000, n_steps), seed=seed)
        s = r["summary"]["depth_ppm"]
        return {"depth_median": s["median"], "depth_std": s["std"], "flat": r["flat"],
                "names": r["names"], "n_steps": r["n_steps"]}
