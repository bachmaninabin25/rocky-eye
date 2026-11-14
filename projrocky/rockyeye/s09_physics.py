"""Ingredient 09 — from eclipse depth to a verdict: bare rock or atmosphere?

Chain of reasoning:

1. The eclipse depth is the planet/star flux ratio in the F1500W band:
       depth = (Rp/Rs)^2 * <B(T_planet)> / <I_star>
   where <.> is a photon-weighted average over the bandpass. Solving for
   T_planet gives the dayside *brightness temperature*.
2. Energy balance predicts the dayside temperature for a given Bond albedo A_B
   and heat-redistribution efficiency eps (Cowan & Agol 2011):
       T_day = T_star / sqrt(a/Rs) * (1 - A_B)^(1/4) * (2/3 - 5/12 eps)^(1/4)
   eps = 0: no redistribution (bare rock; the hottest possible dayside).
   eps = 1: heat spread over the whole planet (thick atmosphere).
3. Compare: how probable is the measured depth under each scenario?

The star's own 15 um surface brightness matters as much as the depth. Three
options are supported: a blackbody at T_eff, a surface brightness derived from a
measured flux density (needs R_star and distance), or a quoted model flux.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from scipy import stats

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
RSUN = 6.957e8
PC = 3.0856775814913673e16

_FILTER = Path(__file__).parent / "refdata" / "JWST_MIRI.F1500W.dat"


def planck_nu(wl_um: np.ndarray, T: float | np.ndarray) -> np.ndarray:
    """Planck spectral radiance B_nu in W m^-2 Hz^-1 sr^-1 at wavelength wl_um (microns)."""
    nu = C / (np.asarray(wl_um) * 1e-6)
    x = H * nu / (KB * np.asarray(T, dtype=float))
    return 2 * H * nu ** 3 / C ** 2 / np.expm1(x)


def load_bandpass(path: Path = _FILTER):
    """F1500W throughput (SVO Filter Profile Service): wavelength in microns, response."""
    if Path(path).exists():
        wl, resp = np.loadtxt(path, unpack=True)
        return wl * 1e-4, resp
    wl = np.linspace(13.5, 16.6, 200)          # fallback: top-hat
    return wl, np.ones_like(wl)


class Band:
    """Photon-weighted band averages.

    MIRI counts photo-electrons, so a spectrum F_nu contributes in proportion
    to F_nu / (h nu) * response. The band average of F_nu is therefore weighted
    by response / nu, i.e. by response / lambda in d(lambda)... written out:
        <F_nu> = int F_nu R dnu/nu / int R dnu/nu
    """

    def __init__(self, wl_um=None, resp=None):
        if wl_um is None:
            wl_um, resp = load_bandpass()
        self.wl = np.asarray(wl_um, dtype=float)
        nu = C / (self.wl * 1e-6)
        order = np.argsort(nu)
        self._nu, self._wl_sorted = nu[order], self.wl[order]
        self._w = np.asarray(resp, dtype=float)[order] / self._nu
        self._norm = np.trapezoid(self._w, self._nu)
        self.pivot_um = float(np.sqrt(np.trapezoid(resp * self.wl, self.wl) / np.trapezoid(resp / self.wl, self.wl)))
        # tabulate <B_nu>(T) once: brightness temperatures are then simple interpolations
        self._T = np.geomspace(30.0, 20000.0, 2000)
        self._B = np.array([self._avg(planck_nu(self._wl_sorted, T)) for T in self._T])

    def _avg(self, f_nu_sorted: np.ndarray) -> float:
        return float(np.trapezoid(f_nu_sorted * self._w, self._nu) / self._norm)

    def planck(self, T):
        """Band-averaged B_nu(T)."""
        return np.exp(np.interp(np.log(np.asarray(T, dtype=float)), np.log(self._T), np.log(self._B)))

    def brightness_temperature(self, intensity):
        """Temperature of the blackbody with this band-averaged intensity."""
        return np.exp(np.interp(np.log(np.asarray(intensity, dtype=float)), np.log(self._B), np.log(self._T)))


# -------------------------------------------------------------------- star ---

def stellar_intensity(band: Band, mode: str, teff=None, flux_mjy=None, rstar_rsun=None, dist_pc=None):
    """Band-averaged stellar surface intensity <I_nu> (W m^-2 Hz^-1 sr^-1).

    mode="blackbody": <B_nu(T_eff)>.
    mode="flux":      from an observed flux density F_nu = pi * I * (R_star/d)^2.
    Arguments may be arrays (Monte-Carlo samples).
    """
    if mode == "blackbody":
        return band.planck(teff)
    if mode == "flux":
        omega = (np.asarray(rstar_rsun) * RSUN / (np.asarray(dist_pc) * PC)) ** 2
        return np.asarray(flux_mjy) * 1e-29 / (np.pi * omega)
    raise ValueError(mode)


# ------------------------------------------------------------------ planet ---

def dayside_temperature(teff, a_rs, albedo=0.0, eps=0.0):
    """Equilibrium dayside temperature for Bond albedo and redistribution efficiency eps."""
    return np.asarray(teff) / np.sqrt(a_rs) * (1 - np.asarray(albedo)) ** 0.25 * (2 / 3 - 5 / 12 * np.asarray(eps)) ** 0.25


def depth_from_temperature(band: Band, T_planet, rp_rs, i_star):
    """Eclipse depth (fraction) of a blackbody planet at T_planet."""
    return np.asarray(rp_rs) ** 2 * band.planck(T_planet) / i_star


def temperature_from_depth(band: Band, depth, rp_rs, i_star):
    """Dayside brightness temperature for a measured depth (NaN where depth <= 0)."""
    inten = np.asarray(depth, dtype=float) * i_star / np.asarray(rp_rs) ** 2
    return np.where(inten > 0, band.brightness_temperature(np.where(inten > 0, inten, 1.0)), np.nan)


def draw_system(system: dict, n: int, rng) -> dict:
    """Monte-Carlo samples of the system parameters, using quoted 1-sigma errors where given."""
    def g(key, default_err=0.0):
        v = system[key]
        e = system.get(key + "_err", default_err)
        return np.full(n, float(v)) if not e else rng.normal(v, e, n)

    out = {"teff": g("teff"), "a_rs": g("a_rs"), "rp_rs": g("rp_rs")}
    for k in ("rstar_rsun", "dist_pc", "flux_mjy"):
        if k in system:
            out[k] = g(k)
    return out


def scenario_depths(band: Band, sys_draws: dict, i_star, scenarios: dict) -> dict:
    """Predicted depth samples (ppm) for each named scenario {name: {albedo, eps}}."""
    out = {}
    for name, sc in scenarios.items():
        if "depth_ppm" in sc:                                   # a literature forward model
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            out[name] = rng.normal(sc["depth_ppm"], sc.get("depth_err_ppm", 0.0), len(sys_draws["teff"]))
            continue
        T = dayside_temperature(sys_draws["teff"], sys_draws["a_rs"], sc.get("albedo", 0.0), sc.get("eps", 0.0))
        out[name] = 1e6 * depth_from_temperature(band, T, sys_draws["rp_rs"], i_star)
    return out


# -------------------------------------------------------- model comparison ---

def depth_likelihood(depth_samples_ppm: np.ndarray):
    """Likelihood of the data as a function of depth, from the posterior samples.

    The fit used a flat prior on depth, so posterior density is proportional to
    likelihood. A Gaussian KDE keeps any skewness of the posterior.
    """
    return stats.gaussian_kde(depth_samples_ppm)


def compare_scenarios(depth_samples_ppm: np.ndarray, predicted: dict, reference: str) -> dict:
    """Evidence for each scenario and its Bayes factor relative to ``reference``.

    A scenario with no free parameters predicts a depth distribution (its width
    comes from the uncertain stellar/orbital inputs). Its evidence is the
    likelihood averaged over that prediction:  Z = E_model[ L(depth) ].
    Also returns the simple "n-sigma" tension between data and prediction.
    """
    like = depth_likelihood(depth_samples_ppm)
    d_med, d_std = np.median(depth_samples_ppm), np.std(depth_samples_ppm, ddof=1)
    out = {}
    for name, pred in predicted.items():
        z = float(np.mean(like(pred[:4000])))
        out[name] = {"depth_ppm": float(np.mean(pred)), "depth_err_ppm": float(np.std(pred)),
                     "evidence": z,
                     "tension_sigma": float((d_med - np.mean(pred)) / np.hypot(d_std, np.std(pred)))}
    z_ref = out[reference]["evidence"]
    for name in out:
        out[name]["bayes_factor_vs_ref"] = out[name]["evidence"] / z_ref if z_ref > 0 else np.nan
        out[name]["ln_B_vs_ref"] = float(np.log(out[name]["evidence"] / z_ref)) if z_ref > 0 and out[name]["evidence"] > 0 else -np.inf
    return out


def interpret_bayes_factor(b: float) -> str:
    """Jeffreys-style words for a Bayes factor B (model vs reference)."""
    if not np.isfinite(b) or b <= 0:
        return "excluded"
    lb = abs(np.log(b))
    word = ("no real preference" if lb < 1 else "weak" if lb < 2.5 else
            "moderate" if lb < 5 else "strong")
    if lb < 1:
        return word
    return f"{word} evidence {'for' if b > 1 else 'against'}"


def infer_redistribution(band: Band, depth_samples_ppm, sys_draws: dict, i_star, albedo=0.0, n_grid=201):
    """Posterior of the redistribution efficiency eps in [0, 1] at fixed albedo (flat prior).

    Returns (eps grid, normalised posterior, 95% upper limit).
    """
    like = depth_likelihood(depth_samples_ppm)
    eps = np.linspace(0, 1, n_grid)
    post = np.empty(n_grid)
    m = min(len(sys_draws["teff"]), 400)
    for j, e in enumerate(eps):
        T = dayside_temperature(sys_draws["teff"][:m], sys_draws["a_rs"][:m], albedo, e)
        i_s = i_star[:m] if np.ndim(i_star) else i_star
        post[j] = np.mean(like(1e6 * depth_from_temperature(band, T, sys_draws["rp_rs"][:m], i_s)))
    post /= np.trapezoid(post, eps)
    cdf = np.cumsum(post) * (eps[1] - eps[0])
    return eps, post, float(np.interp(0.95, cdf / cdf[-1], eps))


def summarise_temperature(band: Band, depth_samples_ppm, sys_draws: dict, i_star, rng) -> dict:
    """Brightness temperature, its ratio R to the bare-rock maximum, and the albedo limit."""
    n = len(sys_draws["teff"])
    d = rng.choice(depth_samples_ppm, n) * 1e-6
    Tb = temperature_from_depth(band, d, sys_draws["rp_rs"], i_star)
    Tmax = dayside_temperature(sys_draws["teff"], sys_draws["a_rs"], 0.0, 0.0)
    ok = np.isfinite(Tb)
    R = Tb[ok] / Tmax[ok]
    q = lambda a: [float(v) for v in np.percentile(a, [15.865, 50, 84.135])]
    # effective albedo of an airless body: R^4 = 1 - A  ->  A = 1 - R^4
    A = 1 - R ** 4
    return {"T_b": q(Tb[ok]), "T_max": q(Tmax), "R": q(R), "frac_positive_depth": float(ok.mean()),
            "albedo_eff": q(A), "albedo_eff_upper95": float(np.percentile(A, 95)),
            "T_fullredist": q(dayside_temperature(sys_draws["teff"], sys_draws["a_rs"], 0.0, 1.0))}
