"""Ingredient 09 (YOUR version): from an eclipse depth to a verdict. Bare rock or atmosphere?

Read first: learntopic/day1 (thermal emission, energy balance) and learncode/09_physics.md.

The chain of reasoning:
  1. depth = (Rp/Rs)^2 * <B(T_planet)> / <I_star>        (band-averaged Planck functions)
     -> invert for the planet's dayside BRIGHTNESS TEMPERATURE.
  2. energy balance predicts the dayside temperature (Cowan & Agol 2011):
     T_day = T_star / sqrt(a/Rs) * (1 - A_B)^(1/4) * (2/3 - 5/12 * eps)^(1/4)
       eps = 0: no heat redistribution -> bare rock (hottest possible dayside)
       eps = 1: heat spread round the planet -> thick atmosphere
  3. compare: how probable is the measured depth under each scenario? (Bayes factors)

Simplification vs projrocky: the band is a top-hat from 13.5 to 16.6 um instead
of the real F1500W throughput curve. Same physics, slightly different numbers.

Check yourself:   python run.py check 09
Target size:      ~100 lines of your own code
"""
from __future__ import annotations

import numpy as np
from scipy import stats

H = 6.62607015e-34      # Planck constant, J s
C = 2.99792458e8        # speed of light, m/s
KB = 1.380649e-23       # Boltzmann constant, J/K
RSUN = 6.957e8          # m
PC = 3.0856775814913673e16  # m


def planck_nu(wl_um, T):
    """Planck spectral radiance B_nu (W m^-2 Hz^-1 sr^-1) at wavelength wl_um (MICRONS), temperature T.

    TODO:  nu = c / lambda  (lambda in metres!),  B_nu = 2 h nu^3 / c^2 / (exp(h nu / k T) - 1)
    Use np.expm1 for the bracket (accurate when h nu << k T).
    """
    raise NotImplementedError


class Band:
    """Photon-weighted averages over a top-hat band, plus the inverse (brightness temperature).

    MIRI counts photons, so a spectrum F_nu is averaged with weight 1/nu:
        <F> = integral F_nu dnu/nu  /  integral dnu/nu
    """

    def __init__(self, lo_um: float = 13.5, hi_um: float = 16.6, n: int = 400):
        """TODO:
          - wavelength grid self.wl (n points, lo..hi), its frequencies self.nu SORTED
            ascending (np.trapezoid needs increasing x), weights 1/nu
          - self.pivot_um = sqrt( int wl dwl / int dwl/wl )   (a single "effective" wavelength)
          - tabulate <B_nu>(T) for T on np.geomspace(30, 20000, 2000) ONCE, so that
            planck() and brightness_temperature() are just interpolations
        """
        raise NotImplementedError

    def planck(self, T):
        """Band-averaged B_nu(T).  TODO: interpolate the table in log-log space."""
        raise NotImplementedError

    def brightness_temperature(self, intensity):
        """Inverse of planck(): the temperature whose band-averaged B equals `intensity`.  TODO"""
        raise NotImplementedError


def stellar_intensity(band: Band, teff):
    """Star's band-averaged surface intensity, treating it as a blackbody.  (given)"""
    return band.planck(teff)


def dayside_temperature(teff, a_rs, albedo=0.0, eps=0.0):
    """Energy-balance dayside temperature (formula in the module docstring).  TODO: one line."""
    raise NotImplementedError


def depth_from_temperature(band: Band, T_planet, rp_rs, i_star):
    """Eclipse depth (a FRACTION, not ppm) of a blackbody planet.  TODO: one line."""
    raise NotImplementedError


def temperature_from_depth(band: Band, depth, rp_rs, i_star):
    """Brightness temperature for a measured depth (fraction). NaN where depth <= 0.

    TODO: planet intensity = depth * i_star / rp_rs^2, then band.brightness_temperature.
    A negative depth has no temperature: return NaN there (np.where, no Python loop).
    """
    raise NotImplementedError


def compare_scenarios(depth_samples_ppm, predicted: dict, reference: str) -> dict:
    """Evidence for each scenario + Bayes factor vs `reference`.

    depth_samples_ppm: MCMC samples of the measured depth (flat prior -> posterior ~ likelihood)
    predicted: {name: array of predicted depths in ppm} (spread = uncertain star/orbit)

    TODO:
      like = stats.gaussian_kde(depth_samples_ppm)          # L(depth), keeps skewness
      for each scenario:
         evidence Z = mean of like(predicted[name])          # "average likelihood over the prediction"
         tension_sigma = (median(data) - mean(pred)) / hypot(std(data), std(pred))
      bayes_factor_vs_ref = Z / Z[reference]
      return {name: {"evidence", "tension_sigma", "bayes_factor_vs_ref"}}
    """
    raise NotImplementedError


def interpret_bayes_factor(b: float) -> str:
    """Words for a Bayes factor (Jeffreys-style scale on |ln B|).

    TODO:  |ln B| < 1 -> "no real preference";  < 2.5 "weak";  < 5 "moderate";  else "strong".
    Except for "no real preference", append " evidence for" if B > 1 else " evidence against".
    B <= 0 or not finite -> "excluded".
    """
    raise NotImplementedError
