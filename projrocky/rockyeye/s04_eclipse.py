"""Ingredient 04 — secondary-eclipse model.

A thin wrapper around ``batman`` plus an independent analytic model used to
check it. The convention everywhere in RockyEye:

    flux(t) / F_star = 1 + depth * shape(t)

where ``shape`` is 1 when the whole planet is visible and 0 when it is fully
behind the star. ``depth`` is therefore the planet-to-star flux ratio, and the
*shape* does not depend on the depth, so one shape can be reused for any depth
(including the negative depths the sampler is allowed to explore).
"""
from __future__ import annotations

from dataclasses import dataclass

import batman
import numpy as np


@dataclass(frozen=True)
class Orbit:
    """Geometry needed for an eclipse light curve (circular orbit for the shape)."""

    period: float   # days
    t0: float       # transit mid-time, same time system as the data (BJD_TDB)
    a_rs: float     # semi-major axis / stellar radius
    inc: float      # inclination, degrees
    rp_rs: float    # planet radius / stellar radius

    @classmethod
    def from_config(cls, system: dict) -> "Orbit":
        return cls(period=system["period"], t0=system["t0"], a_rs=system["a_rs"],
                   inc=system["inc"], rp_rs=system["rp_rs"])

    def predicted_eclipse_time(self, t_ref: float) -> float:
        """Mid-eclipse time (circular orbit) nearest to ``t_ref``: t0 + P*(n + 1/2)."""
        n = np.round((t_ref - self.t0) / self.period - 0.5)
        return self.t0 + self.period * (n + 0.5)

    @property
    def impact_parameter(self) -> float:
        return self.a_rs * np.cos(np.radians(self.inc))

    def duration_full(self) -> float:
        """First-to-fourth-contact duration T14 in days (circular orbit)."""
        b = self.impact_parameter
        arg = np.sqrt((1 + self.rp_rs) ** 2 - b ** 2) / (self.a_rs * np.sin(np.radians(self.inc)))
        return self.period / np.pi * np.arcsin(arg)

    def duration_total(self) -> float:
        """Second-to-third-contact duration T23 in days (planet fully hidden)."""
        b = self.impact_parameter
        arg = np.sqrt((1 - self.rp_rs) ** 2 - b ** 2) / (self.a_rs * np.sin(np.radians(self.inc)))
        return self.period / np.pi * np.arcsin(arg)


def eclipse_shape(t: np.ndarray, t_ecl: float, orbit: Orbit) -> np.ndarray:
    """Visible fraction of the planet's disk at times ``t`` (1 out of eclipse, 0 in eclipse).

    Uses batman's secondary-eclipse mode with fp=1, for which batman returns
    ``1 + visible_fraction``.
    """
    p = batman.TransitParams()
    p.t0 = t_ecl - 0.5 * orbit.period   # unused for the shape, kept self-consistent
    p.t_secondary = t_ecl
    p.per = orbit.period
    p.rp = orbit.rp_rs
    p.a = orbit.a_rs
    p.inc = orbit.inc
    p.ecc = 0.0
    p.w = 90.0
    p.u = []
    p.limb_dark = "uniform"
    p.fp = 1.0
    m = batman.TransitModel(p, np.ascontiguousarray(t, dtype=float), transittype="secondary")
    return m.light_curve(p) - 1.0


def eclipse_flux(t: np.ndarray, depth: float, t_ecl: float, orbit: Orbit) -> np.ndarray:
    """Relative flux ``1 + depth * shape``; equals 1 mid-eclipse and ``1 + depth`` outside."""
    return 1.0 + depth * eclipse_shape(t, t_ecl, orbit)


# ---------------------------------------------------------------- analytic ---

def _circle_overlap_area(r1: float, r2: float, d: np.ndarray) -> np.ndarray:
    """Area of intersection of two circles with radii r1 >= r2 and centre distance d."""
    d = np.abs(np.asarray(d, dtype=float))
    area = np.zeros_like(d)
    inside = d <= r1 - r2
    area[inside] = np.pi * r2 ** 2
    partial = (~inside) & (d < r1 + r2)
    dp = d[partial]
    a1 = r1 ** 2 * np.arccos((dp ** 2 + r1 ** 2 - r2 ** 2) / (2 * dp * r1))
    a2 = r2 ** 2 * np.arccos((dp ** 2 + r2 ** 2 - r1 ** 2) / (2 * dp * r2))
    a3 = 0.5 * np.sqrt((-dp + r1 + r2) * (dp + r1 - r2) * (dp - r1 + r2) * (dp + r1 + r2))
    area[partial] = a1 + a2 - a3
    return area


def eclipse_shape_analytic(t: np.ndarray, t_ecl: float, orbit: Orbit) -> np.ndarray:
    """Same quantity as :func:`eclipse_shape`, from circle-overlap geometry alone.

    The planet is a uniform disk of radius k (in stellar radii); the star a disk
    of radius 1. Their sky-projected separation on a circular orbit is
    z = a/R* * sqrt(sin^2(phi) + cos^2(i) cos^2(phi)) with phi measured from
    mid-eclipse. Only the half of the orbit behind the star can be eclipsed.
    """
    phi = 2 * np.pi * (np.asarray(t, dtype=float) - t_ecl) / orbit.period
    ci = np.cos(np.radians(orbit.inc))
    z = orbit.a_rs * np.sqrt(np.sin(phi) ** 2 + (ci * np.cos(phi)) ** 2)
    hidden = _circle_overlap_area(1.0, orbit.rp_rs, z) / (np.pi * orbit.rp_rs ** 2)
    behind = np.cos(phi) > 0
    return 1.0 - np.where(behind, hidden, 0.0)
