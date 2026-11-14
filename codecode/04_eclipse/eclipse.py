"""Ingredient 04 (YOUR version): the shape of a secondary eclipse, from geometry alone.

Read first: learntopic/day1 (eclipse geometry) and learncode/04_eclipse.md.

Convention used everywhere in RockyEye:

    flux(t) / F_star = 1 + depth * shape(t)

    shape = 1  when the whole planet is visible (out of eclipse)
    shape = 0  when the planet is completely behind the star (mid-eclipse)

The reference uses the `batman` package. You build it WITHOUT batman: the
planet is a uniform disk of radius k = rp_rs, the star a disk of radius 1, and
the hidden fraction of the planet is just the overlap area of two circles.
The tests check you against batman to 1e-6.

Check yourself:   python run.py check 04
Target size:      ~60 lines of your own code
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Orbit:
    """Circular-orbit geometry. Times in days (BJD_TDB), angles in degrees."""

    period: float   # days
    t0: float       # TRANSIT mid-time (planet in front of the star)
    a_rs: float     # semi-major axis / stellar radius
    inc: float      # inclination, degrees (90 = edge-on)
    rp_rs: float    # planet radius / stellar radius

    @property
    def impact_parameter(self) -> float:
        """b = a/R* cos(i): how far from the star's centre the planet's path crosses (given)."""
        return self.a_rs * np.cos(np.radians(self.inc))

    def predicted_eclipse_time(self, t_ref: float) -> float:
        """Mid-ECLIPSE time nearest to ``t_ref``.

        TODO: for a circular orbit the eclipse sits exactly half an orbit after
        the transit: t0 + P*(n + 1/2) for an integer n. Pick the n that puts the
        result closest to t_ref.
        """
        raise NotImplementedError

    def duration_full(self) -> float:
        """T14: first to fourth contact (any part of the planet hidden), in days.

        TODO: T14 = P/pi * arcsin( sqrt((1+k)^2 - b^2) / (a/R* * sin i) )
        """
        raise NotImplementedError

    def duration_total(self) -> float:
        """T23: second to third contact (planet COMPLETELY hidden), in days.

        TODO: same as T14 with (1-k) instead of (1+k).
        """
        raise NotImplementedError


def circle_overlap_area(r1: float, r2: float, d) -> np.ndarray:
    """Area where two circles overlap. Radii r1 >= r2, centre distance d (array).

    TODO, three cases (vectorised with numpy masks, no Python loop over points):
      1. d <= r1 - r2       small circle fully inside  -> pi * r2^2
      2. d >= r1 + r2       no overlap                  -> 0
      3. otherwise          the "lens" formula:
           r1^2 arccos((d^2 + r1^2 - r2^2) / (2 d r1))
         + r2^2 arccos((d^2 + r2^2 - r1^2) / (2 d r2))
         - 0.5 sqrt((-d+r1+r2)(d+r1-r2)(d-r1+r2)(d+r1+r2))
    """
    raise NotImplementedError


def eclipse_shape(t, t_ecl: float, orbit: Orbit) -> np.ndarray:
    """Visible fraction of the planet's disk at times t (1 = all visible, 0 = all hidden).

    TODO:
      1. orbital phase from mid-eclipse:  phi = 2 pi (t - t_ecl) / P
      2. sky separation of the two centres, in stellar radii:
           z = a/R* * sqrt( sin^2(phi) + cos^2(i) cos^2(phi) )
      3. hidden fraction = circle_overlap_area(1, k, z) / (pi k^2)
      4. IMPORTANT: the planet can only be hidden when it is BEHIND the star,
         i.e. cos(phi) > 0. At transit (in front) nothing of the planet is hidden.
      5. return 1 - hidden fraction
    """
    raise NotImplementedError


def eclipse_flux(t, depth: float, t_ecl: float, orbit: Orbit) -> np.ndarray:
    """Relative flux 1 + depth * shape.  TODO: one line."""
    raise NotImplementedError
