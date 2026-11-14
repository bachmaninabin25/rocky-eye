"""Reference for ingredient 04, mapped onto the lean API of 04_eclipse/eclipse.py."""
import numpy as np

from rockyeye.s04_eclipse import Orbit, _circle_overlap_area, eclipse_shape_analytic  # noqa: F401


def circle_overlap_area(r1, r2, d):
    return _circle_overlap_area(r1, r2, d)


def eclipse_shape(t, t_ecl, orbit):
    return eclipse_shape_analytic(t, t_ecl, orbit)


def eclipse_flux(t, depth, t_ecl, orbit):
    return 1.0 + depth * eclipse_shape_analytic(t, t_ecl, orbit)
