"""Reference for ingredient 09, mapped onto the lean API of 09_physics/physics.py.

The lean version uses a simple top-hat band (13.5-16.6 um) instead of the real
F1500W throughput curve, so the reference Band is built with the same top-hat.
"""
import numpy as np

from rockyeye import s09_physics as _ph
from rockyeye.s09_physics import (C, H, KB, PC, RSUN, compare_scenarios,  # noqa: F401
                                  dayside_temperature, depth_from_temperature,
                                  interpret_bayes_factor, planck_nu, temperature_from_depth)

TOPHAT = (13.5, 16.6)


class Band(_ph.Band):
    def __init__(self, lo_um=TOPHAT[0], hi_um=TOPHAT[1], n=400):
        wl = np.linspace(lo_um, hi_um, n)
        super().__init__(wl, np.ones_like(wl))


def stellar_intensity(band, teff):
    return _ph.stellar_intensity(band, "blackbody", teff=teff)
