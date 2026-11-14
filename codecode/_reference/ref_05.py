"""Reference for ingredient 05, mapped onto the lean API of 05_systematics/systematics.py."""
from rockyeye.s05_systematics import (AUX_KEYS, BASELINES, SystematicsModel,  # noqa: F401
                                      binned_rms, red_noise_beta, solve_linear, standardise)


def parse_model(name):
    m = SystematicsModel.parse(name)
    return m.baseline, m.extras


def aux_columns(name):
    return SystematicsModel.parse(name).aux_columns


def n_linear(name):
    return SystematicsModel.parse(name).n_linear


def nonlinear_names(name):
    return SystematicsModel.parse(name).nonlinear_names


def basis(name, th, aux, theta=()):
    return SystematicsModel.parse(name).basis(th, aux, theta)
