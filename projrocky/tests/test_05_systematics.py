"""Ingredient 05: model parsing, basis shapes, and reaching the white-noise residual level."""
import numpy as np
import pytest

from rockyeye.s05_systematics import SystematicsModel, binned_rms, red_noise_beta, solve_linear, standardise


def test_parse_names():
    m = SystematicsModel.parse("exp+xywbg")
    assert m.baseline == "exp" and m.extras == ("x", "y", "w", "bg")
    assert m.aux_columns == ("x", "y", "sx", "sy", "bg")
    assert m.n_linear == 3 + 5 and m.nonlinear_names == ("log_tau",)
    assert SystematicsModel.parse("linear").n_linear == 2
    assert SystematicsModel.parse("exp2").n_nonlinear == 2
    with pytest.raises(ValueError):
        SystematicsModel.parse("spline")
    with pytest.raises(ValueError):
        SystematicsModel.parse("exp+q")


def test_basis_shape_and_ramp():
    th = np.linspace(0, 5, 300)
    aux = {k: np.zeros(300) for k in ("x", "y", "sx", "sy", "bg")}
    B = SystematicsModel.parse("exp+xy").basis(th, aux, [np.log(0.5)])
    assert B.shape == (300, 5)
    assert B[0, 2] == pytest.approx(1.0) and B[-1, 2] == pytest.approx(np.exp(-10))


def test_reaches_white_noise_floor():
    """Fitting the right model leaves residuals at the injected noise level, with beta ~ 1."""
    rng = np.random.default_rng(3)
    th = np.linspace(0, 6, 2000)
    x = standardise(rng.normal(size=th.size))
    truth = 1 + 3e-4 * (th - 3) / 6 + 1e-3 * np.exp(-th / 0.3) + 8e-5 * x
    y = truth + rng.normal(0, 200e-6, th.size)
    B = SystematicsModel.parse("exp+x").basis(th, {"x": x}, [np.log(0.3)])
    c, chi2, _ = solve_linear(y, B)
    resid = y - B @ c
    assert np.std(resid) * 1e6 == pytest.approx(200, rel=0.05)
    assert red_noise_beta(resid, 10.8, 1800) < 1.3
    # the wrong model (no ramp) leaves obvious red noise
    B0 = SystematicsModel.parse("linear").basis(th, {}, [])
    c0, _, _ = solve_linear(y, B0)
    assert red_noise_beta(y - B0 @ c0, 10.8, 1800) > 1.5


def test_binned_rms_white_noise_follows_sqrt_n():
    rng = np.random.default_rng(4)
    sizes, rms, expected = binned_rms(rng.normal(0, 1, 20000))
    sel = sizes < 200
    assert np.allclose(rms[sel], expected[sel], rtol=0.25)
