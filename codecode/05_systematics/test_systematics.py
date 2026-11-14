"""Ingredient 05 tests: parsing, basis columns, and reaching the white-noise floor."""
import numpy as np
import pytest

from conftest import deps

S = deps.get("05")
REF = None


def ref():
    import importlib.util
    global REF
    if REF is None:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "_reference" / "ref_05.py"
        spec = importlib.util.spec_from_file_location("ref05_for_tests", p)
        REF = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(REF)
    return REF


def test_parse_names():
    assert S.parse_model("exp+xywbg") == ("exp", ("x", "y", "w", "bg"))
    assert S.parse_model("exp+bgx") == ("exp", ("x", "bg"))
    assert S.parse_model("linear") == ("linear", ())
    assert S.aux_columns("exp+xywbg") == ("x", "y", "sx", "sy", "bg")
    assert S.n_linear("exp+xywbg") == 3 + 5
    assert S.nonlinear_names("exp") == ("log_tau",)
    assert S.nonlinear_names("exp2") == ("log_tau", "log_tau2")
    with pytest.raises(ValueError):
        S.parse_model("spline")
    with pytest.raises(ValueError):
        S.parse_model("exp+q")


@pytest.mark.parametrize("name,theta", [("const", ()), ("linear", ()), ("poly2", ()),
                                        ("exp+xy", (np.log(0.5),)),
                                        ("exp2+xywbg", (np.log(0.2), np.log(2.0)))])
def test_basis_matches_reference(name, theta):
    rng = np.random.default_rng(1)
    th = np.linspace(0, 5, 300)
    aux = {k: rng.normal(size=300) for k in ("x", "y", "sx", "sy", "bg")}
    B = S.basis(name, th, aux, theta)
    assert B.shape == (300, S.n_linear(name))
    assert np.allclose(B, ref().basis(name, th, aux, theta))


def test_standardise_is_robust():
    rng = np.random.default_rng(2)
    v = rng.normal(5.0, 2.0, 5000)
    v[0] = 1e6                                         # one cosmic ray
    s = S.standardise(v)
    assert np.median(s) == pytest.approx(0, abs=0.02)
    assert np.std(s[1:]) == pytest.approx(1.0, abs=0.05)
    assert s.max() == 8.0                              # clipped
    assert np.allclose(s, ref().standardise(v))


def test_solve_linear_matches_reference():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(500, 4))
    y = M @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(0, 0.1, 500)
    c, chi2, logdet = S.solve_linear(y, M)
    rc, rchi2, rlogdet = ref().solve_linear(y, M)
    assert np.allclose(c, rc) and chi2 == pytest.approx(rchi2) and logdet == pytest.approx(rlogdet)
    assert logdet == pytest.approx(np.linalg.slogdet(M.T @ M)[1])


def test_binned_rms_white_noise_follows_sqrt_n():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 1, 20000)
    sizes, rms, expected = S.binned_rms(r)
    rs, rr, re = ref().binned_rms(r)
    assert np.array_equal(sizes, rs) and np.allclose(rms, rr, equal_nan=True) and np.allclose(expected, re)
    sel = sizes < 200
    assert np.allclose(rms[sel], expected[sel], rtol=0.25)


def test_reaches_white_noise_floor():
    """The RIGHT model leaves white residuals at the injected 200 ppm; the wrong one leaves red noise."""
    rng = np.random.default_rng(3)
    th = np.linspace(0, 6, 2000)
    x = S.standardise(rng.normal(size=th.size))
    truth = 1 + 3e-4 * (th - 3) / 6 + 1e-3 * np.exp(-th / 0.3) + 8e-5 * x
    y = truth + rng.normal(0, 200e-6, th.size)
    B = S.basis("exp+x", th, {"x": x}, [np.log(0.3)])
    c, _, _ = S.solve_linear(y, B)
    resid = y - B @ c
    assert np.std(resid) * 1e6 == pytest.approx(200, rel=0.05)
    assert S.red_noise_beta(resid, 10.8, 1800) < 1.3
    B0 = S.basis("linear", th, {}, [])
    c0, _, _ = S.solve_linear(y, B0)
    assert S.red_noise_beta(y - B0 @ c0, 10.8, 1800) > 1.5
