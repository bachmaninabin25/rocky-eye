"""Ingredient 10: model comparison picks the right model; benchmark arithmetic."""
import numpy as np
import pytest

from conftest import PRIORS, synthetic_light_curve
from rockyeye import s10_compare as cmp
from rockyeye.s06_fit import EclipseFit, make_visit


def test_bic_picks_the_generating_model(orbit):
    lc = synthetic_light_curve(orbit, 2460864.9, seed=11)
    v = make_visit("v", lc, "linear")
    rows = cmp.compare_systematics(v, orbit, PRIORS, ["linear", "exp", "exp+x", "exp+xywbg"])
    by = {r["model"]: r for r in rows}
    assert by["linear"]["dbic"] > 10                 # no ramp term: clearly worse
    assert cmp.choose_model(rows)["model"] == "exp+x"
    assert by["linear"]["beta"] > by["exp+x"]["beta"]


def test_flat_data_prefers_the_simplest_model(orbit):
    lc = synthetic_light_curve(orbit, 2460864.9, amp=0.0, slope=0.0, cx=0.0, seed=12)
    rows = cmp.compare_systematics(make_visit("v", lc, "linear"), orbit, PRIORS, ["linear", "exp", "poly2"])
    assert cmp.choose_model(rows)["model"] == "linear"


def test_detection_bic(three_visits, orbit):
    det = cmp.detection_bic(EclipseFit(three_visits, orbit, PRIORS))
    assert det["dbic"] > 10


def test_benchmark_and_table():
    rows = cmp.benchmark_rows({"joint": (120.0, 20.0)}, [{"ref": "A", "label": "joint", "depth_ppm": 100.0, "err_ppm": 15.0},
                                                         {"ref": "A", "label": "missing", "depth_ppm": 1.0, "err_ppm": 1.0}])
    assert len(rows) == 1 and rows[0]["tension_sigma"] == pytest.approx(0.8)
    md = cmp.md_table(rows, [("ref", "Ref", ""), ("ours", "Ours", ".0f")])
    assert md.splitlines()[2] == "| A | 120 |"
