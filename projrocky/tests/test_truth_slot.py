"""The truth re-validation slot: arithmetic of 'where does the truth sit in the posterior'."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("revalidate_truth", Path(__file__).resolve().parents[1] / "scripts" / "revalidate_truth.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def test_truth_inside_posterior_passes():
    s = np.random.default_rng(0).normal(100.0, 20.0, 200000)
    c = rt.check(s, 110.0)
    assert c["offset_sigma"] == pytest.approx(-0.5, abs=0.02)
    assert c["truth_quantile"] == pytest.approx(0.6915, abs=0.01)
    assert c["inside_68"] and c["inside_95"]


def test_truth_far_away_fails():
    s = np.random.default_rng(1).normal(100.0, 20.0, 200000)
    c = rt.check(s, 30.0)
    assert c["offset_sigma"] == pytest.approx(3.5, abs=0.05)
    assert not c["inside_68"] and not c["inside_95"]


def test_asymmetric_error_uses_the_side_facing_the_truth():
    rng = np.random.default_rng(2)
    s = np.where(rng.random(200000) < 0.5, 100 - np.abs(rng.normal(0, 10, 200000)), 100 + np.abs(rng.normal(0, 30, 200000)))
    assert abs(rt.check(s, 130.0)["offset_sigma"]) < abs(rt.check(s, 70.0)["offset_sigma"])


def test_simulation_truth_slot_is_empty_until_published():
    """Guard: nobody should fill the official truth by accident; it must come with a source."""
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "lhs1140b_sim.yaml").read_text())
    t = cfg["truth"]
    assert set(t) == {"depth_ppm", "dt_min", "source"}
    assert (t["depth_ppm"] is None) == (t["source"] is None)
