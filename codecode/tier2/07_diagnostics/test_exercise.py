import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _load import exercise  # noqa: E402
from rockyeye.s07_diagnostics import split_rhat as ref_rhat  # noqa: E402

X = exercise(__file__)


def test_well_mixed_chain_is_near_one():
    chain = np.random.default_rng(0).normal(size=(4000, 32, 3))
    r = X.split_rhat(chain)
    assert r.shape == (3,) and np.all(np.abs(r - 1) < 0.01)


def test_drifting_chain_is_flagged():
    rng = np.random.default_rng(1)
    chain = rng.normal(size=(4000, 32, 2))
    chain[2000:, :, 0] += 1.0                  # second half sits somewhere else: not converged
    r = X.split_rhat(chain)
    assert r[0] > 1.1 and abs(r[1] - 1) < 0.01


def test_matches_reference():
    chain = np.random.default_rng(2).normal(size=(501, 10, 4)).cumsum(axis=0)   # a sticky random walk
    assert np.allclose(X.split_rhat(chain), ref_rhat(chain))
