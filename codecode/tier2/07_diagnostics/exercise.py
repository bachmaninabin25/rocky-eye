"""Tier 2 exercise for ingredient 07 (MCMC diagnostics): split R-hat.

Read first: learntopic/day2 (convergence) and learncode/07_diagnostics.md.

R-hat asks: "do different pieces of the chain agree?" Cut every walker's chain
in half, treat each half as its own sequence, and compare the variance BETWEEN
sequences with the variance WITHIN them. If everything samples the same
distribution, R-hat -> 1. RockyEye requires R-hat <= 1.01.

Check yourself:   python run.py check 07
Target size:      ~10 lines
"""
import numpy as np


def split_rhat(chain):
    """Split R-hat per parameter.  chain: (n_steps, n_walkers, n_dim) -> (n_dim,)

    TODO:
      half = n_steps // 2
      seq  = first halves and second halves side by side: shape (half, 2*n_walkers, n_dim)
             (np.concatenate along axis=1)
      n    = half
      W    = mean over sequences of each sequence's variance (ddof=1)        "within"
      Bn   = n * variance (ddof=1) of the sequence means                     "between"
      var+ = (n-1)/n * W + Bn / n
      return sqrt(var+ / W)
    """
    raise NotImplementedError
