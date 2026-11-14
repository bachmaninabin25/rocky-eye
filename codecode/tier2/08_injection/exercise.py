"""Tier 2 exercise for ingredient 08 (injection-recovery).

Read first: learncode/08_injection.md.

"Would we have seen an eclipse of depth D if it were there, and measured it
right?" Plant fake eclipses, run the same machinery, compare. Two pieces:

  1. inject_into_stamps: dim the STAR (not the sky) in the images.
  2. linear_depth: the fast estimator used for hundreds of injection trials.
     For depth << 1:  (1 + d*s(t)) * (B c)  ~=  B c + (d*c0) * s(t)
     so the eclipse is just ONE MORE COLUMN of the design matrix, and the depth
     and its error come out of a single least-squares solve.

Check yourself:   python run.py check 08
Target size:      ~20 lines
"""
import numpy as np


def inject_into_stamps(sci, bg, shape, depth):
    """Return a copy of the image cube with an eclipse planted in the star.

    sci: (n, ny, nx) images.  bg: (n,) sky per frame.  shape: (n,) eclipse shape
    (1 out of eclipse, 0 fully in).  depth: fraction (e.g. 150e-6).

    TODO: star = sci - sky. Scale ONLY the star by (1 + depth * (shape - 1)):
    unchanged out of eclipse, dimmer by `depth` in eclipse. Add the sky back.
    Cast the scale and sky to float32 (the images are float32).
    """
    raise NotImplementedError


def linear_depth(B, s, flux):
    """Depth and its 1-sigma error (both ppm) from one linear solve.

    B: (n, k) systematics design matrix, column 0 = ones.  s: (n,) eclipse
    shape MINUS 1 (0 out of eclipse, -1 in).  flux: (n,).

    TODO:
      M = [B, s] (append s as the last column); QR solve for c
      sigma^2 = chi2 / (n - number of columns)
      covariance = sigma^2 * inv(R) inv(R)^T     (QR: M^T M = R^T R)
      the last coefficient is depth * baseline; baseline = median(B @ c[:-1])
      return 1e6 * c[-1] / baseline,  1e6 * sqrt(cov[-1, -1]) / baseline
    """
    raise NotImplementedError
