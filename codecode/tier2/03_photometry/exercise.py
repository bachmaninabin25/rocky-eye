"""Tier 2 exercise for ingredient 03 (photometry): where is the star, and how wide is it?

Read first: learncode/03_photometry.md.

The star's position (x, y) and width (sx, sy) on the detector change slightly
from frame to frame (the telescope jitters). They become the "x, y, w"
decorrelation columns in ingredient 05, so this little function feeds the
systematics model.

Check yourself:   python run.py check 03
Target size:      ~15 lines
"""
import numpy as np


def centroid_and_width(cube, bg, x0, y0, box=5):
    """Flux-weighted centre and RMS width of the star in every frame.

    cube: (n_frames, ny, nx) images.  bg: (n_frames,) sky level per frame.
    (x0, y0): rough star position. Use a FIXED window of half-size `box` around
    round(x0), round(y0) for every frame (a window that moves could hop between
    pixels and add its own noise).

    Pixel (row i, column j) has coordinates x = j, y = i.

    TODO:
      sub = window of each frame minus its bg
      tot = total flux in the window (per frame)
      px  = sub summed over rows    -> profile along x  (shape n_frames, window)
      py  = sub summed over columns -> profile along y
      x   = sum(px * xs) / tot            y likewise
      sx  = sqrt(|sum(px * (xs - x)^2)| / tot)    sy likewise
    Return x, y, sx, sy (each shape (n_frames,)).
    """
    raise NotImplementedError
