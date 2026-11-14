#!/usr/bin/env python
"""Cross-check RockyEye photometry against Eureka! stage 3 on the same calibrated frames.

Two steps, because Eureka! lives in its own conda environment (rockyeye-eureka):

  1. (rockyeye-eureka env)  python scripts/eureka_crosscheck.py run --calints <dir> --out <dir>
       runs Eureka! S3 aperture photometry on *_calints.fits and saves time + flux
  2. (rockyeye env)         python scripts/eureka_crosscheck.py compare --stamps <dir> --eureka <dir>/eureka_lc.npz
       runs RockyEye photometry with the same aperture and sky ring and compares

Same frames in, so differences are purely the extraction algorithms.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

RADIUS, SKYIN, SKYWIDTH = 5.0, 16.0, 14.0

ECF = """
ncpu 1
nfiles 1
max_memory 0.5
indep_batches False
suffix calints
calibrated_spectra False
photometry True
inst miri
pmap 1584
ywindow [None, None]
xwindow [None, None]
src_pos_type gaussian
record_ypos True
dqmask True
expand 1
flag_bg True
bg_thresh [5, 5]
interp_method cubic
centroid_method mgmc
ctr_guess None
ctr_cutout_size 10
centroid_tech com
gauss_frame 15
phot_method photutils
aperture_edge exact
aperture_shape circle
photap {radius}
skyin {skyin}
skywidth {skywidth}
skip_apphot_bg False
isplots_S3 1
nplots 1
testing_S3 False
hide_plots True
save_output True
save_fluxdata False
verbose False
topdir {topdir}
inputdir {inputdir}
outputdir {outputdir}
"""


def run(a):
    import eureka.S3_data_reduction.s3_reduce as s3
    import xarray as xr

    import os

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Eureka! joins inputdir and outputdir onto topdir, so both must be relative to it.
    calints = Path(a.calints).resolve()
    topdir = Path(os.path.commonpath([calints, out]))
    (out / "S3_xcheck.ecf").write_text(ECF.format(
        radius=RADIUS, skyin=SKYIN, skywidth=SKYWIDTH, topdir=str(topdir),
        inputdir=os.path.relpath(calints, topdir), outputdir=os.path.relpath(out / "Stage3", topdir)))
    s3.reduce("xcheck", ecf_path=str(out) + "/")
    files = sorted(glob.glob(str(out / "Stage3" / "**" / "*SpecData.h5"), recursive=True))
    ds = xr.open_dataset(files[-1], engine="h5netcdf")
    print(ds)
    np.savez(out / "eureka_lc.npz", time=np.asarray(ds["time"]), flux=np.asarray(ds["aplev"]).squeeze(),
             bg=np.asarray(ds["skylev"]).squeeze(), x=np.asarray(ds["centroid_x"]).squeeze(),
             y=np.asarray(ds["centroid_y"]).squeeze())
    print("saved", out / "eureka_lc.npz")


def compare(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rockyeye import s03_photometry as ph
    from rockyeye.common import atomic_write_json

    parts = [np.load(p) for p in sorted(glob.glob(a.stamps + "/*_stamp.npz"))]
    parts = [p for p in parts if len(p["time"]) == len(p["sci"])]
    stamps = {k: np.concatenate([p[k] for p in parts]) for k in ("time", "sci", "err", "bad")}
    stamps["origin"], stamps["centre"] = parts[0]["origin"], parts[0]["centre"]
    lc = ph.extract_light_curve(stamps, {"aperture_radii": [RADIUS], "annulus": [SKYIN, SKYIN + SKYWIDTH]})
    e = np.load(a.eureka)
    n = min(len(e["flux"]), len(lc["flux"]))
    ours, theirs = lc["flux_dn_s"][:n], e["flux"][:n]
    ratio = ours / theirs
    rel = ratio / np.median(ratio) - 1
    res = {"n_integrations": int(n), "aperture_px": RADIUS, "annulus_px": [SKYIN, SKYIN + SKYWIDTH],
           "median_flux_ratio_rockyeye_over_eureka": float(np.median(ratio)),
           "rms_relative_difference_ppm": float(np.std(rel) * 1e6),
           "rockyeye_scatter_ppm": float(np.std(np.diff(ours)) / np.sqrt(2) / np.median(ours) * 1e6),
           "eureka_scatter_ppm": float(np.std(np.diff(theirs)) / np.sqrt(2) / np.median(theirs) * 1e6),
           "correlation_of_normalised_flux": float(np.corrcoef(ours, theirs)[0, 1]),
           "median_centroid_offset_px": [float(np.median(lc["x"][:n] - e["x"][:n])), float(np.median(lc["y"][:n] - e["y"][:n]))]}
    for k, v in res.items():
        print(f"{k:45s} {v}")
    if a.json:
        atomic_write_json(Path(a.json), res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--calints", required=True); r.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--stamps", required=True); c.add_argument("--eureka", required=True)
    c.add_argument("--json")
    a = ap.parse_args()
    {"run": run, "compare": compare}[a.cmd](a)
