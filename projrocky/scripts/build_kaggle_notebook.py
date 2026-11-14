#!/usr/bin/env python
"""Build the two Kaggle notebooks from the real modules and configs.

    notebooks/kaggle_lhs1140b_sim.ipynb   simulated LHS 1140 b (106 segments, 199 GB attached)
    notebooks/kaggle_gj3929b.ipynb        real GJ 3929 b       (25 segments, 32 GB attached)

Stages 01-02 (raw ramps -> calibrated stamps) need ~230 GB of raw data, which never
touches the laptop. On Kaggle each dataset is attached to its notebook as a read-only
input. The notebooks are *generated*: they embed rockyeye/common.py, s01_fetch.py,
s02_calibrate.py and the config verbatim, so the code that runs on Kaggle is exactly
the code tested here. Rerun this script after changing any of those files.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["rockyeye/__init__.py", "rockyeye/common.py", "rockyeye/s01_fetch.py", "rockyeye/s02_calibrate.py"]
TARGETS = {
    "lhs1140b_sim": {"dataset": "stsci/rocky-worlds-lhs-1140b-simulations", "what": "9 simulated eclipses of LHS 1140 b",
                     "segments": 106, "gb": 199, "max_hours": 9.0},
    "gj3929b": {"dataset": "stsci/rocky-worlds-gj-3929b-observations", "what": "4 real eclipses of GJ 3929 b",
                "segments": 25, "gb": 32, "max_hours": 9.0},
}
SRC = "/kaggle/working/rockyeye_src"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# RockyEye — stages 01–02 on Kaggle: {name}

{what}: raw JWST ramps → stamps. Runs the official `jwst` pipeline (Detector1 + Image2) on
{segments} raw segments ({gb} GB, attached, never copied) and writes one small **stamp file** per
segment: an 81×81 px cut-out around the star for every integration. Everything after that
(photometry, fitting, physics) runs on the laptop.

**Before running**

1. *Add Input* → add the dataset `{dataset}`.
2. *Settings* → **Internet: On** (for `pip install jwst` and the CRDS reference files). No GPU.
3. *Save Version* → **Save & Run All (Commit)**. A commit runs in the background and keeps
   `/kaggle/working` as the notebook's output.

**How it protects a long run**

* Each segment's stamp is written to `/kaggle/working/stamps/` as soon as that segment is done,
  and `stamps_index.json` is rewritten after every segment.
* A segment that fails is logged and skipped; the run carries on.
* It stops by itself after `MAX_HOURS` = {max_hours} h (Kaggle's limit is 12 h), so the commit always
  finishes and the output is always saved.
* **To continue or retry**: start a new version with the previous version's output attached as an
  input (*Add Input → Your Work → Notebook Output*). Finished stamps are copied over first and
  skipped; only missing segments are computed.

Per segment: read the attached `*_uncal.fits` → 50-integration chunks in `/kaggle/temp` → stage 1 →
stage 2 → cut the stamp → delete chunk, ramp, rateints and calints files → next segment.

**Afterwards**: *Output → Download All*, then on the laptop:
`make import FROM=~/Downloads/<the zip or folder>`.
"""

PARAMS = """
# ---- parameters ---------------------------------------------------------------
TARGET = "{name}"          # config name (one dataset per notebook)
VISITS = None              # e.g. ["eclipse1", "eclipse2"] to do a subset; None = all
MAX_HOURS = {max_hours}            # stop cleanly well before Kaggle's 12 h session limit
JWST_VERSION = "3.0.0"     # same version as the laptop environment
"""

INSTALL = """
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"jwst=={JWST_VERSION}", "pyyaml"], check=True)
import jwst
print("jwst", jwst.__version__)
"""

RUN = '''
import csv, glob, json, os, shutil, sys, time

os.environ["CRDS_PATH"] = "/kaggle/temp/crds_cache"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
sys.path.insert(0, "%(src)s")

import numpy as np
from rockyeye.common import load_config, setup_logging
from rockyeye import s01_fetch, s02_calibrate

setup_logging("/kaggle/working/stamps/calibrate.log")
T0 = time.time()


def find_input_root(cfg):
    """Folder of the attached dataset: the one whose manifest.csv lists this target's data type."""
    want = "simulation" if "sim" in cfg["name"] else "real_observation"
    for man in glob.glob("/kaggle/input/**/manifest.csv", recursive=True):
        with open(man) as f:
            row = next(csv.DictReader(f), None)
        if row and row.get("data_type") == want:
            return os.path.dirname(man)
    raise FileNotFoundError(f"dataset {cfg['source']['dataset']} is not attached (Add Input)")


def restore_previous(out_dir, name):
    """Copy stamps from an attached earlier run so they are not recomputed."""
    n = 0
    for p in glob.glob(f"/kaggle/input/**/stamps/{name}/stage02/**/*", recursive=True):
        if os.path.isfile(p):
            rel = p.split(f"/stamps/{name}/stage02/", 1)[1]
            dest = os.path.join(out_dir, "stage02", rel)
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copyfile(p, dest)
                n += 1
    return n


name = TARGET
cfg = load_config(f"%(src)s/configs/{name}.yaml")
cfg["source"]["local_root"] = find_input_root(cfg)          # read the attached files in place
cfg["paths"] = {"data": f"/kaggle/temp/rockyeye/{name}",     # scratch: chunks, rateints, calints
                "products": f"/kaggle/working/stamps/{name}", "crds": "/kaggle/temp/crds_cache"}
cfg["keep"] = {"raw": True, "rateints": False, "calints": False}
print(f"=== {name}: input {cfg['source']['local_root']}")
print("restored from a previous run:", restore_previous(cfg["paths"]["products"], name), "files")
print(s01_fetch.format_plan(s01_fetch.plan(cfg, VISITS)))
files = s01_fetch.list_files(cfg, VISITS)
key = s02_calibrate.stage_key(cfg)


FAILED = []


def write_index(n_done=None, n_todo=None, failed=None):
    """Rewrite the progress index; called after every segment so it is never stale."""
    if failed is not None:
        FAILED[:] = failed
    done = [f for f in files if s02_calibrate.is_cached(s02_calibrate.stamp_path(cfg, f), key)]
    index = {name: {"stage_key": key, "segments_total": len(files), "segments_done": len(done),
                    "hours_elapsed": round((time.time() - T0) / 3600.0, 2), "failed": list(FAILED),
                    "missing": [f.filename for f in files if f not in done]}}
    tmp = "/kaggle/working/stamps/stamps_index.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(index, fh, indent=1)
    os.replace(tmp, "/kaggle/working/stamps/stamps_index.json")
    return index[name]


write_index()
s02_calibrate.process(cfg, VISITS, max_hours=MAX_HOURS - (time.time() - T0) / 3600.0,
                      continue_on_error=True, after_segment=write_index)
status = write_index()
shutil.rmtree("/kaggle/temp/rockyeye", ignore_errors=True)
print(json.dumps({k: v for k, v in status.items() if k != "missing"}, indent=1))
print("ALL DONE" if status["segments_done"] == status["segments_total"]
      else "INCOMPLETE: start a new version with this version's output attached as input to continue")
''' % {"src": SRC}

QUICKLOOK = '''
# Quick look: crude light curve per visit (sum of the central 11x11 px, sky from the stamp edge).
# Only a sanity check that stamps contain a star and look stable; real photometry is stage 03.
import matplotlib.pyplot as plt

for name in [TARGET]:
    visits = sorted(os.listdir(f"/kaggle/working/stamps/{name}/stage02")) if os.path.isdir(f"/kaggle/working/stamps/{name}/stage02") else []
    if not visits:
        continue
    fig, axes = plt.subplots(len(visits), 1, figsize=(9, 1.8 * len(visits)), squeeze=False)
    for ax, v in zip(axes[:, 0], visits):
        t, f = [], []
        for p in sorted(glob.glob(f"/kaggle/working/stamps/{name}/stage02/{v}/*_stamp.npz")):
            z = np.load(p)
            s = z["sci"]
            c = s.shape[1] // 2
            sky = np.nanmedian(s[:, :6, :], axis=(1, 2))
            f.append(np.nansum(s[:, c - 5:c + 6, c - 5:c + 6] - sky[:, None, None], axis=(1, 2)))
            t.append(z["time"])
        if t:
            t, f = np.concatenate(t), np.concatenate(f)
            ax.plot((t - t[0]) * 24, f / np.nanmedian(f), ".", ms=1)
            ax.set_ylim(0.99, 1.01)
        ax.set_ylabel(v)
    axes[-1, 0].set_xlabel("hours since start")
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()
'''


def build(name: str, info: dict) -> Path:
    fmt = dict(name=name, **info)
    cells = [md(INTRO.format(**fmt)), code(PARAMS.format(**fmt)), code(INSTALL),
             md("## Pipeline code\n\nWritten out verbatim from the RockyEye repository "
                "(`scripts/build_kaggle_notebook.py` generates these cells)."),
             code(f"import os\nfor d in ('rockyeye', 'configs'):\n    os.makedirs(f'{SRC}/' + d, exist_ok=True)")]
    for rel in CODE + _config_chain(f"configs/{name}.yaml"):
        text = (ROOT / rel).read_text()
        cells.append(code(f"%%writefile {SRC}/{rel}\n{text}" if text.strip() else f"%%writefile {SRC}/{rel}\n# package marker\n"))
    cells += [md("## Run stages 01–02"), code(RUN), md("## Quick look"), code(QUICKLOOK)]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = ROOT / "notebooks" / f"kaggle_{name}.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, {len(cells)} cells)")
    return out


def _config_chain(rel: str) -> list[str]:
    """A config plus any file it ``extends``, so the notebook is self-contained."""
    import yaml

    out = [rel]
    parent = yaml.safe_load((ROOT / rel).read_text()).get("extends")
    if parent:
        out += _config_chain(str(Path(rel).parent / parent))
    return out


def main():
    for name, info in TARGETS.items():
        build(name, info)
    old = ROOT / "notebooks" / "rockyeye_kaggle_stage01_02.ipynb"      # superseded combined notebook
    if old.exists():
        old.unlink()


if __name__ == "__main__":
    main()
