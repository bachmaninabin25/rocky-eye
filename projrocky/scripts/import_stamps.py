#!/usr/bin/env python
"""Bring stamps made by the Kaggle notebook into products/<config>/stage02/.

Give it the notebook output as downloaded: a zip file or a folder that contains
``stamps/<config name>/stage02/<visit>/*_stamp.npz``. It copies the files,
checks that every segment in the manifest is present, and checks that the
stamps were made with the same stage 1-2 settings as the local config.
"""
import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockyeye import s01_fetch, s02_calibrate  # noqa: E402
from rockyeye.common import is_cached, load_config, product_dir  # noqa: E402

CONFIGS = ["configs/lhs1140b_sim.yaml", "configs/gj3929b.yaml"]


def import_one(cfg: dict, source: Path) -> dict:
    hits = [p for p in source.rglob("stage02") if p.parent.name == cfg["name"]]
    if not hits:
        return {"found": False}
    dest = product_dir(cfg, "stage02")
    copied = 0
    for f in hits[0].rglob("*"):
        if f.is_file():
            out = dest / f.relative_to(hits[0])
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.stat().st_size != f.stat().st_size:
                shutil.copyfile(f, out)
                copied += 1
    files = s01_fetch.list_files(cfg)
    key = s02_calibrate.stage_key(cfg)
    have = [f for f in files if s02_calibrate.stamp_path(cfg, f).exists()]
    fresh = [f for f in have if is_cached(s02_calibrate.stamp_path(cfg, f), key)]
    visits = {}
    for f in files:
        v = visits.setdefault(f.visit, [0, 0])
        v[1] += 1
        v[0] += f in fresh
    return {"found": True, "copied": copied, "segments": len(files), "present": len(have),
            "matching_config": len(fresh), "visits": visits}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", required=True, help="zip file or folder downloaded from Kaggle")
    a = ap.parse_args()
    src = Path(a.src).expanduser()
    tmp = None
    if src.is_file() and zipfile.is_zipfile(src):
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp.name)
        src = Path(tmp.name)
    ok = True
    for c in CONFIGS:
        cfg = load_config(c)
        r = import_one(cfg, src)
        if not r["found"]:
            print(f"{cfg['name']}: no stamps in this download")
            continue
        print(f"{cfg['name']}: copied {r['copied']} files; {r['present']}/{r['segments']} segments present, "
              f"{r['matching_config']} match the local stage 1-2 config")
        for v, (n, tot) in r["visits"].items():
            print(f"   {v:<10} {n}/{tot}" + ("" if n == tot else "   <-- incomplete"))
        if r["present"] and r["matching_config"] < r["present"]:
            ok = False
            print("   WARNING: some stamps were made with different stage 1-2 settings than this config.\n"
                  "   Rebuild the notebook (scripts/build_kaggle_notebook.py) and rerun it, or revert the config.")
    if tmp:
        tmp.cleanup()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
