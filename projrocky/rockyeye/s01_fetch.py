"""Ingredient 01 — data fetcher with a dry-run mode.

Two sources are supported, both without any login:

* ``kaggle``: the Rocky Worlds Data Challenge datasets, fetched with Kaggle's
  official ``kagglehub`` client (public datasets need no credentials). A
  ``manifest.csv`` lists every ``*_uncal.fits`` file with its size and SHA-256.
* ``mast``: the public JWST archive, listed with astroquery and downloaded
  from the MAST file endpoint.

Design rules (from the spec):
  1. Never download before the size is known and printed.
  2. ``dry_run=True`` prints the plan and touches nothing.
  3. Downloads resume (HTTP Range into a ``.part`` file) and are verified.
  4. Refuse to start a file that would push free disk below a safety floor.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import time
import urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

from .common import data_dir, free_gb, log, resolve

MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file"


@dataclass
class RemoteFile:
    """One raw file we might download."""

    visit: str          # e.g. "eclipse3"
    segment: int        # 1-based segment number within the visit
    filename: str
    url: str
    size_bytes: int
    sha256: str | None  # None when the archive gives no checksum
    local_path: str

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


# ----------------------------------------------------------------- listing ---

def _kagglehub(cfg: dict):
    """Kaggle's official client, with its download cache kept inside the project's data folder."""
    os.environ.setdefault("KAGGLEHUB_CACHE", str(resolve(cfg["paths"].get("kaggle_cache", "data/kagglehub_cache"))))
    import kagglehub

    return kagglehub


def _kaggle_manifest(cfg: dict) -> Path:
    """Fetch (once) the tiny manifest.csv that describes the dataset."""
    src = cfg["source"]
    root = os.environ.get("ROCKYEYE_INPUT_ROOT") or src.get("local_root")
    if root and (Path(root) / "manifest.csv").exists():
        return Path(root) / "manifest.csv"
    path = resolve(src["manifest"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        got = _kagglehub(cfg).dataset_download(src["dataset"], path="manifest.csv")
        shutil.copyfile(got, path)
        log.info("fetched manifest (%d bytes) -> %s", path.stat().st_size, path)
    return path


def _list_kaggle(cfg: dict) -> list[RemoteFile]:
    """Files of a Kaggle dataset, from its manifest.

    Two optional config keys under ``source``:
      local_root        the dataset is already on disk here (a Kaggle notebook's
                        /kaggle/input/<dataset>): nothing is downloaded or deleted.
      visit_from_obsnum {obs number: visit name}: rename by JWST observation
                        number, used to split GJ 3929 b's eclipse 4 into 4a/4b.
    """
    src = cfg["source"]
    raw = data_dir(cfg) / "raw"
    local_root = os.environ.get("ROCKYEYE_INPUT_ROOT") or src.get("local_root")
    by_obs = {int(k): v for k, v in (src.get("visit_from_obsnum") or {}).items()}
    out = []
    with open(_kaggle_manifest(cfg)) as f:
        for row in csv.DictReader(f):
            name = row["filename"]
            visit = row["eclipse"]
            if by_obs and name.startswith("jw") and int(name[7:10]) in by_obs:
                visit = by_obs[int(name[7:10])]
            local = Path(local_root) / row["relative_path"] if local_root else raw / visit / name
            out.append(RemoteFile(
                visit=visit,
                segment=int(row["segment"]),
                filename=name,
                url=f"kaggle://{src['dataset']}/{row['relative_path']}",
                size_bytes=int(row["size_bytes"]),
                sha256=row["sha256"],
                local_path=str(local),
            ))
    return out


def is_input_file(cfg: dict, rf: RemoteFile) -> bool:
    """True if the raw file lives in a read-only input folder that we must never delete from."""
    root = os.environ.get("ROCKYEYE_INPUT_ROOT") or cfg["source"].get("local_root")
    return bool(root) and str(rf.local_path).startswith(str(root))


def _list_mast(cfg: dict) -> list[RemoteFile]:
    """List uncal files of a JWST program on MAST. Visits come from ``source.visits``.

    ``source.visits`` maps our visit names to lists of JWST observation numbers,
    e.g. ``eclipse4: [4, 5]`` because GJ 3929 b's 4th eclipse was split in two.
    The listing is cached to JSON so later dry-runs work offline.
    """
    import json

    src = cfg["source"]
    cache = data_dir(cfg) / "mast_listing.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        from astroquery.mast import Observations

        obs = Observations.query_criteria(
            obs_collection="JWST", proposal_id=str(src["program"]),
            instrument_name="MIRI/IMAGE", filters=src.get("filter", "F1500W"))
        prods = Observations.get_product_list(obs)
        keep = Observations.filter_products(prods, productSubGroupDescription="UNCAL")
        rows = sorted({(str(p["productFilename"]), int(p["size"]), str(p["dataURI"])) for p in keep})
        rows = [list(r) for r in rows]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows, indent=1))
    obs_to_visit = {int(o): v for v, obs_list in src["visits"].items() for o in obs_list}
    raw = data_dir(cfg) / "raw"
    out = []
    for fname, size, uri in rows:
        # jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>-seg<NNN>_mirimage_uncal.fits
        obsnum = int(fname[7:10])
        if obsnum not in obs_to_visit or "-seg" not in fname:
            continue
        seg = int(fname.split("-seg")[1][:3])
        visit = obs_to_visit[obsnum]
        out.append(RemoteFile(
            visit=visit, segment=seg, filename=fname,
            url=f"{MAST_FILE}?uri={urllib.parse.quote(uri, safe='')}",
            size_bytes=size, sha256=None,
            local_path=str(raw / visit / fname)))
    return out


def _list_synthetic(cfg: dict) -> list[RemoteFile]:
    """Placeholder "raw files" for the synthetic test bench (see synth.py): nothing to download."""
    n_seg = cfg["source"].get("segments_per_visit", 3)
    return [RemoteFile(visit=v, segment=s, filename=f"synthetic-{v}-seg{s:03d}_mirimage_uncal.fits",
                       url="synthetic://", size_bytes=0, sha256=None, local_path="")
            for v in cfg["visits"] for s in range(1, n_seg + 1)]


def list_files(cfg: dict, visits: list[str] | None = None) -> list[RemoteFile]:
    """All raw files for this config, ordered by visit then time, optionally filtered."""
    kind = cfg["source"]["kind"]
    files = {"kaggle": _list_kaggle, "mast": _list_mast, "synthetic": _list_synthetic}[kind](cfg)
    if visits:
        files = [f for f in files if f.visit in visits]
    # filename order == time order within a visit (obs number, then segment)
    files.sort(key=lambda f: (f.visit, f.filename))
    return files


# ---------------------------------------------------------------- planning ---

def plan(cfg: dict, visits: list[str] | None = None) -> dict:
    """Work out what a download would involve, without downloading anything."""
    files = list_files(cfg, visits)
    todo = [f for f in files if not Path(f.local_path).exists()]
    per_visit: dict[str, dict] = {}
    for f in files:
        d = per_visit.setdefault(f.visit, {"n_files": 0, "gb": 0.0, "gb_missing": 0.0})
        d["n_files"] += 1
        d["gb"] += f.size_gb
        if f in todo:
            d["gb_missing"] += f.size_gb
    return {
        "n_files": len(files),
        "total_gb": sum(f.size_gb for f in files),
        "missing_gb": sum(f.size_gb for f in todo),
        "largest_file_gb": max((f.size_gb for f in files), default=0.0),
        "free_gb": free_gb(data_dir(cfg)),
        "per_visit": per_visit,
        "files": [asdict(f) for f in files],
    }


def format_plan(p: dict, streaming: bool = True) -> str:
    """Human-readable dry-run report."""
    lines = [f"{'visit':<10} {'files':>5} {'size GB':>8} {'to fetch GB':>12}"]
    for v, d in p["per_visit"].items():
        lines.append(f"{v:<10} {d['n_files']:>5} {d['gb']:>8.2f} {d['gb_missing']:>12.2f}")
    lines.append(f"{'TOTAL':<10} {p['n_files']:>5} {p['total_gb']:>8.2f} {p['missing_gb']:>12.2f}")
    lines.append(f"free disk: {p['free_gb']:.1f} GB; largest single file: {p['largest_file_gb']:.2f} GB")
    if streaming:
        lines.append("streaming mode: one raw file on disk at a time "
                     f"(peak raw footprint ~{p['largest_file_gb']:.1f} GB, not {p['total_gb']:.0f} GB)")
    elif p["missing_gb"] > p["free_gb"]:
        lines.append("WARNING: the full set does NOT fit on this disk; use streaming mode")
    return "\n".join(lines)


# ------------------------------------------------------------- downloading ---

def sha256_of(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(rf: RemoteFile, min_free_gb: float = 8.0, dry_run: bool = False,
             max_retries: int = 8, verify: bool = True) -> Path:
    """Download one file, resuming a partial ``.part`` if present.

    Returns the final path. With ``dry_run`` it only logs what it would do.
    """
    dest = Path(rf.local_path)
    if dest.exists() and dest.stat().st_size == rf.size_bytes:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    need_gb = (rf.size_bytes - have) / 1e9
    if dry_run:
        log.info("[dry-run] would download %s (%.2f GB, %.2f GB remaining)",
                 rf.filename, rf.size_gb, need_gb)
        return dest
    if free_gb(dest.parent) - need_gb < min_free_gb:
        raise OSError(f"refusing to download {rf.filename}: needs {need_gb:.2f} GB and would "
                      f"leave less than {min_free_gb} GB free ({free_gb(dest.parent):.1f} GB free now)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(resolve("data/kagglehub_cache")))

    for attempt in range(1, max_retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= rf.size_bytes:
            break
        try:
            if rf.url.startswith("kaggle://"):
                _fetch_kaggle(rf, part)
            else:
                _fetch_http(rf, part, have)
        except Exception as e:   # network errors of any flavour: wait and resume
            wait = min(300, 10 * 2 ** (attempt - 1))
            log.warning("download %s interrupted (%s: %s); retry %d/%d in %ds", rf.filename,
                        type(e).__name__, str(e)[:80], attempt, max_retries, wait)
            time.sleep(wait)
    size = part.stat().st_size if part.exists() else 0
    if size != rf.size_bytes:
        raise OSError(f"{rf.filename}: got {size} bytes, expected {rf.size_bytes}")
    if verify and rf.sha256:
        digest = sha256_of(part)
        if digest != rf.sha256:
            part.unlink()
            raise OSError(f"{rf.filename}: SHA-256 mismatch (file deleted, rerun to refetch)")
    os.replace(part, dest)
    return dest


def _fetch_http(rf: RemoteFile, part: Path, have: int) -> None:
    """Plain HTTPS download that continues a partial file with a Range request (used for MAST)."""
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(rf.url, headers=headers, stream=True, timeout=(30, 120)) as r:
        if have and r.status_code == 200:   # server ignored Range: start over
            have = 0
            part.unlink(missing_ok=True)
        r.raise_for_status()
        with open(part, "ab" if have else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)


def _fetch_kaggle(rf: RemoteFile, part: Path) -> None:
    """Download one dataset file with kagglehub, then move it out of kagglehub's cache.

    kagglehub keeps its own partial file in the cache and resumes it, so a
    retry after a dropped connection does not start from zero.
    """
    import kagglehub

    dataset, _, rel = rf.url[len("kaggle://"):].partition("/")
    owner_slug = dataset + "/" + rel.split("/", 1)[0]
    rel_path = rel.split("/", 1)[1]
    got = Path(kagglehub.dataset_download(owner_slug, path=rel_path))
    shutil.move(str(got), part)


def fetch_small(url: str, dest: Path, max_mb: float = 200.0, dry_run: bool = False) -> Path:
    """Fetch a small auxiliary file (benchmark light curves, filter curves), size-checked first."""
    dest = Path(dest)
    if dest.exists():
        return dest
    head = requests.head(url, allow_redirects=True, timeout=60)
    size = int(head.headers.get("content-length", 0))
    if size / 1e6 > max_mb:
        raise OSError(f"{url} is {size/1e6:.0f} MB, above the {max_mb} MB limit for auxiliary files")
    if dry_run:
        log.info("[dry-run] would fetch %s (%.2f MB)", url, size / 1e6)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    os.replace(tmp, dest)
    log.info("fetched %s (%.2f MB)", dest.name, len(r.content) / 1e6)
    return dest
