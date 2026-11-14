"""Ingredient 02 — stage 1-2 runner (official ``jwst`` pipeline, config-driven).

Stage 1 (``Detector1Pipeline``) turns raw up-the-ramp reads into a count rate
per pixel per integration ("rateints"). Stage 2 (``Image2Pipeline``) flat-fields
those ("calints"). We do not re-implement either: the config only chooses which
official steps run and with which parameters.

Disk is the constraint on a laptop: the raw simulated set is 199 GB. So the
runner *streams*: for each segment it downloads the raw file, calibrates it,
keeps a small cut-out ("stamp") around the star, and deletes the big files
before touching the next segment. The stamp is all later stages need, and it
lets photometry be redone with any aperture in seconds.

Resumable by construction: a segment whose stamp exists (and whose config hash
matches) is skipped, so an interrupted overnight run continues where it stopped.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from .common import (Timer, atomic_write_json, config_hash, data_dir, is_cached, log,
                     mark_cached, product_dir, read_json, resolve)
from . import s01_fetch


def setup_crds(cfg: dict) -> None:
    """Point the jwst pipeline at a local reference-file cache inside the project."""
    crds = resolve(cfg["paths"].get("crds", "data/crds_cache"))
    crds.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CRDS_PATH", str(crds))
    os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")
    if cfg.get("stage1", {}).get("crds_context"):
        os.environ["CRDS_CONTEXT"] = cfg["stage1"]["crds_context"]


def split_uncal(uncal: Path, chunk_ints: int, work: Path) -> list[Path]:
    """Cut a raw segment into files of at most ``chunk_ints`` integrations.

    Why: stage 1 holds several float copies of the whole ramp cube in memory.
    A 2 GB segment (271 integrations x 45 groups x 256 x 256) needs well over
    16 GB that way; 50 integrations at a time needs a few GB. A chunk is
    written exactly like an official segment file: same headers, with the
    INTSTART/INTEND keywords saying which integrations of the exposure it holds.
    """
    from astropy.io import fits

    work.mkdir(parents=True, exist_ok=True)
    out = []
    with fits.open(uncal, memmap=False) as h:          # .section below reads lazily, slice by slice
        n = h["SCI"].header["NAXIS4"]
        first = h[0].header["INTSTART"]
        if not chunk_ints or n <= chunk_ints:
            return [uncal]
        n_chunks = int(np.ceil(n / chunk_ints))
        edges = np.linspace(0, n, n_chunks + 1).round().astype(int)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            path = work / uncal.name.replace("_mirimage_uncal.fits", f"-c{k:02d}_mirimage_uncal.fits")
            hdus = [fits.PrimaryHDU(header=h[0].header.copy())]
            hdus[0].header["INTSTART"] = first + a
            hdus[0].header["INTEND"] = first + b - 1
            for ext in h[1:]:
                if ext.name in ("SCI", "REFOUT"):
                    # .section reads just these integrations and applies the unsigned-16-bit
                    # offset (BZERO), so the chunk is written as proper uint16 again
                    hdu = fits.ImageHDU(data=ext.section[a:b], header=ext.header.copy(), name=ext.name)
                elif ext.name == "GROUP":                      # per-integration (or per-group) rows
                    per = max(len(ext.data) // n, 1)
                    hdu = fits.BinTableHDU(data=ext.data[a * per:b * per], header=ext.header.copy(), name="GROUP")
                else:                                          # INT_TIMES (whole exposure) and ASDF: unchanged
                    hdu = ext.copy()
                hdus.append(hdu)
            fits.HDUList(hdus).writeto(path, overwrite=True)
            out.append(path)
    return out


def run_stage1(uncal: Path, out_dir: Path, steps: dict) -> Path:
    """Detector1Pipeline: uncal ramps -> rateints (DN/s per integration)."""
    from jwst.pipeline import Detector1Pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    Detector1Pipeline.call(str(uncal), steps=steps or {}, save_results=True, output_dir=str(out_dir),
                           save_calibrated_ramp=False)
    out = out_dir / uncal.name.replace("_uncal.fits", "_rateints.fits")
    if not out.exists():
        raise FileNotFoundError(f"stage 1 did not produce {out}")
    # Only rateints is wanted. Remove the all-integration average and any corrected-ramp
    # file a step may have saved (0.8 GB per chunk): leaving them fills the disk.
    for suffix in ("_rate.fits", "_ramp.fits", "_jump.fits"):
        (out_dir / uncal.name.replace("_uncal.fits", suffix)).unlink(missing_ok=True)
    return out


def run_stage2(rateints: Path, out_dir: Path, steps: dict) -> Path:
    """Image2Pipeline: rateints -> calints (flat-fielded; flux calibration is optional)."""
    from jwst.pipeline import Image2Pipeline

    Image2Pipeline.call(str(rateints), steps=steps or {}, save_results=True, output_dir=str(out_dir))
    out = out_dir / rateints.name.replace("_rateints.fits", "_calints.fits")
    if not out.exists():
        raise FileNotFoundError(f"stage 2 did not produce {out}")
    return out


def find_star(image: np.ndarray, guess: tuple[float, float] | None = None, box: int = 40):
    """Rough star position (x, y) on a median image: brightest 3x3-smoothed pixel.

    The border is ignored, and if ``guess`` is given only a box around it is searched.
    """
    from scipy.ndimage import uniform_filter

    img = np.nan_to_num(image - np.nanmedian(image))
    sm = uniform_filter(img, 3)
    mask = np.zeros_like(sm, dtype=bool)
    ny, nx = sm.shape
    if guess is None:
        mask[8:ny - 8, 8:nx - 8] = True
    else:
        gx, gy = int(round(guess[0])), int(round(guess[1]))
        mask[max(gy - box, 0):gy + box + 1, max(gx - box, 0):gx + box + 1] = True
    sm = np.where(mask, sm, -np.inf)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    return float(ix), float(iy)


def select_times(number: np.ndarray, t_all: np.ndarray, i0, i1, n: int) -> np.ndarray:
    """Mid-integration times of the ``n`` integrations a file holds.

    The INT_TIMES table may list more integrations than the file contains (a
    whole segment, or the whole exposure). Rows are picked by integration
    number (INTSTART..INTEND), never by position; if the header lacks those
    keywords the table must match the data one-to-one.
    """
    if i0 is not None and i1 is not None:
        t = t_all[(number >= i0) & (number <= i1)]
        if len(t) == n:
            return t
    if len(t_all) == n:
        return t_all
    raise ValueError(f"cannot match {n} integrations to the INT_TIMES table "
                     f"({len(t_all)} rows, numbers {number.min()}..{number.max()}, INTSTART={i0}, INTEND={i1})")


def make_stamp(calints: Path, centre: tuple[float, float] | None, half: int,
               guess: tuple[float, float] | None = None) -> dict:
    """Cut a (2*half+1)^2 stamp around the star out of every integration.

    Returns times (BJD_TDB), the science and error cubes, a bad-pixel mask and
    the stamp's origin in full-frame pixel coordinates.
    """
    from jwst import datamodels

    with datamodels.open(str(calints)) as m:
        sci = np.array(m.data, dtype=np.float32)
        err = np.array(m.err, dtype=np.float32)
        dq = np.array(m.dq)
        t = select_times(np.array(m.int_times["integration_number"]),
                         np.array(m.int_times["int_mid_BJD_TDB"], dtype=float) + 2400000.5,
                         m.meta.exposure.integration_start, m.meta.exposure.integration_end, sci.shape[0])
        unit = m.meta.bunit_data
        pixar_sr = m.meta.photometry.pixelarea_steradians or np.nan
    med = np.nanmedian(sci, axis=0)
    if centre is None:
        centre = find_star(med, guess)
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    ny, nx = med.shape
    y0, y1 = max(cy - half, 0), min(cy + half + 1, ny)
    x0, x1 = max(cx - half, 0), min(cx + half + 1, nx)
    do_not_use = (dq[:, y0:y1, x0:x1] & 1) > 0
    return {"time": t, "sci": sci[:, y0:y1, x0:x1], "err": err[:, y0:y1, x0:x1],
            "bad": do_not_use, "origin": np.array([x0, y0]), "centre": np.array(centre),
            "median_frame": med.astype(np.float32), "unit": str(unit), "pixar_sr": float(pixar_sr)}


def stamp_path(cfg: dict, rf: s01_fetch.RemoteFile) -> Path:
    d = product_dir(cfg, "stage02") / rf.visit
    d.mkdir(parents=True, exist_ok=True)
    return d / rf.filename.replace("_uncal.fits", "_stamp.npz")


def stage_key(cfg: dict) -> str:
    return config_hash(cfg.get("stage1"), cfg.get("stage2"), cfg["stamp"])


def process_segment(cfg: dict, rf: s01_fetch.RemoteFile, dry_run: bool = False) -> Path:
    """Fetch -> stage 1 -> stage 2 -> stamp -> clean up, for one raw segment."""
    out = stamp_path(cfg, rf)
    key = stage_key(cfg)
    if is_cached(out, key):
        return out
    if dry_run:
        have = Path(rf.local_path).exists()
        log.info("[dry-run] %s %s: %s, then stage 1-2 -> %s", rf.visit, rf.filename,
                 "raw already on disk" if have else f"download {rf.size_gb:.2f} GB", out.name)
        return out
    setup_crds(cfg)
    st = cfg["stamp"]
    work = data_dir(cfg) / "work" / rf.visit
    keep = cfg.get("keep", {})
    uncal = s01_fetch.download(rf, min_free_gb=cfg["source"].get("min_free_gb", 8.0))
    # the star position is fixed per visit: measured on the first chunk, reused afterwards
    cfile = product_dir(cfg, "stage02") / rf.visit / "centre.json"
    centre = tuple(read_json(cfile)["centre"]) if cfile.exists() else None
    chunks = split_uncal(uncal, cfg.get("stage1", {}).get("chunk_ints", 50), work)
    t1 = t2 = 0.0
    parts = []
    for chunk in chunks:
        with Timer() as tm:
            rateints = run_stage1(chunk, work, cfg.get("stage1", {}).get("steps", {}))
        t1 += tm.dt
        with Timer() as tm:
            calints = run_stage2(rateints, work, cfg.get("stage2", {}).get("steps", {}))
        t2 += tm.dt
        part = make_stamp(calints, centre, st["half_size"], guess=st.get("guess"))
        if centre is None:
            centre = tuple(float(c) for c in part["centre"])
            atomic_write_json(cfile, {"centre": centre, "from": rf.filename})
        parts.append(part)
        if chunk != uncal:
            chunk.unlink(missing_ok=True)
        for product, kind in ((rateints, "rateints"), (calints, "calints")):
            if keep.get(kind, False):
                dest = data_dir(cfg) / kind / rf.visit
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(product), dest / product.name)
            else:
                product.unlink(missing_ok=True)
    stamp = {k: np.concatenate([p[k] for p in parts]) for k in ("time", "sci", "err", "bad")}
    for k in ("origin", "centre", "median_frame", "unit", "pixar_sr"):
        stamp[k] = parts[0][k]
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **stamp, filename=rf.filename, visit=rf.visit, segment=rf.segment)
    os.replace(tmp, out)
    mark_cached(out, key)
    if not keep.get("raw", False) and not s01_fetch.is_input_file(cfg, rf):
        uncal.unlink(missing_ok=True)
    log.info("%s %s done: %d ints in %d chunks | stage1 %.0fs, stage2 %.0fs", rf.visit, rf.filename,
             len(stamp["time"]), len(chunks), t1, t2)
    return out


def process(cfg: dict, visits: list[str] | None = None, dry_run: bool = False,
            max_segments: int | None = None, max_hours: float | None = None,
            continue_on_error: bool = False, after_segment=None) -> list[Path]:
    """Stream every segment of the requested visits through stages 1-2.

    Each finished segment is written to disk (atomically) before the next one
    starts, so a crash or a time limit never costs more than the segment in
    progress, and a rerun skips everything already done.

    max_hours          stop cleanly once the *next* segment would not fit in the
                       budget (a Kaggle session lasts 12 h).
    continue_on_error  log a failed segment and carry on instead of aborting.
    after_segment      callback(n_done, n_todo, failed) after every segment,
                       e.g. to rewrite a progress index.
    """
    import time

    t_start = time.time()
    files = s01_fetch.list_files(cfg, visits)
    key = stage_key(cfg)
    todo = [f for f in files if not is_cached(stamp_path(cfg, f), key)]
    log.info("%d segments total, %d already calibrated, %d to do (%.1f GB to download)",
             len(files), len(files) - len(todo), len(todo),
             sum(f.size_gb for f in todo if not Path(f.local_path).exists()))
    if max_segments is not None:
        todo = todo[:max_segments]
    if dry_run:
        return [process_segment(cfg, rf, dry_run=True) for rf in todo]
    # Download ahead: while segment i is being calibrated, segment i+1 is fetched in a
    # background thread, so the network and the CPU are both busy. At most two raw
    # files are on disk at any time. (With attached inputs the "download" returns at once.)
    from concurrent.futures import ThreadPoolExecutor

    min_free = cfg["source"].get("min_free_gb", 8.0)
    done, failed, longest = [], [], 0.0
    with ThreadPoolExecutor(max_workers=1) as pool:
        nxt = pool.submit(s01_fetch.download, todo[0], min_free) if todo else None
        for i, rf in enumerate(todo, 1):
            elapsed_h = (time.time() - t_start) / 3600.0
            if max_hours is not None and elapsed_h + 1.5 * longest / 3600.0 > max_hours:
                log.warning("time budget of %.1f h reached after %.1f h: %d segments left, rerun to continue",
                            max_hours, elapsed_h, len(todo) - i + 1)
                break
            t_seg = time.time()
            try:
                nxt.result()                              # raises here if the download failed
                nxt = pool.submit(s01_fetch.download, todo[i], min_free) if i < len(todo) else None
                log.info("[%d/%d] %s %s", i, len(todo), rf.visit, rf.filename)
                done.append(process_segment(cfg, rf))
            except Exception as e:                        # noqa: BLE001 - reported, optionally re-raised
                if not continue_on_error:
                    raise
                log.error("FAILED %s %s: %s: %s", rf.visit, rf.filename, type(e).__name__, e)
                failed.append(rf.filename)
                if nxt is None or nxt.done():
                    nxt = pool.submit(s01_fetch.download, todo[i], min_free) if i < len(todo) else None
            longest = max(longest, time.time() - t_seg)
            if after_segment is not None:
                after_segment(len(done), len(todo), failed)
    if failed:
        log.error("%d segments failed: %s", len(failed), ", ".join(failed))
    return done


def load_visit_stamps(cfg: dict, visit: str) -> dict:
    """Concatenate all cached segment stamps of a visit, in time order."""
    files = s01_fetch.list_files(cfg, [visit])
    key = stage_key(cfg)
    parts, missing = [], []
    for rf in files:
        p = stamp_path(cfg, rf)
        if is_cached(p, key):
            parts.append(np.load(p, allow_pickle=False))
        else:
            missing.append(rf.filename)
    if missing:
        raise FileNotFoundError(f"{visit}: {len(missing)} segments not calibrated yet, e.g. {missing[0]}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("time", "sci", "err", "bad")}
    out["origin"] = parts[0]["origin"]
    out["centre"] = parts[0]["centre"]
    out["median_frame"] = parts[0]["median_frame"]
    out["pixar_sr"] = float(parts[0]["pixar_sr"])
    out["unit"] = str(parts[0]["unit"])
    out["segment_starts"] = np.cumsum([0] + [len(p["time"]) for p in parts])[:-1]
    out["n_segments"] = len(parts)
    return out
