"""Shared plumbing: config loading, paths, caching, logging.

Nothing scientific lives here. Every stage script does the same three things:
load a YAML config, work out where its products go, and skip work whose cached
output already matches the config that produced it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("rockyeye")


def setup_logging(logfile: str | Path | None = None, level=logging.INFO) -> None:
    """Log to the terminal and (optionally) append to a file, so overnight runs leave a trail."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("rockyeye")
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _deep_update(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Read a YAML config. `extends: other.yaml` layers this file over another one."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "extends" in cfg:
        parent = load_config(path.parent / cfg.pop("extends"))
        cfg = _deep_update(parent, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(p: str | Path) -> Path:
    """Paths in configs are relative to projrocky/."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def data_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["data"])


def product_dir(cfg: dict, stage: str = "") -> Path:
    d = resolve(cfg["paths"]["products"]) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_hash(*sections) -> str:
    """Short, stable fingerprint of the config sections a product depends on."""
    blob = json.dumps(sections, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def is_cached(path: Path, key: str) -> bool:
    """A product is fresh if it exists and its sidecar records the same config hash."""
    side = Path(str(path) + ".key")
    return path.exists() and side.exists() and side.read_text().strip() == key


def mark_cached(path: Path, key: str) -> None:
    Path(str(path) + ".key").write_text(key)


def atomic_write_json(path: Path, obj) -> None:
    """Write JSON via a temp file so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def free_gb(path: Path) -> float:
    import shutil

    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
