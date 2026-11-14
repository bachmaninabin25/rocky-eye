"""Picks, for every ingredient, either YOUR file or the reference (projrocky) version.

You never need to edit this. It reads the environment variable ROCKY_MINE, a
comma-separated list of ingredient numbers that should use your code:

    ROCKY_MINE=04        -> your eclipse.py, reference for everything else
    ROCKY_MINE=04,05,06  -> your 04, 05 and 06
    ROCKY_MINE=all       -> all four of yours (the full rebuild)
    (unset)              -> everything is the reference

run.py sets it for you. Each ingredient is loaded by file path because folder
names like "04_eclipse" are not valid Python package names.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent / "projrocky"
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

INGREDIENTS = {
    "04": ("04_eclipse", "eclipse.py"),
    "05": ("05_systematics", "systematics.py"),
    "06": ("06_fit", "fit.py"),
    "09": ("09_physics", "physics.py"),
}

_cache: dict[str, object] = {}


def mine() -> set[str]:
    raw = os.environ.get("ROCKY_MINE", "").strip()
    if raw == "all":
        return set(INGREDIENTS)
    return {x.strip().zfill(2) for x in raw.split(",") if x.strip()}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get(nn: str):
    """Module for ingredient ``nn`` ("04", "05", "06" or "09"): yours or the reference."""
    nn = nn.zfill(2)
    use_mine = nn in mine()
    key = f"{nn}:{'mine' if use_mine else 'ref'}"
    if key not in _cache:
        if use_mine:
            folder, fname = INGREDIENTS[nn]
            root = Path(os.environ.get("ROCKY_MINE_DIR", HERE))
            _cache[key] = _load(f"mine_{nn}", root / folder / fname)
        else:
            _cache[key] = _load(f"ref_{nn}", HERE / "_reference" / f"ref_{nn}.py")
    return _cache[key]


def who(nn: str) -> str:
    return "mine" if nn.zfill(2) in mine() else "reference"
