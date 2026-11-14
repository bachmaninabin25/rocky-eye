"""Loads the exercise file next to a test (or a private solution dir, if ROCKY_MINE_DIR is set)."""
import importlib.util
import os
from pathlib import Path


def exercise(test_file: str):
    here = Path(test_file).resolve().parent
    root = os.environ.get("ROCKY_MINE_DIR")
    path = (Path(root) / "tier2" / here.name / "exercise.py") if root else here / "exercise.py"
    spec = importlib.util.spec_from_file_location(f"ex_{here.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
