#!/usr/bin/env python
"""Stage 07 - depth -> brightness temperature -> bare rock vs heat redistribution."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockyeye import pipeline  # noqa: E402
from rockyeye.common import PROJECT_ROOT, load_config, setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML config, e.g. configs/lhs1140b_sim.yaml")
    ap.add_argument("--visits", nargs="*", help="subset of visits (default: all in the config)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    a = ap.parse_args()
    cfg = load_config(a.config)
    setup_logging(PROJECT_ROOT / "logs" / f"{cfg['name']}_07_physics.log")
    pipeline.stage_physics(cfg, a.force)

if __name__ == "__main__":
    main()
