#!/usr/bin/env python
"""Stage 02 - stream raw segments through the official jwst stage 1-2 pipeline.

Downloads one segment at a time, keeps a small stamp around the star, deletes the
big files. Safe to interrupt and rerun: finished segments are skipped."""
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
    ap.add_argument("--dry-run", action="store_true", help="print what would be downloaded/processed")
    ap.add_argument("--max-segments", type=int, help="stop after this many segments")
    a = ap.parse_args()
    cfg = load_config(a.config)
    setup_logging(PROJECT_ROOT / "logs" / f"{cfg['name']}_02_calibrate.log")
    pipeline.stage_fetch(cfg, a.visits, dry_run=True)
    pipeline.stage_calibrate(cfg, a.visits, dry_run=a.dry_run, max_segments=a.max_segments)

if __name__ == "__main__":
    main()
