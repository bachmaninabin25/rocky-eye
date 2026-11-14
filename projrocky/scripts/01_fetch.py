#!/usr/bin/env python
"""Stage 01 - list the raw files and their sizes. Downloads only with --download.

The default is a dry run: nothing is written except the small manifest/listing."""
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
    ap.add_argument("--download", action="store_true", help="actually download everything (needs the disk space)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    setup_logging(PROJECT_ROOT / "logs" / f"{cfg['name']}_01_fetch.log")
    pipeline.stage_fetch(cfg, a.visits, dry_run=not a.download)

if __name__ == "__main__":
    main()
