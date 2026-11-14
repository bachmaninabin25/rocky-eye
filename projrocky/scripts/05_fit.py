#!/usr/bin/env python
"""Stage 05 - emcee fits: joint over all visits, then each visit alone.

Chains are checkpointed to HDF5 every step; rerun to resume after an interruption."""
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
    ap.add_argument("--force", action="store_true", help="discard cached chains and start over")
    a = ap.parse_args()
    cfg = load_config(a.config)
    setup_logging(PROJECT_ROOT / "logs" / f"{cfg['name']}_05_fit.log")
    pipeline.stage_fit(cfg, a.visits, a.force)

if __name__ == "__main__":
    main()
