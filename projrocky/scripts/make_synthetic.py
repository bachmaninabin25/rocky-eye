#!/usr/bin/env python
"""Generate the synthetic stamps (known eclipse) that stand in for stage 02 in the laptop verification."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockyeye import synth  # noqa: E402
from rockyeye.common import PROJECT_ROOT, atomic_write_json, load_config, product_dir, setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/synthetic.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    setup_logging(PROJECT_ROOT / "logs" / f"{cfg['name']}_make.log")
    truth = synth.write_stamps(cfg, a.force)
    if truth:
        atomic_write_json(product_dir(cfg, "stage02") / "truth.json", {"config": cfg["truth"], "visits": truth})


if __name__ == "__main__":
    main()
