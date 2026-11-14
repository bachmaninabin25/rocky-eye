#!/usr/bin/env python
"""Re-validate against the official injected values of the simulated LHS 1140 b eclipses.

STScI had not published the truth when this pipeline was built (the Data Challenge
workshop is Nov 16-18, 2026). Once it is public, either fill in the config slot

    truth: {depth_ppm: <value>, dt_min: <value or null>, source: "<where you got it>"}

in configs/lhs1140b_sim.yaml, or pass it on the command line:

    python scripts/revalidate_truth.py --config configs/lhs1140b_sim.yaml --depth-ppm 85 --source "RWDDT workshop slides"

It needs the joint fit (stage 05) to exist. It never refits anything: it only asks
how far the posterior we already have sits from the truth. Works for the synthetic
config too, whose truth is known.

``dt_min`` is the eclipse-time offset from phase 0.5 in minutes, in the convention
of this pipeline (ephemeris from the config). Leave it null if only a depth is given.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockyeye import pipeline  # noqa: E402
from rockyeye.common import atomic_write_json, load_config, product_dir, read_json  # noqa: E402


def check(samples: np.ndarray, truth: float) -> dict:
    """Where does the truth fall in the posterior?"""
    med = float(np.median(samples))
    lo, hi = np.percentile(samples, [15.865, 84.135])
    err = (hi - med) if truth > med else (med - lo)          # the error bar on the truth's side
    q = float(np.mean(samples < truth))                       # posterior quantile of the truth
    return {"truth": truth, "median": med, "minus": float(med - lo), "plus": float(hi - med),
            "offset": med - truth, "offset_sigma": (med - truth) / err, "truth_quantile": q,
            "inside_68": bool(lo <= truth <= hi),
            "inside_95": bool(np.percentile(samples, 2.275) <= truth <= np.percentile(samples, 97.725))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--depth-ppm", type=float, help="true eclipse depth (overrides the config slot)")
    ap.add_argument("--dt-min", type=float, help="true eclipse-time offset from phase 0.5, minutes")
    ap.add_argument("--source", help="where the truth comes from (URL, slide deck, paper)")
    # 2 sigma, not 1: with honest error bars the truth falls outside 1 sigma 32% of the time,
    # so a 1-sigma bar would "fail" a correct pipeline one run in three.
    ap.add_argument("--pass-sigma", type=float, default=2.0, help="pass if |offset| <= this many sigma (default 2)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    truth = dict(cfg.get("truth") or {})
    if a.depth_ppm is not None:
        truth.update(depth_ppm=a.depth_ppm, source=a.source or truth.get("source"))
    if a.dt_min is not None:
        truth["dt_min"] = a.dt_min
    if truth.get("depth_ppm") is None:
        sys.exit(f"No truth yet for {cfg['name']}: fill `truth.depth_ppm` in {a.config} or pass --depth-ppm.")
    if not (product_dir(cfg, "stage05") / "joint_samples.npz").exists():
        sys.exit("No joint fit found: run scripts/05_fit.py first.")

    s = pipeline.load_samples(cfg, "joint")
    out = {"config": cfg["name"], "source": truth.get("source"), "pass_sigma": a.pass_sigma,
           "depth_ppm": check(s["depth_ppm"], float(truth["depth_ppm"]))}
    if truth.get("dt_min") is not None and "dt_min" in s["names"]:
        out["dt_min"] = check(s["flat"][:, s["names"].index("dt_min")], float(truth["dt_min"]))
    # per-eclipse depths against the same truth: do the visits scatter around it as they should?
    summary = product_dir(cfg, "stage08") / "summary.json"
    if summary.exists():
        ours = read_json(summary)["ours"]
        pulls = {k: (v[0] - truth["depth_ppm"]) / v[1] for k, v in ours.items() if not k.startswith("joint")}
        if pulls:
            p = np.array(list(pulls.values()))
            out["per_eclipse"] = {"pulls": pulls, "chi2": float(np.sum(p ** 2)), "n": len(p),
                                  "rms_pull": float(np.sqrt(np.mean(p ** 2)))}
    out["passed"] = bool(abs(out["depth_ppm"]["offset_sigma"]) <= a.pass_sigma)
    path = product_dir(cfg, "validation") / "truth_check.json"
    atomic_write_json(path, out)

    d = out["depth_ppm"]
    print(f"{cfg['name']}: truth = {d['truth']:.1f} ppm ({out['source'] or 'source not given'})")
    print(f"  recovered depth  {d['median']:.1f} +{d['plus']:.1f} -{d['minus']:.1f} ppm")
    print(f"  offset           {d['offset']:+.1f} ppm = {d['offset_sigma']:+.2f} sigma "
          f"(truth at posterior quantile {d['truth_quantile']:.3f})")
    if "dt_min" in out:
        t = out["dt_min"]
        print(f"  eclipse time     {t['median']:.2f} +{t['plus']:.2f} -{t['minus']:.2f} min vs truth {t['truth']:.2f} "
              f"({t['offset_sigma']:+.2f} sigma)")
    if "per_eclipse" in out:
        pe = out["per_eclipse"]
        print(f"  per-eclipse      chi2 = {pe['chi2']:.1f} for {pe['n']} eclipses, rms pull {pe['rms_pull']:.2f} (expect ~1)")
    print(f"  truth inside the 68% interval: {'yes' if d['inside_68'] else 'no'}; inside 95%: {'yes' if d['inside_95'] else 'no'}")
    print(f"  {'PASSED' if out['passed'] else 'FAILED'}: |offset| {'<=' if out['passed'] else '>'} {a.pass_sigma:g} sigma")
    print(f"  written to {path}")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
