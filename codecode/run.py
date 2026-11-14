"""Your control panel for the rebuild.

    python run.py status                 which ingredients pass right now
    python run.py check 04               test YOUR ingredient 04 (Tier 1: 04 05 06 09, Tier 2: 03 07 08)
    python run.py check 06 --fast        skip the slow MCMC test
    python run.py check 04 --ref         run the same tests on the reference (to see them pass)
    python run.py demo                   run the whole mini-pipeline on reference code
    python run.py demo --mine 04,05      ...with YOUR 04 and 05 plugged in
    python run.py demo --mine all        ...with all four of your Tier 1 pieces

Run from this folder (codecode/) with the rockyeye conda env active:
    conda activate rockyeye
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIER1 = {"04": "04_eclipse", "05": "05_systematics", "06": "06_fit", "09": "09_physics"}
TIER2 = {"03": "tier2/03_photometry", "07": "tier2/07_diagnostics", "08": "tier2/08_injection"}


def _pytest(folder: str, mine: str, fast: bool, quiet: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ, ROCKY_MINE=mine)
    cmd = [sys.executable, "-m", "pytest", folder, "-p", "no:cacheprovider", "-q"]
    if fast:
        cmd += ["-m", "not slow"]
    if quiet:
        cmd += ["--no-header", "-rN"]
    return subprocess.run(cmd, cwd=HERE, env=env, capture_output=quiet, text=True)


def check(nn: str, fast: bool, ref: bool) -> int:
    nn = nn.zfill(2)
    if nn in TIER1:
        mine = "" if ref else nn
        print(f"== ingredient {nn}: testing {'the REFERENCE' if ref else 'YOUR'} {TIER1[nn]}")
        return _pytest(TIER1[nn], mine, fast).returncode
    if nn in TIER2:
        if ref:
            print("Tier 2 tests already compare your exercise against the reference.")
        print(f"== Tier 2 exercise {nn}: {TIER2[nn]}")
        return _pytest(TIER2[nn], "", fast).returncode
    print(f"unknown ingredient {nn}; choose from {sorted(TIER1) + sorted(TIER2)}")
    return 2


def status() -> None:
    print("ingredient                  tests")
    for nn, folder in sorted({**TIER1, **TIER2}.items()):
        r = _pytest(folder, nn if nn in TIER1 else "", fast=True, quiet=True)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        mark = "DONE " if r.returncode == 0 else "todo "
        print(f"  {mark} {nn} {folder:<24} {last}")
    print("\n(fast mode: the slow MCMC test in 06 is skipped here; run `python run.py check 06` for it)")


def demo(mine: str) -> None:
    """The whole mini-pipeline: synthetic data -> best fit -> MCMC -> physics verdict."""
    os.environ["ROCKY_MINE"] = mine
    sys.path.insert(0, str(HERE))
    import numpy as np

    import _deps
    from conftest import GJ, PRIORS, synthetic_light_curve

    print("pieces:", ", ".join(f"{nn}={_deps.who(nn)}" for nn in TIER1))
    ecl, fitm, phys = _deps.get("04"), _deps.get("06"), _deps.get("09")
    orbit = ecl.Orbit(**GJ)

    print("\n1. two synthetic visits with a KNOWN 150 ppm eclipse (truth made with batman)")
    visits = [fitm.make_visit(f"v{k}", synthetic_light_curve(t_mid=2460864.9 + 23.5 * k, seed=k), "exp+x")
              for k in range(2)]
    fit = fitm.EclipseFit(visits, orbit, PRIORS)
    print("\n2. best fit (differential evolution) and detection BIC")
    ml = fit.max_likelihood()
    flat = fit.max_likelihood(depth_fixed=0.0)
    print(f"   depth {ml['p'][0]:.0f} ppm, timing {ml['p'][1]:+.1f} min, noise "
          f"{', '.join(f'{s:.0f}' for s in ml['sigma_ppm'])} ppm, dBIC(eclipse vs none) = {flat['bic'] - ml['bic']:.0f}")
    print("\n3. MCMC (2000 steps, ~1 minute)")
    res = fit.run_mcmc(ml["p"], n_walkers=32, n_steps=2000)
    d, e = res["depth_median"], res["depth_std"]
    print(f"   depth = {d:.0f} +/- {e:.0f} ppm   (truth 150: off by {(d - 150) / e:+.1f} sigma)")

    print("\n4. physics: bare rock or atmosphere?")
    rng = np.random.default_rng(0)
    band = phys.Band()
    teff, a_rs, rp = rng.normal(3369, 50, 4000), rng.normal(17.05, 0.4, 4000), rng.normal(0.0318, 0.0007, 4000)
    i_star = phys.stellar_intensity(band, teff)
    pred = {name: 1e6 * phys.depth_from_temperature(band, phys.dayside_temperature(teff, a_rs, 0.0, eps), rp, i_star)
            for name, eps in (("bare rock", 0.0), ("thick atmosphere", 1.0))}
    samples = res["flat"][:, 0]
    comp = phys.compare_scenarios(samples, pred, reference="thick atmosphere")
    tb = phys.temperature_from_depth(band, np.median(samples) * 1e-6, 0.0318, phys.stellar_intensity(band, 3369.0))
    print(f"   brightness temperature ~ {float(tb):.0f} K  (bare-rock maximum ~ {phys.dayside_temperature(3369, 17.05):.0f} K)")
    for name, c in comp.items():
        print(f"   {name:<17} predicts {pred[name].mean():.0f} ppm, tension {c['tension_sigma']:+.1f} sigma, "
              f"B vs atmosphere = {c['bayes_factor_vs_ref']:.3g} ({phys.interpret_bayes_factor(c['bayes_factor_vs_ref'])})")
    print("\n(the synthetic 150 ppm is not tuned to either scenario; the point is that the chain runs end to end)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("nn")
    c.add_argument("--fast", action="store_true")
    c.add_argument("--ref", action="store_true")
    sub.add_parser("status")
    d = sub.add_parser("demo")
    d.add_argument("--mine", default="")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(check(a.nn, a.fast, a.ref))
    if a.cmd == "status":
        status()
    if a.cmd == "demo":
        demo(a.mine)


if __name__ == "__main__":
    main()
