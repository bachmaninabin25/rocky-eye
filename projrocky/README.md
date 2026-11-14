# projrocky — RockyEye reference pipeline

Raw JWST/MIRI 15 µm eclipse data → eclipse depth → dayside temperature → *bare rock or atmosphere?*

Results and method: [REPORT.md](REPORT.md). How the code works: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).
What is left to run: [../HANDOFF.md](../HANDOFF.md).

## Layout

```
projrocky/
  rockyeye/              one module per ingredient (same numbers as SPEC.md)
    s01_fetch.py         data fetcher + dry-run (Kaggle via kagglehub, MAST via HTTPS)
    s02_calibrate.py     official jwst stage 1–2 runner, streaming + chunked
    s03_photometry.py    exact-aperture photometry → light curve
    s04_eclipse.py       batman eclipse model + analytic cross-check
    s05_systematics.py   ramp / drift / centroid / width / background models
    s06_fit.py           marginal likelihood + emcee (HDF5-checkpointed)
    s07_diagnostics.py   autocorrelation, split R-hat, ESS, Geweke
    s08_injection.py     injection-recovery (pixel level and light-curve level)
    s09_physics.py       depth → brightness temperature → scenario comparison
    s10_compare.py       model comparison, benchmarks, figures, tables
    pipeline.py          stage orchestration + caching (what the scripts call)
    common.py            config, paths, cache keys, logging
  scripts/               one thin CLI per stage: 01_fetch … 08_report, validate
  configs/               lhs1140b_sim.yaml (simulation), gj3929b.yaml (real data)
  tests/                 pytest suite, synthetic data only (no downloads)
  data/                  raw + intermediate data, never committed
  products/<config>/     cached stage outputs, figures, tables
  logs/
```

## Setup

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate rockyeye
make test                               # ~20 s, needs no data
```

## Running

Stages 01–02 (raw ramps → stamps) run on Kaggle; raw data never touch the laptop.

```bash
make notebooks                    # regenerate notebooks/kaggle_{lhs1140b_sim,gj3929b}.ipynb
# run them on Kaggle (see ../HANDOFF.md), download the outputs, then:
make import FROM=~/Downloads/<kaggle output>
make sim            # stages 03-08 on the simulated LHS 1140 b stamps
make validate       # pixel-level injection gate on the simulated frames
make real           # GJ 3929 b - refuses to run until the gate has passed
make truth          # after Nov 18, 2026: compare with the official injected depth
make synthetic      # end-to-end check on synthetic stamps with a known 100 ppm eclipse
make dry-run        # sizes of both raw datasets; downloads nothing
```

Every script takes `--config`. Each stage stores a hash of the config sections it depends on next
to its output; change a relevant value and only the affected stages rerun (`--force` overrides).
MCMC chains are checkpointed to HDF5 every step, so rerunning continues a chain. The Makefile caps
numerical libraries at 4 threads (half the cores).
