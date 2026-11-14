# Spec: RockyEye, an independent JWST reduction of GJ 3929 b

> Paste this whole file into a fresh Fable session opened in this folder.
> Build ONLY `projrocky/` with Fable. The kit (`learntopic/`, `learncode/`, `codecode/`) is generated later on Opus.

## Goal
An independent pipeline from raw JWST MIRI 15 μm eclipse data to a verdict, for rocky exoplanet
GJ 3929 b (Rocky Worlds DDT, program 9235). The question: **bare rock or atmosphere?**

## Constraints
- Runs on a MacBook M2, CPU only. Long jobs must be resumable and safe to run overnight.
- Use the official `jwst` pipeline for stages 1–2. Cross-check against Eureka!, but write my own
  light-curve extraction, systematics model and fitting.
- Always check data size before downloading, and include a dry-run mode.
- Raw data lives in `projrocky/data/` and is never committed (see `.gitignore`).

## Folder structure
```
rockyproject/
  projrocky/    ← full working project. I will never edit this.
  learntopic/   ← 3-day topic notes, independent of this codebase
  learncode/    ← piece-by-piece code notes
  codecode/     ← where I write my own version
  PROGRESS.md   ← checklist across all phases
```

## projrocky milestones
1. **Validate first** on the simulated LHS 1140 b eclipses from the data challenge. Don't touch real
   data until it recovers the injected depth.
2. **Reduce:** raw ramps from MAST → calibrated frames → photometry → light curves.
3. **Systematics:** ramp, drift, centroid and background decorrelation. Compare at least 3 systematics models.
4. **Fit:** `batman` eclipse model plus systematics, with an `emcee` MCMC. Convergence diagnostics,
   corner plots, joint fit of all 4 visits.
5. **Injection-recovery:** plant fake eclipses and measure detection sensitivity.
6. **Physics:** eclipse depth → dayside brightness temperature → bare rock vs heat redistribution,
   with BIC or evidence for model comparison.
7. **Benchmark** against the Rocky Worlds team's published results.

Deliverables: paper-style `REPORT.md`, config-driven cached scripts (one per stage), `HOW_IT_WORKS.md`, tests.
Structure `projrocky/` as the 10 ingredients below (one module per ingredient, same numbering),
so the kit can mirror it exactly.

## learntopic/
Standalone notes that never reference `projrocky` code. Each day: concepts, small standalone code
examples, text diagrams, 2–3 external readings, 5 self-check questions with answers at the bottom.
- `day1_eclipses_and_thermal_emission.md`: eclipse geometry, eclipse depth, brightness temperature,
  heat redistribution, why a bare rock is hotter on the dayside
- `day2_bayesian_inference_mcmc.md`: likelihoods, priors, posteriors, MCMC, convergence, model comparison
- `day3_jwst_detectors_and_systematics.md`: how MIRI records data, ramps, drift, time-series noise sources, decorrelation

## learncode/ + codecode/ ingredients
| # | Ingredient | Gate |
|---|---|---|
| 01 | Data fetcher + dry-run | lists the right files |
| 02 | Stage 1–2 runner (config-driven) | matches reference frames |
| 03 | Aperture photometry → light curve | flux within 0.1% of reference |
| 04 | Eclipse model wrapper | matches the analytic test cases |
| 05 | Systematics models | reaches the reference residual level |
| 06 | Likelihood + `emcee` fit | depth within 1σ of reference |
| 07 | Convergence diagnostics | flags a deliberately bad chain |
| 08 | Injection-recovery | recovers injected depths |
| 09 | Brightness temp + bare rock vs atmosphere | same verdict as reference |
| 10 | Model comparison + report | same conclusion as reference |

## Shared kit rules
**codecode/** — each ingredient folder `NN_<name>/` contains:
- `skeleton.py`: exact signatures, docstrings and TODOs, empty bodies. Compiles and imports from day one. No solutions.
- `test_*.py`: passes on the `projrocky` reference, fails on the empty skeleton. Includes the gate above.
- `hints.md`: 3 tiers per TODO (nudge → approach → near-solution).
- Swappable: `make check N` tests my ingredient N; `make run --mine 01,02` runs the full pipeline
  with my pieces plus reference for the rest.
- **Assembly:** when all ingredients are done, `make run --mine all` must run the complete project
  using only `codecode/`, with zero imports from `projrocky`.

**learncode/NN_<name>.md:** what the piece does, where it sits in the pipeline, key design decisions
and why (including rejected alternatives), annotated excerpts of the real code, gotchas, and how it's tested.

**Model split:** build `projrocky/` with Fable. Generate `learntopic/`, `learncode/` and `codecode/` with Opus.

## Done when
- The simulated recovery passes.
- The real-data depth comes with honest error bars.
- The verdict is stated with its statistical strength.
- It's benchmarked against the published result.
- The full kit exists.
- Everything runs on my Mac outside the build session.
