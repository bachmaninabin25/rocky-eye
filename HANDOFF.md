# RockyEye — HANDOFF (final, 2026-11-14)

**The project is done.** Raw JWST data → eclipse depth → verdict, validated, cross-checked and written
up. What's left is the learning phase.

## The result
**GJ 3929 b: eclipse depth 126 ± 20 ppm** (4 MIRI 15 µm eclipses, joint fit, ΔBIC = 25, converged,
the four eclipses agree: χ² = 2.3 for 3 dof). Dayside brightness temperature **~674 K**, against
737 K for a perfectly dark bare rock (R = 0.91 ± 0.09).
- Agrees with the official Rocky Worlds reduction (142 ± 19 ppm, −0.6σ) and Xue et al. 2025 (−0.5σ).
- **Thick (Venus-like, 100 mbar) atmospheres ruled out at ~4.5σ.** Bare rock or a thin atmosphere both fit.
- Full write-up: `projrocky/REPORT.md`. Numbers: `projrocky/products/gj3929b/stage08/tables.md`.

## How it was validated
1. **Synthetic stamps** (known 100 ppm eclipse): recovered 117 ± 15 (+1.1σ).
2. **Pixel-level gate on STScI's simulated LHS 1140 b frames:** 0 / 80 / 160 ppm planted →
   0.1 / 81.6 / 160.0 recovered. **PASSED.**
3. **Injection-recovery on GJ 3929 b's own noise:** at ≥150 ppm the error bars are honest
   (scatter 0.97–1.07× the quoted error); small low bias of 6–11 ppm.
4. **Eureka! cross-check:** photometry agrees to 59 ppm RMS on the same frames.

## Known issues (documented, don't affect the GJ result)
- **Simulated LHS 1140 b's own eclipse fit** (separate from the gate, which passed): 41 +15/−23 ppm,
  no detection, timing ~131 min off phase 0.5 consistently across visits, joint chain not converged
  (R̂ 1.028). The consistent timing offset suggests the sim config's ephemeris (t0/period in
  `configs/lhs1140b_sim.yaml`) doesn't match what STScI used. Unresolved; doesn't affect GJ 3929 b.
- **Eclipse 4** (split exposure) is essentially noise (−36 ± 124 ppm); the result leans on eclipses 1–3.
- **Below ~100 ppm the method is unreliable** (scatter 2.6–3.6× the quoted error; timing can lock on
  the wrong place). Never quote an upper limit from this pipeline.

## What's left
1. ~~Official-truth comparison (`make truth`)~~: **skipped by choice.** The slot (`truth:` in
   `configs/lhs1140b_sim.yaml`) and `scripts/revalidate_truth.py` stay, in case it's ever wanted.
2. **Learning phase:** `PROGRESS.md`, Phase E. Start at `learncode/00_START_HERE.md`.
3. Optional: empty `_to_delete/` (1.7 GB of superseded and cache files, list in its own README).

## Rerun anything
```bash
cd ~/rockyproject/projrocky && conda activate rockyeye
make test            # 58 tests
make real            # stages 03-08 on GJ 3929 b from the stored stamps (cached: instant if nothing changed)
```
Stages 01–02 (raw → stamps) run on Kaggle via `projrocky/notebooks/`; instructions are in each
notebook's first cell. The previous, longer handoff is at `_to_delete/HANDOFF_2026-11-13_old.md`.
