# RockyEye — Progress
First unchecked box = where I am. Each phase needs the one above it.
("Phase" = building/running the project. "Tier" = how deep I learn each piece: 1 rebuild, 2 exercise, 3 skim.)

## Phase A — Build projrocky (Fable / Opus)
- [x] Pipeline: 10 ingredient modules, stage scripts, configs (`projrocky/rockyeye/`, `scripts/`, `configs/`)
- [x] Stage 1–2 proven on one real segment of the simulated data (locally, then raw data deleted)
- [x] Synthetic end-to-end check (known 100 ppm eclipse): stages 03–08 + validation gate — see REPORT.md §5.1
- [x] Tests: `make test` and `make test-all` pass
- [x] Two Kaggle notebooks for stages 01–02 (`projrocky/notebooks/`), per-segment saving, resumable
- [x] Eureka! cross-check of photometry on the same frames (59 ppm RMS agreement)
- [x] Stellar parameters checked against Kemmer+22 / Beard+22; official HLSP benchmark fetched
- [x] `docs/HOW_IT_WORKS.md`; `REPORT.md` template (methods done, results as placeholders)

## Phase B — Get the stamps (Kaggle; nothing raw on the Mac)
- [x] Run `kaggle_gj3929b.ipynb` (25/25 segments, 4.5 h, one session; imported 2026-11-13)
- [x] Run `kaggle_lhs1140b_sim.ipynb` (106/106 segments, 2 sessions; imported 2026-11-14)
- [x] Both `stamps_index.json` show every segment done, no failures
- [x] `make import FROM=<download>` for each; importer reports every visit complete and matching config

## Phase C — Validate on the simulation
- [x] `make sim` (stages 03–08 on simulated LHS 1140 b) — joint chain hit step cap unconverged (no detectable eclipse)
- [x] `make validate` — pixel-level injection gate PASSED 2026-11-14 (0/80/160 → 0.1/81.6/160.0 ppm)
- [x] REPORT.md §5.2–5.3 filled in (truth comparison still pending)
- [x] ~~Re-validate vs the official truth after Nov 18~~ — skipped by choice (slot + script kept)

## Phase D — The real result
- [x] `make real` (stages 03–08 on GJ 3929 b) — 2026-11-14: 126 ± 20 ppm, T_b 674 ± 64 K
- [x] REPORT.md §6 results, §7 benchmark discussion, §8 conclusion, abstract
- [x] Real GJ 3929 b result + REPORT.md complete
- [ ] Everything runs on my Mac, outside the build session

## Phase E — Learn it (after exams)
- [x] Kit generated: `learntopic/`, `learncode/`, `codecode/` (start at `learncode/00_START_HERE.md`)

### Topics (3 days, `learntopic/`)
- [ ] Day 1 — Eclipses and thermal emission (run every snippet, do the self-check)
- [ ] Day 2 — Bayesian inference and MCMC (the most important day)
- [ ] Day 3 — JWST detectors and systematics

### Tier 1 — REBUILD (`codecode/`, `python run.py check NN`)
- [ ] 04 eclipse model — notes read (`learncode/04_eclipse.md`)
- [ ] 04 eclipse model — my code passes
- [ ] 05 systematics — notes read
- [ ] 05 systematics — my code passes
- [ ] 06 likelihood + MCMC — notes read
- [ ] 06 likelihood + MCMC — my code passes (incl. the slow MCMC test)
- [ ] 09 physics verdict — notes read
- [ ] 09 physics verdict — my code passes

### Tier 2 — UNDERSTAND + small exercise
- [ ] 07 diagnostics — notes read · `split_rhat` passes
- [ ] 08 injection — notes read · `inject` + `linear_depth` pass
- [ ] 03 photometry — notes read · `centroid_and_width` passes
- [ ] Configs — `learncode/12_configs.md` read; can explain every value in `configs/gj3929b.yaml`

### Tier 3 — SKIM (notes only)
- [ ] `learncode/tier3_plumbing.md`: fetch, calibrate, compare, pipeline, common, scripts, Makefile

### Finish line
- [ ] `python run.py demo --mine all` runs the whole chain on my code only
- [ ] Can explain the project cold: why MCMC, why these noise models, how I know the error bar is honest
- [ ] Public GitHub repo, my rebuild committed piece by piece, README with the headline result + one plot
