# tests/ — how the reference was proven correct (`projrocky/tests/`, 12 files, 58 tests)

Run: `make test` (57 fast tests, ~20 s) or `make test-all` (adds a ~1 min MCMC). All synthetic:
no network, no JWST files.

## How to read a test file
Each test is a **claim with evidence**. Read the name, predict how you'd test it, then read the
code. Three kinds recur:
- **Against an independent implementation** (batman vs geometry, our apertures vs photutils,
  analytic likelihood vs brute-force integration). Two methods agreeing is strong evidence.
- **Against a planted truth** (synthetic data with a known depth or noise level).
- **Deliberately broken inputs** (bad chains, a full disk, a failed segment). Tests that the
  alarms fire, not just the happy path.

For your Tier 1 rebuild: read the matching file **before** you start. `codecode/` has its own
tests (same ideas, lean API); the originals show the full version.

## File by file

| File | What it proves |
|---|---|
| `conftest.py` | Shared fixtures: GJ 3929 b orbit, `synthetic_light_curve()` (known 150 ppm eclipse, ramp, slope, centroid term), three synthetic visits, default priors. |
| `test_01_fetch.py` (7) | Lists files in order; dry run downloads nothing; refuses when the disk would fill; resumes partial downloads and verifies SHA-256; Kaggle path uses the official client. |
| `test_02_calibrate.py` (7) | Times matched by integration number (the segment-2 bug); chunk splitting is lossless (the BZERO bug); star finder ignores edges; cache key changes only with calibration settings; a failed segment is skipped and the run resumes; stops inside the time budget. |
| `test_03_photometry.py` (5) | Aperture weights sum to πr² and match photutils; flux within 0.1% of truth; centroid and sky recovered; cosmic rays and bad pixels repaired. |
| `test_04_eclipse.py` (5) | Shape is 0 mid-eclipse and 1 outside; batman matches circle geometry to 1e-6; durations match T14/T23; eclipse at phase 0.5; nothing hidden at transit. |
| `test_05_systematics.py` (4) | Model-name parsing; basis column shapes; the right model leaves 200 ppm white residuals (β ~ 1) while the wrong one leaves red noise; binned RMS falls as 1/√n. |
| `test_06_fit.py` (7) | Parameter layout; prior bounds; **marginal likelihood = brute-force integral**; best fit near the truth and ΔBIC > 10; MCMC recovers the depth within 1σ, converges, and resumes from its saved chain; regression test for the ΔBIC ≈ 600 bug. |
| `test_07_diagnostics.py` (5) | τ of an AR(1) process matches theory; a good chain passes; walkers that disagree, too-short chains, drifting chains and stuck walkers are each flagged. |
| `test_08_injection.py` (3) | Pixel injection scales only the star; fast estimator recovers known depths; the injection grid is unbiased and its error bars are honest. |
| `test_09_physics.py` (7) | Planck vs hand value and Rayleigh-Jeans limit; band is F1500W; brightness-temperature round trip; 737 K bare rock for GJ 3929 b (Xue+25); depth↔temperature inverse; flux→intensity; the verdict picks the true scenario. |
| `test_10_compare.py` (4) | BIC picks the model that generated the data; flat data → simplest model; detection ΔBIC; benchmark arithmetic and tables. |
| `test_truth_slot.py` (4) | The post-Nov-18 re-validation maths: truth inside the posterior passes, far away fails, asymmetric errors use the side facing the truth, the slot is empty until published. |

## Three tests worth reading line by line
1. `test_06_fit.py::test_marginal_likelihood_matches_brute_force`: the whole fit rests on this formula.
2. `test_07_diagnostics.py::test_flags_drifting_chain_and_stuck_walkers`: how to build a broken chain on purpose.
3. `test_08_injection.py::test_injection_grid_is_unbiased_and_calibrated`: what "honest error bars" means as code.
