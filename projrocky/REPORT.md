# RockyEye: an independent reduction of the JWST/MIRI 15 µm eclipses of GJ 3929 b

> **STATUS (2026-11-14): complete.** The comparison with the official LHS 1140 b truth (§5.3) was
> not done (by choice); the pixel-level validation gate (§5.2) is the validation used.
> All numbers come from the generated tables `products/<config>/stage08/tables.md`.

## Abstract

We reduce the four JWST/MIRI F1500W secondary eclipses of GJ 3929 b (Rocky Worlds DDT,
program 9235) from the raw ramps. Calibration uses the official `jwst` stages 1–2; photometry,
systematics models and a joint `emcee` fit are our own. The pipeline was first validated on
synthetic stamps with a known eclipse and by planting eclipses of 0, 80 and 160 ppm in the
calibrated frames of the simulated LHS 1140 b data set, all recovered within 0.1σ with the depth
scale preserved to 2%. For GJ 3929 b we measure a joint eclipse depth of **126 ± 20 ppm**
(ΔBIC = 25 against no eclipse) at Δt = −12.2 (+2.9/−1.8) min from phase 0.5, and the four eclipses
agree (χ² = 2.3 for 3 dof). With our measured stellar flux this is a dayside brightness
temperature of **674 ± 64 K**, R = T_b/T_max = 0.91 ± 0.09. Thick (100 mbar) CO₂-bearing
atmospheres are excluded at 4.4–4.5σ (Bayes factors ≤ 10⁻⁵ against a bare rock) for all three
stellar models; full heat redistribution is disfavoured at 1.5–2.4σ; a bare rock, partial
redistribution and a thin (30 mbar) atmosphere cannot be told apart. The joint depth agrees with
the official archive product (142.5 ± 18.6 ppm) and with Connors et al. (2026; 118 ± 22 ppm) within
1σ, and our eclipses 1+2 (142 ± 25 ppm) agree with Xue et al. (2025; 160 ± 26 ppm).

## 1. Introduction

GJ 3929 b is a 1.09 R⊕ planet on a 2.62-day orbit around a nearby M dwarf (Kemmer et al. 2022;
Beard et al. 2022). It is the first target of the Rocky Worlds DDT programme, which measures
15 µm secondary eclipses of rocky planets to ask whether they kept an atmosphere. At 15 µm a
planet with no atmosphere shows the hottest possible dayside (no heat is carried to the night
side); an atmosphere lowers the dayside temperature, and CO₂ absorbs in this very band, so the
eclipse is shallower. For GJ 3929 b the dark-bare-rock limit is T_max = 737 ± 14 K (Xue et al. 2025).

Two published analyses exist. Xue et al. (2025), from the first two eclipses: 160 ± 26 ppm,
T_b = 782 ± 79 K, "likely a bare rock". Connors et al. (2026), from all four: 118 ± 22 ppm,
T_b = 641 (+59/−64) K, still consistent with bare rock but leaving room for thin atmospheres;
individual eclipses differ by up to 2.5σ. The official Rocky Worlds archive products (HLSP)
give 142.5 ± 18.6 ppm for the four eclipses combined. This report is a third, independent
reduction from the raw ramps, built to be validated first on data with a known answer.

## 2. Data

| | GJ 3929 b (real) | LHS 1140 b (simulated) |
|---|---|---|
| Source | Rocky Worlds Data Challenge, Kaggle `stsci/rocky-worlds-gj-3929b-observations` (program DD 9235 uncal files, frozen copy) | Kaggle `stsci/rocky-worlds-lhs-1140b-simulations` |
| Eclipses | 4 (2025 Jul 8, 2025 Jul 31, 2026 Jan 30, 2026 Feb 10) | 9 |
| Files / size | 25 uncal segments / 32 GB | 106 uncal segments / 199 GB |
| Mode | MIRI imaging, F1500W, SUB128, FASTR1, 47 groups | F1500W, SUB256, FASTR1, 45 groups |
| Integration time | ~5.6 s | 13.5 s |
| Notes | eclipse 4 split in two exposures (obs 4 and 5), 16 min gap near mid-eclipse | truth not public until after Nov 18, 2026 |

Raw data never reach the laptop. Stages 01–02 run in two Kaggle notebooks
(`notebooks/kaggle_lhs1140b_sim.ipynb`, `notebooks/kaggle_gj3929b.ipynb`) with the datasets
attached; their only output is one stamp file per segment (§3.1).

System parameters adopted (config files give the full list and sources):

| Parameter | GJ 3929 b | Source |
|---|---|---|
| Period, T0 (BJD_TDB) | 2.6162644 d, 2460452.8997 | Xue et al. 2025, Table 1 |
| a/R★, i, Rp/R★ | 17.05 ± 0.43, 89.3°, 0.0318 ± 0.0007 | Xue et al. 2025, Table 1 |
| T_eff, R★ | 3369 ± 51 K, 0.315 ± 0.010 R☉ | Kemmer et al. 2022, Table 2 |
| Distance | 15.830 ± 0.006 pc | Kemmer et al. 2022 (Gaia EDR3 parallax 63.173 ± 0.020 mas) |

Beard et al. (2022) give T_eff = 3384 ± 88 K, R★ = 0.32 ± 0.01 R☉, d = 15.822 ± 0.006 pc; the
difference changes T_max by 3 K and the flux-based stellar surface brightness by 3%.

## 3. Methods: from ramps to light curves

### 3.1 Calibration (official pipeline)
`jwst` 3.0.0, `Detector1Pipeline` then `Image2Pipeline`, CRDS context `jwst_1584.pmap` when
tested (the context actually used is recorded in each run's log). Deviations from defaults:

* jump step: rejection threshold 6σ (default 4σ over-flags a bright star in time-series mode);
  the across-integration sigma-clipping variant disabled (`minimum_sigclip_groups` set very high)
  so results do not depend on how integrations are grouped.
* each segment is calibrated in chunks of 50 integrations to bound memory (~4 GB).
* stage 2: flat field and photometric calibration on (frames in MJy/sr), resampling off.

For time-series exposures the pipeline's own parameter references skip the first-frame and RSCD
steps; the last-frame step runs. EMI correction found no matching configuration for the simulated
SUB256 data and was skipped there. For GJ 3929 b (SUB128) it ran, correcting the 390 Hz and
10 Hz components (Kaggle log). Both data sets were calibrated with CRDS context `jwst_1584.pmap`.

From every calibrated integration an 81 × 81 px stamp centred on the star is kept (science,
error, bad-pixel mask, mid-integration BJD_TDB matched by integration number).

### 3.2 Photometry (own code)
Per integration: (1) bad pixels and >5σ temporal outliers replaced by the pixel's running median
(11 integrations); (2) sky = sigma-clipped mean in a 16–30 px annulus; (3) centroid and RMS
widths from flux-weighted moments in a fixed 11 × 11 px window; (4) sum in a circular aperture
with exact pixel-overlap weights, centred on that integration's centroid. Radii 3–8 px are
tried; the radius with the smallest high-pass scatter (running median of 101 integrations
removed) is adopted. That statistic is blind to the eclipse depth.

Cross-check against Eureka! (stage 3, photutils exact apertures) on the same 271 calibrated
integrations of simulated eclipse 2, 5 px aperture, 16–30 px annulus: normalised fluxes agree to
59 ppm RMS per integration (the per-integration scatter is ~720 ppm in both), correlation 0.998
(`products/lhs1140b_sim/validation/eureka_crosscheck.json`).

### 3.3 Absolute stellar flux
Aperture 5.69 px, sky 8.63–11.45 px, aperture correction 1.497 (CRDS `jwst_miri_apcorr_0014`,
70% encircled energy), 3% systematic uncertainty adopted.

## 4. Methods: from light curves to a verdict

### 4.1 Model
For each visit, flux(t) = [1 + δ·s(t − t_ecl)] × Σ_k c_k B_k(t) + noise. δ is the eclipse depth;
s is the eclipse shape from `batman` (uniform planet disk, circular-orbit geometry fixed at the
values of §2); t_ecl = predicted phase-0.5 time + Δt. The systematics basis B_k is a constant,
a linear trend, optionally an exponential ramp exp(−t/τ) (or two, or a quadratic/cubic instead),
and optionally linear terms in centroid x, y, PSF widths and sky level. Noise is white with one
σ per visit, fitted.

### 4.2 Systematics model selection
Six candidates per visit, all including linear decorrelation against centroid x and y:
`linear+xy`, `poly2+xy`, `exp+xy`, `exp+xyw` (+ PSF widths), `exp+xywbg` (+ widths, sky),
`exp2+xy` (two ramps). Centroid terms are a floor, not a choice, because of the synthetic test in
§5.1: without them, pointing drift tripled the depth scatter while BIC still preferred dropping
them. Each candidate is fitted by maximum likelihood (differential evolution over δ, Δt, τ;
linear coefficients by least squares). The model with the fewest parameters within ΔBIC ≤ 2 of
the best is adopted.
Residuals are checked for red noise with the binned-RMS curve and β at the eclipse timescale.
The depth under *every* candidate is reported, not only the chosen one.

### 4.3 Likelihood and sampling
The linear coefficients are marginalised analytically (flat priors), leaving
ln L = −(N−K)/2 ln(2πσ²) − χ²_min/2σ² − ½ ln det(MᵀM) per visit. Sampled: δ (uniform −500 to
1000 ppm, negative allowed), Δt (uniform; ±240 min for the simulation, −120 to +60 min for
GJ 3929 b), and per visit ln σ and ln τ (τ log-uniform 0.02–8 h). `emcee`, 64 walkers,
differential-evolution moves, HDF5 backend, started in a small ball around the maximum-likelihood
point. Light curves are binned by 4 integrations. The first third of each chain is discarded.

Fits: (i) all visits jointly, shared δ and Δt; (ii) each eclipse alone with Δt restricted to ±3σ
of the joint value; (iii) for GJ 3929 b, eclipses 1+2 jointly, to compare with Xue et al. (2025).
GJ 3929 b's eclipse 4 is two exposures with independent systematics and one depth.

### 4.4 Convergence
Required for every fit: chain ≥ 50 τ, split R̂ ≤ 1.01, effective samples ≥ 1000, Geweke |z| ≤ 3,
acceptance 0.05–0.9 with no stuck walker. The sampler stops on the first three (τ stable to 10%
between checks); the report flags any failure of the rest.

### 4.5 Detection significance
ΔBIC between the fit and the same systematics with δ ≡ 0.

### 4.6 Injection-recovery
*Pixel level* (validation gate): the real eclipse is masked; eclipses of 0, 80, 160 ppm are
planted in the calibrated frames in the middle of the longest clean stretch beside it (requiring
≥ 0.25 eclipse-durations of baseline on each side), then photometry and joint MCMC rerun.
Pass: every recovered depth within 2σ of its injected value, and the depth scale preserved
((recovered − recovered at 0) / injected within 5% of 1). 2σ rather than 1σ because honest error
bars miss the truth at 1σ a third of the time; the 5% gain test is the sharp one.
*Light-curve level* (sensitivity): for each depth, N trials of fitted systematics + cyclically
shifted, sign-flipped residuals + injected eclipse, refitted with a linearised estimator with the
eclipse time scanned across the prior. Reported: bias, scatter vs formal error, 3σ detection rate.

### 4.7 Physics
Band-averaged (photon-weighted, F1500W throughput from the SVO Filter Profile Service) Planck
functions convert δ to a dayside brightness temperature T_b for three stellar surface-brightness
choices: from our measured flux with R★ and d (primary); blackbody at T_eff; from a literature
model flux. Uncertainties in T_eff, a/R★, Rp/R★, R★, d and flux are propagated by Monte Carlo.
Predicted depths: T_day = T★/√(a/R★) (1−A)^¼ (2/3 − 5ε/12)^¼ for bare rock (ε = 0; A = 0, 0.1),
partial (ε = 0.5) and full redistribution (ε = 1; A = 0, 0.3), plus three literature forward
models with CO₂ (Connors et al. 2026, Table 4). Each scenario's evidence is the depth likelihood
(KDE of the posterior, valid because the depth prior is flat) averaged over its predicted-depth
distribution; Bayes factors are quoted against the dark bare rock. Also: R = T_b/T_max and the
95% upper limit on ε.

## 5. Validation

### 5.1 Synthetic stamps (truth known) — done

Six visits of synthetic 81 × 81 px stamps (`rockyeye/synth.py`): Airy PSF, sky, signal-dependent
noise (~590 ppm per 13.5 s integration), settling ramp of 2–5 × 10⁻³ with τ ≈ 0.3–0.7 h, drift,
3 mpx jitter with a flux–position coupling, cosmic rays, 12 dead pixels; eclipse of **100 ppm**
at Δt = +12 min. Stages 03–08 and the pixel-level gate were run exactly as for real data.

<!-- SYNTHETIC-RESULTS:BEGIN -->
**Result: the injected 100 ppm is recovered.** Joint depth **117 ± 15 ppm (+1.1σ)**; eclipse
time Δt = 14.7 (+2.0/−1.8) min vs 12.0 min true (+1.5σ). The truth lies at the 13th percentile
of the posterior: outside the 68% interval, inside the 95% interval
(`products/synthetic/validation/truth_check.json`). All fits converged (max R̂ 1.009,
≥ 142 τ, ESS ≥ 9000); residuals white at the eclipse timescale (β ≤ 1.07).

| Eclipse | Depth (ppm) | Single-eclipse ΔBIC |
|---|---|---|
| 1 | 68 ± 37 | −9.0 |
| 2 | 148 ± 39 | +3.2 |
| 3 | 181 ± 40 | +9.3 |
| 4 | 140 ± 41 | +0.1 |
| 5 | 79 ± 33 | −6.3 |
| 6 | 131 ± 34 | +3.1 |
| **Joint** | **117 ± 15** | **+46.2** |

The single-eclipse depths agree with each other (χ² = 6.7 for 5 dof, p = 0.24). Photometric
scatter equals the pipeline's own propagated error (ratio 0.95–1.02 per visit).

*Pixel-level gate (§4.6): PASSED.* Injected 0 → 1.5 ± 23.2, 80 → 82.3 ± 22.3,
160 → 163.6 ± 21.5 ppm; differential gain 1.010 and 1.013 (depth scale preserved to ~1%).

*Light-curve-level injection (100 trials per depth):*

| Injected (ppm) | Mean recovered | Scatter | Bias | Scatter / formal error | Detected at 3σ |
|---|---|---|---|---|---|
| 0 | −0.2 | 29.3 | −0.2 ± 2.9 | 2.04 | 0% |
| 50 | 52.3 | 22.1 | +2.3 ± 2.2 | 1.53 | 1% |
| 100 | 100.2 | 14.1 | +0.2 ± 1.4 | 0.98 | 74% |
| 150 | 149.5 | 14.8 | −0.5 ± 1.5 | 1.03 | 100% |
| 250 | 247.2 | 15.6 | −2.8 ± 1.6 | 1.08 | 100% |

No measurable bias at any depth. Below about twice the error the scatter exceeds the
fixed-timing formal error (ratio 1.5–2): with the eclipse time free, a weak eclipse locks onto
noise. The MCMC marginalises over timing, and the detection threshold uses the empirical null
scatter (3 × 29 = 88 ppm), not the formal error.

**What the first run taught (and changed the method).** The first synthetic run gave
144 ± 14 ppm (+3σ). Switching ingredients off one at a time: with noise off, the pipeline gives
99.9 ppm, so photometry, eclipse model and fitter are unbiased. The excess came from pointing
drift (a slow random walk of a few mpx, coupled to flux) in visits where model selection had
dropped the centroid terms. Over 10 noise realisations of one visit:

| Model | With drift | Without drift |
|---|---|---|
| `exp` (no centroid terms) | 107 ± 24 ppm, scatter **77** | 103 ± 7, scatter 23 |
| `exp+xy` | 105 ± 8 ppm, scatter **24** | 106 ± 7, scatter 23 |

Unbiased on average either way, but without centroid terms the scatter is three times the quoted
error, and neither BIC nor β flagged it. Centroid terms are therefore always included (§4.2).

Two further bugs were found and fixed on this data: the no-eclipse comparison model was evaluated
outside the timing prior, which inflated single-eclipse ΔBIC to ~600; and the gate's fixed
injection offset fell outside 8.6 h visits.

Figures: `products/synthetic/figures/` (`phase_joint.png`, `depths.png`, `injection_recovery.png`,
`validation_injected.png`, `corner_joint.png`, `trace_joint.png`, `allan_joint.png`).
The physics stage also ran on the synthetic data as a code-path test; its temperatures are not
meaningful, because the 100 ppm truth is arbitrary rather than physical.
<!-- SYNTHETIC-RESULTS:END -->

### 5.2 Simulated LHS 1140 b: pixel-level injection gate

**PASSED** (2026-11-14). The simulated eclipse was masked and eclipses planted in the
calibrated frames of all nine visits; photometry and the joint MCMC were rerun on each.

| Injected (ppm) | Recovered (ppm) | Pull (σ) | Differential gain |
|---|---|---|---|
| 0 | 0.1 ± 15.1 | +0.01 | — |
| 80 | 81.6 ± 13.5 | +0.11 | 1.018 |
| 160 | 160.0 ± 13.3 | +0.00 | 0.999 |

Every depth is within 2σ (in fact within 0.11σ), and the depth scale is preserved to 2%, inside
the 5% bar. This unlocked the GJ 3929 b analysis. Figure:
`products/lhs1140b_sim/figures/validation_injected.png`.

### 5.3 Simulated LHS 1140 b: recovered eclipse and official truth

No significant eclipse is found in the simulated data. Joint depth 41 (+15/−23) ppm, ΔBIC = −5.0
(the no-eclipse model is preferred). The joint chain **did not meet the convergence criteria**:
it stopped at the 60 000-step cap with R̂ = 1.028 and 49 τ (criteria: R̂ ≤ 1.01, ≥ 50 τ). The timing
posterior sits at Δt = +131 (+4/−13) min of a ±240 min prior, which with no detectable eclipse is
the sampler finding small dips in the noise, not a measured time. The single-eclipse fits all
converged:

| Eclipse | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Depth (ppm) | −0 ± 46 | −88 ± 51 | 29 ± 49 | 105 ± 36 | 2 ± 53 | −15 ± 41 | 56 ± 52 | 159 ± 43 | 31 ± 40 |

Their weighted mean is 39 ± 15 ppm; the scatter is larger than the errors (χ² = 20.7 for 8 dof,
p = 0.01), driven by eclipses 2, 4 and 8, and eclipse 8 shows red noise (β = 1.78). Photometric
scatter is 1.14–1.20 × the pipeline's propagated error in every visit.

Light-curve injections into these light curves are unbiased within 10 ppm at all depths, with the
scatter equal to the formal error to within 10–56% from 80 ppm up; below that the free timing locks
onto noise (scatter 3–3.5 × formal). LHS 1140 b is temperate (T_max ≈ 290 K), so a small or absent
15 µm eclipse is physically plausible; whether 41 ppm is right waits for the official truth.

*Not done (by choice). The official injected values are published by STScI after the Data Challenge workshop (Nov 16–18, 2026); `scripts/revalidate_truth.py` and the `truth:` slot in `configs/lhs1140b_sim.yaml` remain if this comparison is ever wanted.*

## 6. Results: GJ 3929 b

Run 2026-11-14 after the gate passed. Figures: `products/gj3929b/figures/`.

### 6.1 Photometry

| Fit unit | Integrations | Duration (h) | Aperture (px) | Scatter (ppm) | Pipeline error (ppm) | Ratio |
|---|---|---|---|---|---|---|
| Eclipse 1 | 3387 | 5.37 | 8.0 | 682 | 569 | 1.20 |
| Eclipse 2 | 3863 | 6.13 | 4.5 | 701 | 582 | 1.20 |
| Eclipse 3 | 3891 | 6.17 | 4.5 | 695 | 583 | 1.19 |
| Eclipse 4a | 2980 | 4.73 | 4.0 | 715 | 590 | 1.21 |
| Eclipse 4b | 1637 | 2.60 | 3.5 | 686 | 610 | 1.12 |

Scatter is 12–21% above the propagated pipeline error, as in the simulation (14–20%). The fit
uses its own σ per visit, so this does not bias the error bars (`raw_light_curves.png`).

### 6.2 Systematics

Chosen models (simplest within ΔBIC 2 of the best): eclipse 1 `exp+xywbg`, eclipse 2 `linear+xy`,
eclipse 3 `poly2+xy`, eclipse 4a `linear+xy`, eclipse 4b `exp+xyw`. Models without a ramp leave
red noise (β up to 2.3); the chosen ones give β ≤ 1.13 per visit. One caution: the per-visit
maximum-likelihood depths, with timing free, move a lot between candidates that fit equally
well. In eclipse 2, `linear+xy` gives +149 ppm and `poly2+xy` −115 ppm at ΔBIC 0.1. In the joint
fit the shared timing removes most of this freedom, and the single-eclipse MCMC fits restrict
timing to the joint value. All candidate depths are in `tables.md`.

### 6.3 Eclipse depth

| Fit | Depth (ppm) | Δt (min) | ΔBIC (eclipse) |
|---|---|---|---|
| **All four eclipses** | **126 ± 20** | **−12.2 (+2.9/−1.8)** | **+25.4** |
| Eclipses 1+2 | 142 (+25/−24) | −11.9 (+1.6/−2.4) | +19.9 |
| Eclipse 1 | 144 (+38/−39) | −12.0 | +2.1 |
| Eclipse 2 | 137 ± 33 | −12.5 | +6.3 |
| Eclipse 3 | 110 ± 35 | −9.9 | −1.5 |
| Eclipse 4 (4a + 4b) | −36 (+125/−122) | −12.1 | −12.7 |

The four single-eclipse depths are consistent with each other (χ² = 2.3 for 3 dof, p = 0.52),
and their weighted mean, 126 ± 20 ppm, equals the joint fit. Eclipse 4 carries almost no weight:
its two exposures have independent systematics and the 16 min gap falls near mid-eclipse, so
neither half holds both ingress and egress. Its joint residuals also show red noise (β = 1.82).
(`fit_joint.png`, `phase_joint.png`, `depths.png`, `corner_joint.png`.)

### 6.4 Eclipse timing

All fits agree on Δt ≈ −12 min. If the eclipse ephemeris is right, this is
e·cos ω = πΔt/(2P) = −0.0051 (+0.0012/−0.0008), i.e. a near-circular orbit. The light-travel time
across the orbit (2a/c ≈ 0.4 min) is not subtracted. An error in the predicted phase-0.5 time of this
size is equally possible, so we do not claim an eccentricity.

### 6.5 Convergence

Every fit converged: max R̂ ≤ 1.010, ≥ 133 τ, ESS ≥ 8488 (`trace_joint.png`).

### 6.6 Detection significance

ΔBIC = 25.4 for the joint fit (a "very strong" preference for the eclipse). Measured against the
empirical null scatter of the injection test (51 ppm at 0 ppm injected; §6.7), 126 ppm is 2.5σ.
That figure is conservative, because the null scatter includes timing freedom across the whole
180 min prior, which the data do not have once the eclipse is seen.

### 6.7 Injection-recovery (light-curve level, 200 trials per depth)

| Injected (ppm) | Mean recovered | Scatter | Bias | Scatter / formal error |
|---|---|---|---|---|
| 0 | −3.3 | 50.7 | −3.3 ± 3.6 | 2.58 |
| 50 | 13.7 | 64.2 | −36.3 ± 4.5 | 3.27 |
| 100 | 74.5 | 70.5 | −25.5 ± 5.0 | 3.59 |
| 150 | 143.6 | 19.1 | −6.4 ± 1.4 | 0.97 |
| 200 | 192.5 | 21.0 | −7.5 ± 1.5 | 1.07 |
| 300 | 289.0 | 21.0 | −11.0 ± 1.5 | 1.07 |

From 150 ppm up the formal error is right (ratio 0.97–1.07), with a small low bias of 6–11 ppm (≈ 4%).
Below 100 ppm the scanned timing locks onto noise and the scatter is 3× the formal error. At our
126 ppm we are between these regimes; the real fit's timing is tight (±2–3 min), which puts it
in the well-behaved one. A bias of this size would make the true depth slightly deeper, i.e.
slightly more rock-like (`injection_recovery.png`).

### 6.8 Stellar flux

Measured 19.05 ± 0.57 mJy (3% systematic; visits 19.00–19.09). HLSP: 18.86 ± 0.12 mJy (+0.3σ);
Connors et al. (2026): 19.19 ± 0.12 mJy (−0.2σ). The five visits agree to 0.5%.

### 6.9 Temperature and scenarios

| Stellar model | Star T_b (K) | Planet T_b (K) | R = T_b/T_max | ε (95% upper) |
|---|---|---|---|---|
| Measured flux (primary) | 2898 | 674 (+66/−62) | 0.91 ± 0.09 | < 0.90 |
| Blackbody at T_eff | 3370 | 743 (+70/−69) | 1.01 ± 0.09 | < 0.79 |
| Literature (SPHINX) model | 2865 | 670 (+65/−62) | 0.91 ± 0.09 | < 0.91 |

T_max = 737 K. Scenarios, with the primary stellar model (Bayes factor B against a dark bare rock;
the other two stellar models in `tables.md`):

| Scenario | Predicted depth (ppm) | Data − model | B vs bare rock |
|---|---|---|---|
| Bare rock, A = 0 | 148 ± 13 | −0.9σ | 1 |
| Bare rock, A = 0.1 | 141 ± 13 | −0.7σ | 1.25 |
| Partial redistribution, ε = 0.5 | 125 ± 11 | +0.0σ | 1.54 |
| Full redistribution, A = 0 | 93 ± 9 | +1.5σ | 0.50 |
| Full redistribution, A = 0.3 | 77 ± 7 | +2.3σ | 0.12 |
| Thin O₂ + 0.1% CO₂, 30 mbar | 91 ± 6 | +1.7σ | 0.43 |
| O₂ + 1% CO₂, 100 mbar | 37 ± 2 | +4.4σ | 1.4 × 10⁻⁶ |
| Venus-like, 100 mbar | 36 ± 2 | +4.5σ | 2.8 × 10⁻⁷ |

With a blackbody star the planet looks hotter (R = 1.01). Full redistribution is then disfavoured
at 2.4σ (A = 0, B = 0.063) and 3.1σ (A = 0.3, B = 0.011), and the 30 mbar atmosphere at B = 0.25.
The 100 mbar models are excluded at B ≤ 10⁻⁵ under all three (`physics_scenarios.png`).

## 7. Benchmark against published results

Published values the pipeline is compared with (already in `configs/gj3929b.yaml`):

| Quantity | Xue+25 | Connors+26 (FN-PCA) | Connors+26 (poly3) | RWDDT HLSP | RockyEye |
|---|---|---|---|---|---|
| Eclipse 1 (ppm) | 177 ± 46 | 190 ± 40 | 115 ± 53 | −9.1 ± 95.5 | **144 ± 38** |
| Eclipse 2 (ppm) | 143 ± 35 | 155 ± 34 | 183 ± 40 | 16.2 ± 78.1 | **137 ± 33** |
| Eclipse 3 (ppm) | — | 60 ± 34 | 92 ± 39 | 174.0 ± 37.4 | **110 ± 35** |
| Eclipse 4 (ppm) | — | 97 ± 78 | 129 ± 89 | 190.7 ± 130.9 | **−36 ± 124** |
| Eclipses 1+2 (ppm) | 160 ± 26 | 172 ± 26 | — | — | **142 ± 25** |
| All four (ppm) | — | 118 ± 22 | 126 ± 25 | 142.5 ± 18.6 | **126 ± 20** |
| T_b (K) | 782 ± 79 (SPHINX), 745 ± 76 (observed flux) | 641 (+59/−64) | — | — | **674 ± 64** (measured flux), 670 ± 64 (SPHINX), 743 ± 70 (blackbody) |

HLSP single-eclipse entries are the archive's v1.0.1 eclipse catalogs; the combined entry is
"checkpoint02" (files in `data/benchmarks/gj3929b_hlsp/`). All analyses use the same photons, so
differences between them measure method, not independent noise; a "tension in σ" overstates how
surprising a difference is.

**Combined depths agree.** Our 126 ± 20 ppm is within 0.6σ of the HLSP, 0.3σ of Connors+26
(FN-PCA) and identical to Connors+26 (poly3). Our eclipses 1+2 (142 ± 25 ppm) sit 0.5σ below
Xue+25 and 0.9σ below Connors+26. The published combined values span 118–172 ppm; ours is in the
lower half of that range, near Connors+26.

**Single eclipses scatter, for everyone.** Eclipses 1 and 2: we agree with both papers within 1σ,
while the HLSP finds almost nothing (−9 and 16 ppm). Both papers and we see ~140–190 ppm, so the
HLSP's per-eclipse values are the odd ones out, probably from their per-eclipse systematics choice.
Eclipse 3: our 110 ppm lies between Connors+26 (60) and the HLSP (174), within 1.3σ of each.
Eclipse 4: we get essentially no constraint (±124 ppm). Connors+26 report 97 ± 78 and the HLSP
191 ± 131; our larger error comes from fitting the two exposures with fully independent
systematics, which costs most of the eclipse signal when the gap sits mid-eclipse. §6.2 shows that
on this data plausible systematics models move single-eclipse depths by more than their error
bars, which is the likeliest cause of these differences.

**Temperature.** With the measured stellar flux our T_b = 674 ± 64 K, between Connors+26
(641 K) and Xue+25 (745 ± 76 K with observed flux). The stellar model moves T_b by ~70 K (a full
1σ), more than the differences in depth: a blackbody star gives 743 K. Like Connors+26 we rely on
the star's measured 15 µm flux, which fixes the main systematic; Xue+25's higher T_b comes mostly
from their larger depth.

## 8. Conclusion

**GJ 3929 b has a hot dayside and no thick atmosphere.** The joint eclipse depth is
126 ± 20 ppm, detected at ΔBIC = 25 and consistent across the four eclipses. The dayside brightness
temperature is 674 ± 64 K, R = T_b/T_max = 0.91 ± 0.09.

* **Excluded:** 100 mbar atmospheres with CO₂ (Venus-like; O₂ + 1% CO₂), at 4.4–4.5σ,
  B ≈ 10⁻⁶–10⁻⁷ against a bare rock. This holds for all three stellar models.
* **Disfavoured:** full heat redistribution, at 1.5σ (B = 0.50) with our measured stellar flux,
  rising to 2.4σ (B = 0.063) with a blackbody star; ε < 0.90 at 95% (< 0.79 with a blackbody).
* **Not excluded:** a dark or slightly reflective bare rock, partial redistribution (ε = 0.5,
  B = 1.5) and a thin 30 mbar O₂ + 0.1% CO₂ atmosphere (B = 0.43). The data cannot separate these.

So the result is *consistent with a bare rock* and rules out a thick atmosphere, but it does not
prove a bare rock: a thin or partial atmosphere fits equally well.

**How robust it is.** The verdict on thick atmospheres does not depend on the stellar model or on
which eclipses are used (eclipses 1+2 alone give 142 ± 25 ppm). The stellar model moves T_b by
~70 K and R from 0.91 to 1.01, which is what decides how strongly full redistribution is
disfavoured. Injection tests show the error bar is honest at this depth, with a possible 6–11 ppm
low bias, which would push the answer slightly further towards a bare rock.

**Caveats.** (1) Single-eclipse depths are sensitive to the systematics model (§6.2), so the
per-eclipse numbers are less reliable than the joint one. (2) Eclipse 4 contributes almost nothing,
and its residuals show red noise (β = 1.82). (3) Photometric scatter is ~20% above the pipeline's
propagated error, absorbed by fitting σ per visit. (4) The validation gate passed, but the
simulation's own joint fit did not converge, and the comparison with its official truth (§5.3)
is still to come.

## 9. Limitations (already known)

* Circular-orbit eclipse shape with fixed a/R★ and i; timing is free. The duration error from
  the ±0.43 uncertainty in a/R★ is ~2.5%, small against the depth error.
* One white-noise σ per visit; red noise is diagnosed (β), not modelled.
* Systematics models are parametric (ramp + decorrelation), not data-driven (no pixel-level PCA).
* The physics scenarios are blackbody energy-balance models; atmosphere spectra are taken from
  the literature, not computed.
* Stages 1–2 ran on Kaggle (GJ 3929 b: 25/25 segments in one 4.5 h run; simulation: 106/106 in
  two runs, resumed from the first run's output); locally only one simulated segment was tested.
* Each eclipse-4 exposure has its own systematics model; a model sharing terms across the gap
  could recover more of that eclipse.

## References

Beard, C., et al. 2022, ApJ (arXiv:2207.10672) · Connors, N. J., et al. 2026, arXiv:2606.07511 ·
Cowan, N. B., & Agol, E. 2011, ApJ, 729, 54 · Kemmer, J., et al. 2022, A&A (arXiv:2202.00970) ·
Xue, Q., et al. 2025, ApJL (arXiv:2508.12516) · Rocky Worlds DDT HLSP, MAST, doi:10.17909/qsyr-ny68 ·
Rocky Worlds DDT Data Challenge 2026, kaggle.com/competitions/rocky-worlds-data-challenge
