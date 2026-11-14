# 08 — Injection-recovery (`rockyeye/s08_injection.py`, 165 lines) · Tier 2

HOW_IT_WORKS §8 · exercise: `codecode/tier2/08_injection/`

## What it does
Answers "would we have seen an eclipse of depth D, and measured it right?" by planting fake
eclipses and running the same machinery. Two levels:

1. **Pixel level** (`inject_into_stamps`): dim the star (not the sky) in the actual frames, rerun
   photometry and the MCMC. This is the **validation gate**: `pipeline.stage_validate` masks the
   real eclipse, plants 0, 80 and 160 ppm in the longest clean stretch beside it, and passes only
   if every one comes back within 2σ and the depth *scale* holds to 5%. The real planet's config
   is locked until this passes.
2. **Light-curve level** (`run_injection_grid`): hundreds of trials per depth. New noise comes from
   the real residuals cyclically shifted ("prayer bead"), which keeps their correlations. Each
   trial uses the fast linear estimator.

## The fast estimator (your exercise)
For depths ≪ 1: `(1 + d·s)·(B c) ≈ B c + (d·c₀)·s`, so the eclipse is **one more linear
column**. Depth and its formal error come out of a single least-squares solve. That's what makes
hundreds of trials cheap.

## Outputs
Bias (recovered − injected), `error_ratio` (scatter of recovered depths / quoted error; ≈1 means
honest error bars), detection fraction at 3σ per depth.

## Lessons
- **The gate couldn't place its fake eclipse** at first: a fixed offset fell off the end of the
  8.6 h visits. Now it finds the longest clean stretch.
- **Pass bars at 2σ, not 1σ.** With honest errors the truth lands outside 1σ a third of the
  time, so a 1σ bar would fail a *correct* pipeline one run in three.

## Can you explain it?
1. Why inject at the pixel level at all, if the light-curve level is faster?
   <details><summary>answer</summary>Pixel injection tests photometry too (aperture, sky,
   centroids). Light-curve injection assumes photometry is fine.</details>
2. Why "prayer bead" noise instead of fresh Gaussian noise?
   <details><summary>answer</summary>Real residuals contain correlated noise; Gaussian noise
   would make the test too optimistic.</details>
3. What does error_ratio = 1.5 mean?
   <details><summary>answer</summary>Recovered depths scatter 1.5× more than the quoted error
   says: the error bars are too small by that factor.</details>

## Exercise
`inject_into_stamps` + `linear_depth`: ~20 lines. `python run.py check 08`.
