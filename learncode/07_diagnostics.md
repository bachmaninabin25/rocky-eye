# 07 — MCMC convergence diagnostics (`rockyeye/s07_diagnostics.py`, 135 lines) · Tier 2

HOW_IT_WORKS §7 · learntopic day 2 · exercise: `codecode/tier2/07_diagnostics/` (split R̂)

## What it does
Decides whether an MCMC chain can be trusted. Nothing *proves* convergence, so several
independent warning lights must all be off.

| Check | Threshold | Catches |
|---|---|---|
| chain length / τ | ≥ 50 | too short (τ itself unreliable below that) |
| split R̂ | ≤ 1.01 | walkers / halves disagree |
| effective samples (ESS) | ≥ 1000 | too few independent draws for 1σ quantiles |
| Geweke z (first 10% vs last 50%) | ≤ 3 | still drifting |
| acceptance | 0.05–0.9, no walker < 0.02 | stuck or random-walking |

## The key ideas
- **τ, the integrated autocorrelation time:** how many steps make one independent sample.
  ACF via FFT, averaged over walkers, summed out to a self-consistent window M ≥ 5τ (Sokal).
  ESS = N_samples / τ.
- **Split R̂:** cut each walker in half; compare variance *between* sequences with variance
  *within*. Same distribution → 1.
- **Tests build deliberately broken chains** (two clusters, 3τ long, drifting, stuck walkers)
  and check each gets flagged. Test the alarm, not just the happy path.
- The same criteria are used *to stop the sampler* in 06. That was a bug fix: stopping on
  length alone let R̂ = 1.015 chains through.

## Can you explain it?
1. Why isn't "I ran 100,000 steps" enough?
   <details><summary>answer</summary>Length only matters relative to τ. 100k steps with τ = 5000
   is just 20 independent samples, and a stuck chain can be any length.</details>
2. What does R̂ = 1.2 mean in plain words?
   <details><summary>answer</summary>Different parts of the chain sample noticeably different
   regions; the combined spread is ~20% wider than each part's own spread, so it hasn't mixed.</details>
3. Why use several diagnostics instead of the best one?
   <details><summary>answer</summary>Each catches a different failure; none catches all.</details>

## Exercise
`split_rhat`: ~10 lines. `python run.py check 07`.
