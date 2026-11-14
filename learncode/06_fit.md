# 06 — Likelihood and MCMC (`rockyeye/s06_fit.py`, 347 lines) · Tier 1: rebuild (the hard one)

HOW_IT_WORKS §6 · learntopic day 2 · rebuild: `codecode/06_fit/`

## What it does
Turns light curves into **depth ± error**. Builds the posterior probability of the parameters
given the data, finds its peak (max likelihood, for BIC), then samples it with `emcee`.

## Where it sits
Uses 04 (shape) and 05 (basis, solve). Feeds 07 (are the chains converged?), 08 (injection uses
the same model), 09 (depth samples → verdict), 10 (BIC, detection).

## The model
```
flux(t) = [1 + depth·shape(t − t_ecl)] × [B(t;θ)·c] + noise,   noise ~ N(0, σ²)
```
Parameters: `depth_ppm`, `dt_min` (eclipse-time offset, shared by all visits), and per visit
`log_sigma` plus ramp timescales. The linear c's are integrated out:
```
ln L = −(N−K)(ln σ + ½ ln 2π) − χ²_min/(2σ²) − ½ ln det(MᵀM)
```
M = eclipse-modulated basis (K columns). Visits multiply → their ln L add, depth shared.

## The key ideas
1. **Marginalise the linear parameters analytically.** The Gaussian integral over c gives the
   formula above. `N−K` because K of the N normalisation factors are "used up". Verified by
   brute-force numerical integration in a test.
2. **Profile σ for max likelihood.** For the best fit, σ² = χ²/N exactly, so the optimiser only
   minimises Σ N ln(χ²/N). Differential evolution (global) because the ramp timescale has local
   minima.
3. **BIC counts the linear c's too.** `k = free non-linear + one σ per visit + all linear c's`.
4. **Depth may go negative** (prior −500…1000 ppm). A positive-only prior biases weak signals up.
5. **The shape is a lookup table** (`ShapeTable`): batman once on a 1 s grid, then `np.interp`.
   One likelihood call ≈ 0.2 ms.
6. **Safe to kill and rerun.** emcee's HDF5 backend stores every step; rerunning continues.
7. **Stops on convergence, not length:** chain ≥ 50τ, τ stable within 10%, split R̂ ≤ 1.01 (07).

## Lessons from bugs (from HANDOFF/REPORT)
- **The MCMC stopped before R̂ converged.** The first version stopped on chain length alone;
  the slow test caught chains "long enough" with R̂ = 1.015. Now all three criteria must pass.
- **ΔBIC ≈ 600 for 3σ signals.** The no-eclipse model left the timing at 0, *outside* a narrow
  timing prior, so every trial point was rejected and it was scored at a random ramp.
  Fix: park unused parameters inside their prior. A regression test now guards it.

## Annotated real code
```python
def design(self, p, i):
    v = self.visits[i]
    theta = p[self._slices[i]][1:]                     # this visit's ramp timescales
    B = v.model.basis(v.th, v.aux, theta)              # systematics columns (05)
    for tb in v.breaks:                                # split exposures may jump in flux
        B = np.column_stack([B, (v.t > tb).astype(float)])
    E = 1.0 + p[0]*1e-6*self.shape(v.t, self.t_ecl(p, i))   # eclipse factor (04)
    return E[:, None] * B                              # flux ≈ E * (B c): multiply row-wise
```
`breaks`: GJ 3929 b's eclipse 4 was split into two exposures with a 16 min gap. Each piece gets
its own flux offset (your lean version skips this).

## Gotchas
- `dt_min` is in **minutes**, times are in **days**: `/1440`.
- emcee walkers must *start* inside the prior, or they never move.
- Burn-in: the first third is thrown away before any summary.

## Can you explain it?
1. What is a likelihood, and why is it not "the probability the parameters are right"?
   <details><summary>answer</summary>P(data | parameters). The posterior P(parameters | data)
   needs the prior too (Bayes' theorem).</details>
2. Why integrate out c instead of sampling it?
   <details><summary>answer</summary>Closed-form, exact, and it cuts the dimension from ~60 to
   ~20; MCMC cost grows fast with dimension.</details>
3. Why differential evolution for the best fit instead of `scipy.minimize`?
   <details><summary>answer</summary>The ramp timescale creates multiple local optima; a local
   optimiser can get stuck. DE searches globally, then polishes locally.</details>
4. How do you know the error bar is honest?
   <details><summary>answer</summary>Converged chains (07), white residuals (05 β), and injection
   tests (08) showing the scatter of recovered depths matches the quoted error.</details>
5. What would you change for 9 visits instead of 4?
   <details><summary>answer</summary>Nothing structural: likelihoods add, depth is shared.
   Dimension grows by ~2 per visit, which emcee handles.</details>

## Rebuild checklist
- [ ] `t_ecl`, `design`, `log_prior`, `log_like` (5 fast tests pass now)
- [ ] `max_likelihood` (profile σ, DE, BIC)
- [ ] `run_mcmc`
- [ ] `python run.py check 06` green (includes ~2 min MCMC)
- [ ] `python run.py demo --mine 04,05,06`
