# 05 — Systematics models (`rockyeye/s05_systematics.py`, 168 lines) · Tier 1: rebuild

HOW_IT_WORKS §5 · learntopic day 3 · rebuild: `codecode/05_systematics/`

## What it does
Describes everything that changes the measured flux that **isn't** the planet: the detector
warming up (a ramp over the first hour), slow drift, and the flux following the star's position
(x, y), width (sx, sy) and the sky (bg). Plus tools to check whether what's left is white noise.

## Where it sits
Input: light-curve columns from 03. Used by 06 (the fit), 08 (fast injection estimator) and 10
(model comparison picks which model each visit gets).

## The key ideas
1. **Linear in the coefficients.** Every model is `S(t) = Σ c_k B_k(t; θ)`. The c's (level,
   slope, ramp amplitude, decorrelation weights) are solved exactly by least squares; only θ (ramp
   timescales) is non-linear. That one decision turns a 60-dimensional fit into ~20 dimensions.
2. **Models are strings.** `"exp+xy"` = constant + slope + exponential ramp + x + y. The fit can
   try a menu of them (`candidate_models` in the config) and BIC picks among them (10).
3. **Centred, unit-range time.** `(th − mean)/ptp`: the slope column is then nearly
   independent of the constant column, which keeps the matrix well conditioned.
4. **QR, not normal equations.** `MᵀM` squares the condition number. QR gives the same answer
   stably, and log det(MᵀM) for free: `2 Σ log|R_ii|` (needed by the marginal likelihood).
5. **Robust standardisation.** Decorrelation vectors are scaled by median/MAD, so one cosmic
   ray can't rescale x for the whole visit.
6. **White-noise test.** Bin the residuals in groups of n: white noise falls as 1/√n. If it
   flattens, correlated ("red") noise is left and error bars are too small. β = measured /
   expected RMS at the eclipse timescale; β ≈ 1 good, β > 1.5 warning.

## The big lesson (it's in the config comments too)
The first synthetic run came out **3σ too deep**. Cause: the star wanders a few milli-pixels,
the flux follows, and a model *without* x, y absorbs that drift into the depth. Over 10 noise
realisations the depth wasn't biased on average, but its **scatter was 77 ppm while the quoted
error said ~35**, and neither BIC nor β caught it. With x, y always in: scatter 24 ppm, and
nothing lost when there is no drift.
**Rule: if a nuisance effect is physically expected, keep its term; let statistics choose only
among the extras.** Great interview story.

## Annotated real code
```python
def solve_linear(y, M):
    q, r = np.linalg.qr(M)                 # M = QR, Q orthonormal, R upper-triangular
    c = np.linalg.solve(r, q.T @ y)        # least-squares coefficients
    resid = y - M @ c
    logdet = 2.0 * np.sum(np.log(np.abs(np.diag(r))))   # log det(M^T M) = log det(R^T R)
    return c, float(resid @ resid), float(logdet)
```

## Gotchas
- θ stores **log** τ, so τ stays positive and the sampler works on a sensible scale.
- The ramp uses raw `th` (hours since visit start), so it's exactly 1 at the first point.
- `exp2` has two timescales that can swap places; 06's prior forces τ₂ > τ₁.

## Can you explain it?
1. Why solve the linear coefficients instead of sampling them?
   <details><summary>answer</summary>For fixed non-linear parameters the best coefficients have
   a closed form, and with flat priors the integral over them is analytic. Sampling them would
   add dozens of dimensions and slow MCMC enormously for no gain.</details>
2. What does a binned-RMS curve that flattens tell you?
   <details><summary>answer</summary>Correlated noise remains; the formal error bars
   (which assume white noise) are too small.</details>
3. Why median/MAD instead of mean/std?
   <details><summary>answer</summary>Robust to outliers: one bad point barely moves the median.
   1.4826 × MAD equals σ for Gaussian data.</details>
4. BIC preferred dropping x,y. Why didn't the project follow it?
   <details><summary>answer</summary>BIC scores fit vs complexity on one realisation; it can't see
   that the depth's *scatter across realisations* tripled. Physical knowledge (the star does
   move) overrode it.</details>

## Rebuild checklist
- [ ] `parse_model` + helpers
- [ ] `basis` (column order!)
- [ ] `standardise`, `solve_linear`, `binned_rms`
- [ ] `python run.py check 05` green
