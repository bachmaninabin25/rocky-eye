# Day 2 — Bayesian inference and MCMC

The most important day: this is the core skill, and it's the same skill quant research uses.
Goal: by tonight you can write a likelihood, sample it, and tell whether the samples can be
trusted. Every snippet runs on its own (`numpy`, `scipy`, `emcee`).

---

## 1. The vocabulary, precisely

```
P(θ | data)  =  P(data | θ) · P(θ)  /  P(data)
 posterior        likelihood   prior     evidence
```
- **Likelihood** L(θ) = P(data | θ): how probable the data are *if* the parameters were θ.
  Not "the probability θ is right."
- **Prior**: what you allow before seeing the data. RockyEye uses flat boxes ("depth anywhere in
  −500…1000 ppm").
- **Posterior**: what you believe afterwards. Its spread *is* the error bar.
- **Evidence** P(data): normalisation for one model, but the key quantity when comparing models.

Gaussian noise with standard deviation σ, model m(t; θ):
```
ln L = −½ Σ (y_i − m_i)²/σ² − N ln σ − (N/2) ln 2π
```
The first term is −χ²/2. Maximising L with σ fixed = least squares.

## 2. A toy eclipse fit, by brute force first

```python
import numpy as np
rng = np.random.default_rng(1)
t = np.linspace(-0.1, 0.1, 400)                     # days
box = (np.abs(t) > 0.025).astype(float)             # crude eclipse: 1 out, 0 in
true_depth, sigma = 150e-6, 250e-6
y = 1 + true_depth*box + rng.normal(0, sigma, t.size)

def lnL(depth):
    shape = 1 + depth*box
    base = np.sum(y*shape)/np.sum(shape**2)      # best baseline for this depth (least squares)
    return -0.5*np.sum((y - base*shape)**2)/sigma**2

depths = np.linspace(0, 300e-6, 301)
like = np.array([lnL(d) for d in depths])
post = np.exp(like - like.max()); post /= np.trapezoid(post, depths)
mean = np.trapezoid(depths*post, depths); sd = np.sqrt(np.trapezoid((depths-mean)**2*post, depths))
print(f"depth = {mean*1e6:.0f} +/- {sd*1e6:.0f} ppm (truth 150)")
```

A grid works in 1–2 dimensions. RockyEye has ~10–20 dimensions: a grid with 100 points per axis
would need 10²⁰ evaluations. That's why MCMC exists.

## 3. Integrating out "boring" linear parameters

The baseline `base` above is a **nuisance parameter**: you need it, you don't care about it. When
the model is *linear* in such parameters (y ≈ M c), the integral over them has a closed form:

```
∫ L dc  ∝  σ^-(N−K) · exp(−χ²_min / 2σ²) · det(MᵀM)^-½
```
χ²_min = least-squares residual, K = number of linear parameters. So you solve the linear part
exactly (least squares) and sample only the non-linear parameters. Check it numerically:

```python
M = np.column_stack([np.ones_like(t), t])                # baseline + slope, linear
yy = 1 + 2e-3*t + rng.normal(0, sigma, t.size)
c, res, *_ = np.linalg.lstsq(M, yy, rcond=None)
chi2 = float(res[0]); N, K = M.shape
analytic = -(N-K)*np.log(sigma) - chi2/(2*sigma**2) - 0.5*np.linalg.slogdet(M.T@M)[1] + (K/2)*np.log(2*np.pi)
# brute force over a 2-D grid of (c0, c1)
c0 = np.linspace(c[0]-6e-5, c[0]+6e-5, 301); c1 = np.linspace(c[1]-1e-3, c[1]+1e-3, 301)
G0, G1 = np.meshgrid(c0, c1, indexing="ij")
r = yy[None, None, :] - (G0[..., None] + G1[..., None]*t)
ll = -0.5*np.sum(r**2, axis=2)/sigma**2 - N*np.log(sigma)
brute = np.log(np.trapezoid(np.trapezoid(np.exp(ll-ll.max()), c1, axis=1), c0)) + ll.max()
print("analytic", round(analytic, 3), " brute force", round(brute, 3))
```
The two agree. This is exactly `log_like` in the project (with the ½ ln 2π constants arranged
slightly differently), and it's why the sampler only explores depth, timing, noise and ramp
timescales instead of 60+ coefficients.

## 4. MCMC in one paragraph, then in code

MCMC = a random walk through parameter space that spends time in each region **in proportion to
its posterior probability**. Collect the positions → a histogram of them *is* the posterior.

Metropolis, the simplest version: propose a jump; accept it with probability
min(1, P_new/P_old). Downhill jumps are sometimes accepted, which is what lets it map the whole
peak instead of just finding its top.

```python
def lnpost(d):
    if not -500e-6 < d < 1000e-6:              # flat prior box; negative depth allowed!
        return -np.inf
    return lnL(d)

chain, d, lp = [], 0.0, lnpost(0.0)
for _ in range(20000):
    prop = d + rng.normal(0, 20e-6)
    lp_new = lnpost(prop)
    if np.log(rng.uniform()) < lp_new - lp:
        d, lp = prop, lp_new
    chain.append(d)
chain = np.array(chain[5000:])                  # drop burn-in
print(f"Metropolis: {chain.mean()*1e6:.0f} +/- {chain.std()*1e6:.0f} ppm")
```

**emcee** (what RockyEye uses): many "walkers" that propose jumps using each other's positions,
so they adapt automatically to stretched or correlated posteriors. No step size to tune.

```python
import emcee
nw = 16
p0 = 150e-6 + 1e-6*rng.normal(size=(nw, 1))
s = emcee.EnsembleSampler(nw, 1, lambda p: lnpost(p[0]))
s.run_mcmc(p0, 3000)
flat = s.get_chain(discard=1000, flat=True)[:, 0]
print(f"emcee: {np.median(flat)*1e6:.0f} +/- {flat.std()*1e6:.0f} ppm")
```

**Why allow negative depths?** If the truth is ~0, noise pushes measurements both ways. A prior
that forbids negatives throws away the low half and the average moves up: you'd "detect" eclipses
that aren't there.

## 5. Can you trust the chain? Convergence

A chain can look fine and be wrong. Warning lights (RockyEye requires all of them off):

- **Autocorrelation time τ**: steps per independent sample. 20,000 steps with τ = 2000 is only 10
  independent samples. Rule: chain length ≥ 50 τ.
- **Split R̂**: cut each walker's chain in half; compare variance *between* pieces with variance
  *within*. All sampling the same thing → R̂ ≈ 1. RockyEye: ≤ 1.01.
- **Effective sample size** = samples / τ.
- **Acceptance fraction**: ~0 = stuck, ~1 = steps too small to go anywhere.

```python
ch = s.get_chain()                             # (steps, walkers, dim)
print("tau ~", emcee.autocorr.integrated_time(ch, quiet=True))
half = ch.shape[0]//2
seq = np.concatenate([ch[:half], ch[half:2*half]], axis=1)
W = seq.var(axis=0, ddof=1).mean(axis=0); B = half*seq.mean(axis=0).var(axis=0, ddof=1)
print("split R-hat", np.sqrt(((half-1)/half*W + B/half)/W))
```
(This short chain includes the burn-in, so R̂ may come out a bit above 1.01. Try it with
`ch[1000:]`, and try a 300-step chain to watch the warnings fire.)

## 6. Comparing models: BIC and Bayes factors

- **BIC** = k ln N − 2 ln L_max. More parameters always fit better; BIC charges ln N for each.
  Lower is better; ΔBIC < 2 means nothing, > 10 is strong. RockyEye uses it to pick systematics
  models ("fewest parameters within ΔBIC 2") and to ask "is there an eclipse at all?"
  (depth = 0 vs free: ΔBIC > 10 = detection).
- **Bayes factor** = Z₁ / Z₂, the ratio of evidences. For a scenario with no free parameters that
  predicts a depth distribution: Z = average of the likelihood over that prediction. That's how
  "bare rock vs atmosphere" becomes a number.

```python
from scipy import stats
like = stats.gaussian_kde(flat*1e6)              # posterior with flat prior ∝ likelihood
rock = rng.normal(125, 6, 4000); atmo = rng.normal(78, 5, 4000)   # predicted depths (ppm)
Z_rock, Z_atmo = like(rock).mean(), like(atmo).mean()
print("Bayes factor rock/atmosphere:", round(Z_rock/Z_atmo, 1))
```

## 7. The trap list (all of these happened or nearly happened in RockyEye)
1. Stopping a chain on length alone → R̂ still 1.015. Stop on *all* criteria.
2. Evaluating a model with a parameter outside its prior → every point rejected → nonsense ΔBIC.
3. A model too simple for the noise → error bars too small by 3×, invisible to BIC. Check with
   injection tests: does the *scatter* of recovered values match the quoted error?
4. Pass/fail bars at 1σ fail a correct analysis 32% of the time. Use 2σ.

## Readings
1. Hogg, Bovy & Lang (2010), *Data analysis recipes: Fitting a model to data*, arXiv:1008.4686. Opinionated and practical.
2. Foreman-Mackey et al. (2013), *emcee: The MCMC Hammer*, PASP 125, 306 (arXiv:1202.3665).
3. Hogg & Foreman-Mackey (2018), *Data analysis recipes: Using Markov Chain Monte Carlo*, ApJS 236, 11 (arXiv:1710.06068). Convergence, in plain language.

## Self-check (answers at the bottom)
1. Why is "likelihood" not "probability that the parameters are correct"?
2. Why does integrating out linear parameters change `N` to `N − K` in the σ term?
3. A chain has 50,000 steps and τ = 4,000. Is it long enough by RockyEye's rule?
4. Two systematics models: ΔBIC = 1.5 in favour of the one with 3 more parameters. Which do you pick?
5. Why can a flat prior on depth that excludes negatives bias the result?

---
---

### Answers
1. It's P(data | θ). Turning it into P(θ | data) needs a prior and normalisation (Bayes' theorem).
2. Each integrated linear direction absorbs one factor of σ from the Gaussian normalisation;
   K dimensions of the N-dimensional Gaussian are "used up" by the fit.
3. No: 50,000 / 4,000 = 12.5 τ, far below 50 τ.
4. The simpler one: ΔBIC < 2 isn't meaningful, and ties go to fewer parameters.
5. With a true depth near zero, noise scatters estimates both ways; cutting off the negative side
   shifts the posterior up, which would manufacture detections.
