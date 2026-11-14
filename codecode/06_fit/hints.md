# Hints — 06 fit

Do them in this order: `t_ecl` → `design` → `log_prior` → `log_like` (now 5 of 7 fast tests pass) → `max_likelihood` → `run_mcmc`.

## `t_ecl`
`return self.t_pred[i] + p[1] / 1440.0`. There are 1440 minutes in a day.

## `design`
1. **Nudge:** `self.slices[i]` is a Python `slice`; `p[self.slices[i]]` is that visit's block `[log_sigma, log_tau, ...]`.
2. **Approach:** `theta = p[self.slices[i]][1:]`; build `B` with `SYS.basis`; build the eclipse factor `ecl = 1 + p[0]*1e-6*ECL.eclipse_shape(...)`.
3. **Near-solution:** `return ecl[:, None] * B`. The `[:, None]` turns `(n,)` into `(n, 1)` so it multiplies every column row by row.

## `log_prior`
1. **Nudge:** return `-np.inf` as soon as anything is out of bounds; return `0.0` at the end.
2. **Approach:** check `p[0]` and `p[1]`, then loop `for sl in self.slices:` with `q = p[sl]`, `q[0]` = log_sigma, `q[1:]` = taus.
3. **Near-solution:** `if len(taus) == 2 and taus[1] <= taus[0]: return -np.inf`

## `log_like`
1. **Nudge:** it's a sum over visits of the formula in the module docstring.
2. **Approach:**
   ```python
   M = self.design(p, i)
   _, chi2, logdet = SYS.solve_linear(v.flux, M)
   ls = p[self.slices[i]][0]
   ```
3. **Near-solution:**
   ```python
   total += -(v.n - M.shape[1]) * (ls + 0.5 * LOG2PI) - 0.5 * chi2 * np.exp(-2 * ls) - 0.5 * logdet
   ```
   **Why `N - K` and not `N`:** integrating out K linear coefficients "uses up" K of the N Gaussian normalisation factors. `test_marginal_likelihood_matches_brute_force` checks exactly this against numerical integration.

## `max_likelihood`
1. **Nudge:** write two inner functions: `expand(x)`, which turns the optimiser's short vector into a full `p`, and `obj(x)`.
2. **Approach:**
   - `free` = list of indices to optimise. Start with `[0, 1]`, drop both if `depth_fixed == 0`, then add each visit's tau indices `range(sl.start + 1, sl.stop)`.
   - bounds: `depth_lim` for 0, `dt_lim` for 1, `log_tau_lim` for the rest.
   - in `expand`, set `p[0] = depth_fixed or 0`, park `p[1]` mid-prior, set every `log_sigma` to something in-prior like `-7` (it doesn't matter, sigma is profiled), then `p[free] = x`.
   - `obj` returns `1e30` if `log_prior(p)` is `-inf`, else `sum(v.n * np.log(chi2_v / v.n))`.
3. **Near-solution:** after `res = optimize.differential_evolution(obj, bounds, seed=seed, tol=1e-8, popsize=20, maxiter=400, polish=True)`, loop over visits to compute `s2 = chi2/n`, set `p[sl.start] = 0.5*np.log(s2)`, accumulate `lnL += -0.5*n*(LOG2PI + np.log(s2) + 1)`. Then
   `k = len(free) + len(self.visits) + sum(SYS.n_linear(v.model) for v in self.visits)`.

## `run_mcmc`
1. **Nudge:** `import emcee` inside the function.
2. **Approach:** `scale = np.array([5.0 if n.startswith("depth") else 1.0 if n.startswith("dt") else 0.02 for n in self.names])`; `start = p0 + scale * rng.standard_normal((nw, ndim))`; `while not np.isfinite(self.log_prior(start[k])): redraw`.
3. **Near-solution:**
   ```python
   sampler = emcee.EnsembleSampler(nw, self.ndim, self.log_prob,
                                   moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
   sampler.run_mcmc(start, n_steps)
   flat = sampler.get_chain(discard=n_steps // 3, flat=True)
   ```

## Debugging tips
- `python run.py check 06 --fast` while working; the full check includes a ~2 min MCMC.
- If the MCMC depth is way off but `max_likelihood` is right, your walkers start outside the prior or your `log_prob` returns NaN. Check `np.isfinite(fit.log_prob(p0))`.
