# Hints — 05 systematics

## `parse_model`
1. **Nudge:** `name.partition("+")` gives `(before, "+", after)`, and `after` is `""` if there's no `+`.
2. **Approach:** loop over `("bg", "x", "y", "w")`. If the key is in the leftover string, record it and remove it once (`rest.replace(key, "", 1)`). Whatever is left at the end is junk, so raise `ValueError`.
3. **Near-solution:** `return base, tuple(sorted(extras, key=EXTRA_ORDER.index))`. Why `bg` goes first: removing multi-letter keys before single letters means no key can ever eat part of another one. It's a cheap habit that prevents a whole class of parsing bugs.

## `aux_columns`, `n_linear`, `nonlinear_names`
- `aux_columns`: `tuple(c for e in extras for c in AUX_KEYS[e])`
- `n_linear`: `BASELINES[base][0] + len(aux_columns(name))`
- `nonlinear_names`: `BASELINES[base][1]`

## `basis`
1. **Nudge:** build a Python list `cols`, append arrays in the documented order, and finish with `np.column_stack(cols)`.
2. **Approach:** the time column is `tc = (th - th.mean()) / max(np.ptp(th), 1e-12)`. Centring makes the constant and the slope nearly independent, which keeps the matrix well conditioned. The ramp is `np.exp(-th / np.exp(theta[0]))` (theta is log tau, and the ramp uses *raw* `th`, not `tc`, so it starts at exactly 1).
3. **Near-solution:** after the baseline columns: `cols += [aux[c] for c in aux_columns(name)]`.

## `standardise`
1. **Nudge:** median and MAD instead of mean and std.
2. **Approach:** `mad = 1.4826 * np.nanmedian(np.abs(v - med))`. Fallback: `if not np.isfinite(mad) or mad == 0: mad = np.nanstd(v) or 1.0`.
3. **Near-solution:** `return np.clip((v - med) / mad, -8, 8)`

## `solve_linear`
1. **Nudge:** `q, r = np.linalg.qr(M)`.
2. **Approach:** `c = np.linalg.solve(r, q.T @ y)`; residual `y - M @ c`; `chi2 = resid @ resid`.
3. **Near-solution:** `logdet = 2 * np.sum(np.log(np.abs(np.diag(r))))`, because det(MᵀM) = det(RᵀR) = (∏ Rᵢᵢ)². Check it against `np.linalg.slogdet(M.T @ M)[1]` once.

## `binned_rms`
1. **Nudge:** for each bin size `s`, reshape the first `nb*s` points to `(nb, s)` and average along axis 1.
2. **Approach:** `sizes = np.unique(np.round(np.geomspace(1, max(max_bin, 2), 30)).astype(int))`; append `np.nan` when `nb <= 1`.
3. **Near-solution:** `nb = n // sizes`; `expected = rms[0] / np.sqrt(sizes) * np.sqrt(nb / np.maximum(nb - 1, 1))`.

## If `test_reaches_white_noise_floor` fails
Your basis or solve is off. Print `B[:3]` and compare with `ref().basis(...)` in a Python shell.
