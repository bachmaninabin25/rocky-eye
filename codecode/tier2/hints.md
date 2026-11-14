# Hints — Tier 2 exercises

## 03 `centroid_and_width`
- Window: `cube[:, cy-box:cy+box+1, cx-box:cx+box+1] - bg[:, None, None]` with `cx, cy = int(round(x0)), int(round(y0))`.
- Coordinates: `xs = np.arange(cx-box, cx+box+1, dtype=float)`, and `ys` likewise.
- Profiles: `px = sub.sum(axis=1)` (collapse rows, so you're left with x), `py = sub.sum(axis=2)`.
- `x = (px * xs).sum(axis=1) / tot`; width `np.sqrt(np.abs((px * (xs[None] - x[:, None])**2).sum(axis=1) / tot))`.

## 07 `split_rhat`
- `seq = np.concatenate([chain[:half], chain[half:2*half]], axis=1)`
- `W = seq.var(axis=0, ddof=1).mean(axis=0)` and `Bn = n * seq.mean(axis=0).var(axis=0, ddof=1)`
- `np.sqrt(((n-1)/n * W + Bn/n) / W)`

## 08 `inject_into_stamps`
- `scale = (1.0 + depth * (shape - 1.0)).astype(np.float32)[:, None, None]`
- `sky = bg.astype(np.float32)[:, None, None]`; `return sky + (sci - sky) * scale`

## 08 `linear_depth`
- `M = np.column_stack([B, s])`, QR solve, `s2 = chi2 / (n - M.shape[1])`
- `ri = np.linalg.inv(r)`; `cov = s2 * ri @ ri.T`
- `base = np.median(B @ c[:-1])`; return `1e6*c[-1]/base, 1e6*np.sqrt(cov[-1,-1])/base`
- Why the error bar is honest: the test injects 60 fake eclipses and checks the scatter of the answers matches the quoted error.
