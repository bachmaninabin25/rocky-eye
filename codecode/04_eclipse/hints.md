# Hints — 04 eclipse

Open one tier at a time. Try for 20 minutes before going down a tier.

## `predicted_eclipse_time`
1. **Nudge:** transits happen at `t0 + n*P`. Eclipses are half an orbit later.
2. **Approach:** count how many periods (minus a half) `t_ref` is from `t0`, round to the nearest integer `n`, then rebuild the time.
3. **Near-solution:**
   ```python
   n = np.round((t_ref - self.t0) / self.period - 0.5)
   return self.t0 + self.period * (n + 0.5)
   ```

## `duration_full` / `duration_total`
1. **Nudge:** both are the same formula with `1+k` or `1-k`. Write a private helper `_dur(self, kk)`.
2. **Approach:** `b = self.impact_parameter`, `sin_i = np.sin(np.radians(self.inc))`. Angles in degrees are the classic bug.
3. **Near-solution:** `self.period / np.pi * np.arcsin(np.sqrt(kk**2 - b**2) / (self.a_rs * sin_i))`

## `circle_overlap_area`
1. **Nudge:** start with `area = np.zeros_like(d)` and fill it using boolean masks for the three cases.
2. **Approach:** `inside = d <= r1 - r2`; `partial = ~inside & (d < r1 + r2)`; apply the lens formula only to `x = d[partial]`, so you never take `arccos` of something outside [-1, 1] and never divide by `d = 0`.
3. **Near-solution:** the three-term formula from the docstring with `x` in place of `d`, assigned to `area[partial]`. Remember `d = np.abs(np.asarray(d, float))` first.

## `eclipse_shape`
1. **Nudge:** follow the five numbered steps in the docstring literally.
2. **Approach:** `phi`, then `z` (separation in stellar radii), then `hidden = circle_overlap_area(1.0, k, z) / (np.pi * k**2)`. The "behind the star" condition is `np.cos(phi) > 0`.
3. **Near-solution:**
   ```python
   return 1.0 - np.where(np.cos(phi) > 0, hidden, 0.0)
   ```
   If `test_transit_is_not_an_eclipse` fails, you forgot this `where`. At transit `z` is also small, so without it you'd "hide" the planet while it's in front of the star.

## `eclipse_flux`
`return 1.0 + depth * eclipse_shape(t, t_ecl, orbit)`

## If `test_matches_batman` fails by ~1e-3
You're probably using degrees inside `np.cos`. Use `np.radians(orbit.inc)`.
