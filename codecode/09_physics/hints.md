# Hints — 09 physics

## `planck_nu`
1. **Nudge:** convert microns to metres first: `nu = C / (wl_um * 1e-6)`.
2. **Approach:** `x = H * nu / (KB * T)`; use `np.asarray` so it works for arrays of wavelength or temperature.
3. **Near-solution:** `return 2 * H * nu**3 / C**2 / np.expm1(x)`

## `Band.__init__`
1. **Nudge:** `np.trapezoid(y, x)` needs `x` increasing. Frequencies decrease as wavelength increases, so sort by `nu`.
2. **Approach:**
   ```python
   self.wl = np.linspace(lo_um, hi_um, n)
   nu = C / (self.wl * 1e-6); order = np.argsort(nu)
   self.nu, self._wl = nu[order], self.wl[order]
   self._w = 1.0 / self.nu
   self._norm = np.trapezoid(self._w, self.nu)
   ```
3. **Near-solution:** `self._T = np.geomspace(30, 20000, 2000)` and
   `self._B = np.array([np.trapezoid(planck_nu(self._wl, T) * self._w, self.nu) / self._norm for T in self._T])`.
   Pivot: `np.sqrt(np.trapezoid(self.wl, self.wl) / np.trapezoid(1/self.wl, self.wl))`.

## `planck` / `brightness_temperature`
1. **Nudge:** B(T) rises enormously fast, so interpolate in **log-log**.
2. **Near-solution:** `np.exp(np.interp(np.log(T), np.log(self._T), np.log(self._B)))`, and the inverse just swaps the roles of `_T` and `_B`.

## `dayside_temperature`
`teff / np.sqrt(a_rs) * (1 - albedo)**0.25 * (2/3 - 5/12*eps)**0.25`. With eps=0 the last factor is (2/3)^¼; with eps=1 it's (1/4)^¼.

## `depth_from_temperature` / `temperature_from_depth`
- depth = `rp_rs**2 * band.planck(T) / i_star`
- inverse: `inten = depth * i_star / rp_rs**2`; then
  `np.where(inten > 0, band.brightness_temperature(np.where(inten > 0, inten, 1.0)), np.nan)`.
  The inner `where` avoids `log` of a negative number; the outer one puts the NaN back.

## `compare_scenarios`
1. **Nudge:** `like = stats.gaussian_kde(depth_samples_ppm)`; `like(x)` evaluates the likelihood curve at depths `x`.
2. **Approach:** evidence = `np.mean(like(pred))`. That's "the likelihood averaged over what this scenario predicts."
3. **Near-solution:** loop once to fill `evidence` and `tension_sigma`, then a second loop for `bayes_factor_vs_ref = Z / Z_ref`.

## `interpret_bayes_factor`
`lb = abs(np.log(b))`, then the thresholds 1 / 2.5 / 5. Mind the exact strings; the tests compare them.
