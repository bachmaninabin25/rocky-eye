# 04 — The eclipse model (`rockyeye/s04_eclipse.py`, 115 lines) · Tier 1: rebuild

HOW_IT_WORKS §4 · learntopic day 1 · rebuild: `codecode/04_eclipse/`

## What it does
Gives the **shape** of a secondary eclipse in time: 1 when the planet is fully visible, 0 when
it's fully behind the star, smooth in between (ingress and egress). The flux model is
`1 + depth * shape`.

## Where it sits
Called by 06 thousands of times per second (via a lookup table), by 08 to plant fake eclipses,
and by `synth.py` to make data with a known answer.

## The key ideas
1. **Shape and depth are separate.** The shape only depends on geometry (period, a/R*,
   inclination, Rp/R*). The depth is a single number multiplied on. So one shape serves every
   depth, including the negative depths the sampler is allowed to try.
2. **It's a circle-overlap problem.** Star = disk of radius 1, planet = disk of radius k. Hidden
   fraction = overlap area / πk². The only physics is the sky separation
   `z = a/R* · sqrt(sin²φ + cos²i · cos²φ)`.
3. **Behind vs in front.** `z` is small at *both* transit and eclipse. `cos φ > 0` (φ measured
   from mid-eclipse) is what says "behind".
4. **Never trust one implementation.** The production code uses `batman` (the field-standard
   package) with `fp = 1` so it returns `1 + visible fraction`. The analytic version exists
   *only* to check batman in the tests. They agree to 1e-6.

## Annotated real code
```python
def eclipse_shape(t, t_ecl, orbit):
    p = batman.TransitParams()
    p.t_secondary = t_ecl          # batman wants the eclipse time ...
    p.t0 = t_ecl - 0.5*orbit.period  # ... and a consistent transit time
    p.fp = 1.0                     # planet flux = 1  ->  light curve = 1 + visible_fraction
    p.u = []; p.limb_dark = "uniform"   # the PLANET'S disk is uniform (limb darkening is a star thing)
    m = batman.TransitModel(p, t, transittype="secondary")
    return m.light_curve(p) - 1.0  # strip the 1 -> visible fraction
```
```python
def predicted_eclipse_time(self, t_ref):
    n = np.round((t_ref - self.t0) / self.period - 0.5)   # which orbit
    return self.t0 + self.period * (n + 0.5)              # half an orbit after transit
```

## Gotchas
- Degrees vs radians in `np.cos(inc)`: the classic silent bug.
- A circular orbit is assumed for the *shape*. GJ 3929 b's eclipse actually comes ~54 min early
  (slight eccentricity); that's absorbed by the timing parameter `dt_min` in 06, not by changing
  the shape.
- `t0` is the *transit* time from the literature, not the eclipse time.

## How it's tested
`tests/test_04_eclipse.py`: limits (0 mid-eclipse, 1 outside), batman vs analytic < 1e-6,
durations measured from the curve match the T14/T23 formulas, eclipse is at exactly phase 0.5,
nothing is hidden at transit.

## Can you explain it?
1. Why is the eclipse depth ≈ (Rp/Rs)² × (planet brightness / star brightness)?
   <details><summary>answer</summary>The planet's light is blocked; that light is its area
   (πRp²) times its surface brightness, relative to the star's area times its surface brightness.</details>
2. What's the difference between T14 and T23?
   <details><summary>answer</summary>T14: first to last contact (any part hidden). T23: second
   to third contact (fully hidden). The difference is ingress + egress.</details>
3. Why compare batman with your own geometry instead of just trusting batman?
   <details><summary>answer</summary>Two independent implementations agreeing is evidence both
   are right; it also catches misuse of batman's conventions (e.g. fp, t_secondary).</details>
4. Why does the code allow negative depths?
   <details><summary>answer</summary>Noise can make a zero eclipse look negative. Forbidding it
   would push every weak measurement upward (biased). See 06.</details>

## Rebuild checklist
- [ ] `circle_overlap_area` (three cases, vectorised)
- [ ] `eclipse_shape` (with the behind-the-star mask)
- [ ] durations and predicted time
- [ ] `python run.py check 04` green · `python run.py demo --mine 04`
