# 09 — Physics: depth → verdict (`rockyeye/s09_physics.py`, 227 lines) · Tier 1: rebuild

HOW_IT_WORKS §9 · learntopic day 1 · rebuild: `codecode/09_physics/`

## What it does
Converts the measured eclipse depth into the planet's dayside **brightness temperature**, compares
it with what a bare rock vs an atmosphere would give, and turns that into Bayes factors and words.

## The chain
```
depth = (Rp/Rs)² × <B_ν(T_planet)> / <I_star>        (band-averaged over F1500W)
T_day = T_star/√(a/Rs) × (1−A)^¼ × (2/3 − 5/12·ε)^¼   (Cowan & Agol 2011)
```
GJ 3929 b: dark bare rock (A=0, ε=0) → 737 K; full redistribution (ε=1) → 577 K.

## The key ideas
1. **Band averages, photon-weighted.** MIRI counts photons, so F_ν is averaged with weight
   (response/ν). The real code uses the actual F1500W throughput curve; your lean version uses a
   top-hat 13.5–16.6 µm.
2. **Tabulate, then interpolate in log-log.** <B>(T) is computed once on 2000 temperatures;
   brightness temperature is the inverse interpolation.
3. **The star matters as much as the planet.** The depth is a ratio, so an error in the star's
   15 µm brightness is an error in T_planet. The real code supports three stellar models (blackbody
   at T_eff; from the *measured* flux + radius + distance; a literature value) and runs all of them.
4. **Uncertainty propagation by Monte Carlo.** Draw T_eff, a/Rs, Rp/Rs from their errors
   (`draw_system`) → each scenario predicts a *distribution* of depths.
5. **Evidence = average likelihood over the prediction.** Flat depth prior in 06 ⇒ the
   posterior's KDE *is* the likelihood curve. Z = mean of L(predicted depths). Bayes factor = Z₁/Z₂.
   Jeffreys-style words on |ln B|: <1 none, <2.5 weak, <5 moderate, else strong.
6. Also reported: R = T_b/T_max (1 = bare rock), effective albedo 1 − R⁴, and a 95% upper limit
   on ε (`infer_redistribution`).

## Annotated real code
```python
def compare_scenarios(depth_samples_ppm, predicted, reference):
    like = stats.gaussian_kde(depth_samples_ppm)        # L(depth), keeps any skew
    for name, pred in predicted.items():
        z = float(np.mean(like(pred[:4000])))           # Z = E_scenario[ L(depth) ]
        ... "tension_sigma": (d_med - mean(pred)) / hypot(d_std, std(pred))
    ... bayes_factor_vs_ref = Z / Z_ref
```

## Gotchas
- `np.expm1` in Planck: accurate when hν ≪ kT.
- Negative depth → no temperature (NaN), not a crash.
- Stellar values were verified against papers (Kemmer+22) before use; an earlier draft had
  numbers "from memory" from a different paper (Beard+22). Always cite the table you used.

## Can you explain it?
1. Why is a bare rock's dayside the hottest possible?
   <details><summary>answer</summary>No atmosphere to carry heat to the night side (ε=0), and a
   dark surface reflects nothing (A=0): all absorbed starlight re-radiates from the dayside.</details>
2. What is a brightness temperature?
   <details><summary>answer</summary>The temperature a blackbody would need to be as bright as the
   planet in this band. Not necessarily the "real" surface temperature.</details>
3. Why a Bayes factor instead of "the depth is 2σ from the atmosphere prediction"?
   <details><summary>answer</summary>It compares how well each scenario predicts the data,
   including the scenarios' own uncertainty, and gives a ratio you can interpret consistently.
   The tension in σ is reported too, as a sanity check.</details>
4. A 3% error in the star's flux: what happens to the planet's temperature?
   <details><summary>answer</summary>It feeds straight into the planet's inferred intensity (3%).
   At 15 µm and ~740 K, x = hν/kT ≈ 1.3 and d ln I / d ln T = x·eˣ/(eˣ−1) ≈ 1.8, so T shifts by
   about 3%/1.8 ≈ 1.7%, roughly 12 K. Small, but it's a systematic (same direction every time),
   so more data won't average it away.</details>

## Rebuild checklist
- [ ] `planck_nu`, `Band` (sorted ν, weights, table, pivot)
- [ ] temperatures both ways, `dayside_temperature`
- [ ] `compare_scenarios`, `interpret_bayes_factor`
- [ ] `python run.py check 09` green · `python run.py demo --mine all` (the finish line)
