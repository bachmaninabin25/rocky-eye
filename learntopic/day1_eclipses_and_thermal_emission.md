# Day 1 — Eclipses and thermal emission

Goal for today: explain, from scratch, why a dip of ~0.015% in a star's infrared light tells you
whether a planet has an atmosphere. No project code today; every snippet here runs on its own
(`numpy` only, plus `matplotlib` if you want plots).

---

## 1. Transit vs secondary eclipse

```
            observer  ←───────────────────────────────

 TRANSIT                         SECONDARY ECLIPSE
 (planet in front)               (planet behind)

      ☀  ●                              ● ☀
   star dims: planet blocks         total light drops: the star
   a bit of STARLIGHT               blocks the PLANET'S OWN light
   depth ≈ (Rp/Rs)²                 depth ≈ (Rp/Rs)² × (planet brightness / star brightness)
   ~1000 ppm for GJ 3929 b          ~100–150 ppm for GJ 3929 b
```

A transit tells you the planet's **size**. A secondary eclipse tells you its **brightness**, and
at 15 µm (mid-infrared) a ~700 K planet's brightness is its own thermal glow. That's the whole
trick: the eclipse depth is a thermometer.

## 2. Geometry: when does the planet hide?

Circular orbit, phase φ measured from mid-eclipse. The planet's centre is at sky distance

```
z(φ) = (a/R*) · sqrt( sin²φ + cos²i · cos²φ )      (in units of the star's radius)
```

- `a/R*` = orbit size in star radii (17 for GJ 3929 b, so a tight orbit)
- `i` = inclination (90° = edge-on)
- `b = (a/R*) cos i` = impact parameter: how far off-centre the path crosses

The planet (radius k = Rp/R*) is hidden where it overlaps the star's disk, **and only when it's
behind** (cos φ > 0). Hidden fraction = overlap area of two circles / πk².

```python
import numpy as np

def overlap(r1, r2, d):
    """Area of overlap of circles r1 >= r2 at distance d."""
    d = np.abs(np.asarray(d, float)); a = np.zeros_like(d)
    a[d <= r1 - r2] = np.pi * r2**2
    m = (d > r1 - r2) & (d < r1 + r2); x = d[m]
    a[m] = (r1**2*np.arccos((x**2+r1**2-r2**2)/(2*x*r1)) + r2**2*np.arccos((x**2+r2**2-r1**2)/(2*x*r2))
            - 0.5*np.sqrt((-x+r1+r2)*(x+r1-r2)*(x-r1+r2)*(x+r1+r2)))
    return a

P, a_rs, inc, k = 2.616, 17.05, 89.3, 0.0318          # GJ 3929 b-ish
t = np.linspace(-0.06, 0.06, 2001)                     # days from mid-eclipse
phi = 2*np.pi*t/P
z = a_rs*np.sqrt(np.sin(phi)**2 + (np.cos(np.radians(inc))*np.cos(phi))**2)
hidden = np.where(np.cos(phi) > 0, overlap(1.0, k, z)/(np.pi*k**2), 0.0)
shape = 1 - hidden                                     # 1 = visible, 0 = fully hidden
print("eclipse lasts", round(np.ptp(t[shape < 1])*24, 2), "h; fully hidden for",
      round(np.ptp(t[shape == 0])*24, 2), "h")
```

Output: about 1.18 h total, 1.11 h fully hidden. Ingress and egress are short because the planet is
tiny compared with the star.

**Durations by formula** (check them against the code above):
```
T14 = P/π · arcsin( sqrt((1+k)² − b²) / (a/R* · sin i) )      first → fourth contact
T23 = same with (1−k)                                          planet fully hidden
```

## 3. How bright is a hot planet at 15 µm? Planck

A blackbody's spectral radiance (per frequency):
```
B_ν(T) = 2hν³/c² · 1/(exp(hν/kT) − 1)
```

```python
h, c, kB = 6.626e-34, 2.998e8, 1.381e-23
def B_nu(wl_um, T):
    nu = c/(wl_um*1e-6)
    return 2*h*nu**3/c**2/np.expm1(h*nu/(kB*T))

for T in (300, 577, 737, 3369):
    print(f"T={T:5d} K   B_nu(15um) = {B_nu(15, T):.3e}")
# the star (3369 K) is far brighter per area, but only ~5-10x at 15 um, not 1000x as in the optical
```

Eclipse depth = (Rp/Rs)² × B_ν(T_planet) / B_ν(T_star). With k = 0.0318:
```python
for Tp in (577, 737):
    print(Tp, "K ->", round(1e6*k**2*B_nu(15, Tp)/B_nu(15, 3369)), "ppm")
```
About 78 and 125 ppm. That ~47 ppm gap is what the whole project is trying to resolve, with
error bars of ~15–20 ppm. (Real stars aren't perfect blackbodies at 15 µm, which is why the project
also uses the star's *measured* flux. Day 1 version: blackbody is fine.)

**Band averaging.** The detector sees a band (F1500W ≈ 13.5–16.6 µm), not one wavelength, and
counts photons, so the average weights F_ν by 1/ν:
```python
wl = np.linspace(13.5, 16.6, 400); nu = c/(wl*1e-6); o = np.argsort(nu)
def band_avg(T):
    return np.trapezoid(B_nu(wl[o], T)/nu[o], nu[o]) / np.trapezoid(1/nu[o], nu[o])
print(band_avg(737)/B_nu(15, 737))     # close to 1: the band average ≈ the value at ~15 um
```

**Brightness temperature** = the T at which a blackbody matches the measured brightness. Invert
`band_avg` by interpolation. That's what you report: "the dayside's brightness temperature is X K."

## 4. Energy balance: what temperature *should* the dayside have?

The planet absorbs starlight and re-radiates it. Cowan & Agol (2011):
```
T_day = T_star / sqrt(a/R*) · (1 − A_B)^¼ · (2/3 − 5/12 · ε)^¼
```
- `A_B` = Bond albedo (fraction of light reflected). Dark rock ≈ 0.
- `ε` = heat redistribution efficiency. 0 = each spot re-radiates where it absorbed (no wind,
  no ocean: **bare rock**). 1 = heat spread evenly over the whole planet (**thick atmosphere**).

```python
def T_day(teff, a_rs, A=0.0, eps=0.0):
    return teff/np.sqrt(a_rs) * (1-A)**0.25 * (2/3 - 5/12*eps)**0.25
print(round(T_day(3369, 17.05)), "K bare rock;", round(T_day(3369, 17.05, eps=1)), "K full redistribution")
# 737 K and 577 K
```

Why 2/3 and 1/4: with ε = 1 the absorbed power (πR² F) is radiated from the whole sphere (4πR²) →
factor 1/4. With ε = 0, each dayside point re-radiates locally; averaged over the dayside as seen at
eclipse the factor becomes 2/3. The dayside of a bare rock is the **hottest possible**.

## 5. Why rocky planets around M dwarfs, why now

- M dwarfs are small and cool: the planet/star contrast at 15 µm is the best you can get.
- The question "do rocky planets around M dwarfs keep atmospheres at all?" is open. These stars
  flare and blast planets with X-rays/UV, which may strip atmospheres. A hot dayside (deep eclipse)
  = bare rock; a cooler one = something carrying heat away.
- JWST's MIRI at 15 µm is the first instrument sensitive enough for ~100 ppm eclipses on rocky
  worlds. TRAPPIST-1 b and c (2023) came out consistent with no thick atmosphere. Rocky Worlds DDT
  is a 500-hour program doing this systematically.

## Readings
1. Winn (2010), *Transits and Occultations*, arXiv:1001.2010. §1–3: the standard reference for the geometry.
2. Cowan & Agol (2011), *The Statistics of Albedo and Heat Recirculation on Hot Exoplanets*, ApJ 729, 54 (arXiv:1011.0428). The T_day formula.
3. Koll et al. (2019), *Identifying Candidate Atmospheres on Rocky M Dwarf Planets via Eclipse Photometry*, ApJ 886, 140 (arXiv:1907.13145). Exactly this project's logic.

## Self-check (answers at the bottom)
1. Why is a secondary eclipse at 15 µm a thermometer, while one in visible light mostly isn't?
2. A planet is twice as far from the same star. Bare-rock T_day changes by what factor?
3. The measured depth is 140 ± 20 ppm; bare rock predicts 125, full redistribution 78. What would you say?
4. Why must the "behind the star" condition (cos φ > 0) be in the shape model?
5. Name one reason a planet could have a *shallower* eclipse than a dark bare rock without having an atmosphere.

---
---

### Answers
1. At 15 µm a ~700 K planet emits a lot of its own light, so the depth measures its thermal
   brightness. In visible light the planet emits almost nothing; the depth there is reflected light (albedo).
2. T ∝ 1/sqrt(a), so 1/√2 ≈ 0.71×.
3. 0.75σ above bare rock, 3.1σ above full redistribution: consistent with a bare rock, a thick
   atmosphere is disfavoured. Day 2 makes this a Bayes factor.
4. At transit the planet's centre is also close to the star's centre (z small). Without the
   condition the model would "hide" the planet while it's in front.
5. A reflective surface (A > 0, e.g. fresh ice or bright minerals) lowers T_day without any heat transport.
