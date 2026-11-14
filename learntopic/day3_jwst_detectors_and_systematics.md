# Day 3 — JWST detectors and systematics

Goal: understand where the raw numbers come from, why the light curve is never flat, and how
you separate the detector's misbehaviour from a 100 ppm planet. This is the "real research skill"
day. In quant terms: the signal is tiny and the structured noise is huge, so model the noise.

---

## 1. From photons to numbers: how MIRI reads

MIRI's detector doesn't take snapshots. During one **integration** it reads every pixel many times
**without resetting**: a "ramp" of accumulated charge, up the ramp.

```
charge
  │                 ●  ← group 45
  │             ●
  │         ●           slope = count rate (what we want)
  │     ●
  │ ●  ← group 1
  └──────────────────── time within one integration (~10 s)
```

- **group** = one read; GJ 3929 b: 47 groups per integration (the simulation uses 45).
- **integration** = one ramp → one image after fitting the slope.
- **segment** = a file of a few hundred integrations (~1–2 GB raw).
- **visit** = one eclipse observation, ~8 h, thousands of integrations.

The official `jwst` pipeline turns ramps into images:
- **Stage 1 (Detector1)**: saturation flags, reference-pixel and dark corrections, **jump detection**
  (a cosmic ray makes the ramp jump; flag it), then **ramp fitting** → `rateints` (slope per
  integration).
- **Stage 2 (Image2)**: flat-field and flux calibration → `calints`.

Then *your* code takes over: cut an 81×81 px stamp around the star, do photometry.

## 2. Photometry in one picture

```
   ┌───────────────────────────┐
   │      sky annulus (ring)    │   sky level = robust mean of ring pixels
   │   ┌───────────────────┐   │
   │   │   ○ aperture ○    │   │   star flux = Σ (pixel − sky) inside the circle,
   │   │      ★            │   │   with EXACT partial-pixel weights at the edge
   │   └───────────────────┘   │
   └───────────────────────────┘
```

```python
import numpy as np
rng = np.random.default_rng(0)
yy, xx = np.mgrid[:41, :41]
def frame(x0, y0, flux=1e5, sky=50.0, w=1.6):
    psf = np.exp(-((xx-x0)**2 + (yy-y0)**2)/(2*w**2)); psf *= flux/psf.sum()
    return psf + sky + rng.normal(0, 3, psf.shape)

def crude_aperture(img, x0, y0, r=5, rin=12, rout=18):
    d = np.hypot(xx-x0, yy-y0)
    sky = np.median(img[(d > rin) & (d < rout)])
    return np.sum(img[d < r] - sky)          # all-in / all-out pixels: the crude version

f1 = crude_aperture(frame(20.0, 20.0), 20.0, 20.0)
f2 = crude_aperture(frame(20.03, 20.0), 20.03, 20.0)   # star moved 0.03 px
print("flux change from a 0.03 px move:", round(1e6*(f2/f1 - 1)), "ppm")
```
Run it a few times: changes of hundreds to thousands of ppm, from noise *and* from pixels flipping
in and out of the circle. The eclipse is 150 ppm. RockyEye uses exact overlap weights so pixel flipping disappears, but the lesson
stands: **position changes leak into the flux**, at a level comparable to the eclipse.

## 3. The systematics zoo (why the light curve is never flat)

| Effect | Looks like | Model column |
|---|---|---|
| detector settling ("ramp", persistence) | fast exponential at the start of a visit, ~0.1–1 h | `exp(−t/τ)`; also trim the first chunk |
| slow drift | linear / quadratic trend over hours | `t`, `t²` |
| pointing jitter | flux follows star position x, y | `x`, `y` (decorrelation) |
| PSF "breathing" | flux follows PSF width | `sx`, `sy` |
| sky / background changes | flux follows sky level | `bg` |
| split exposure | a jump in level at a gap | step function after the gap |
| cosmic rays, bad pixels | single-point outliers | repaired in photometry, clipped |

The full model: **flux = eclipse(t) × systematics(t) + white noise**, with
systematics = Σ cₖ Bₖ(t). Linear in the c's (Day 2, §3): solve them exactly, sample only τ.

```python
t = np.linspace(0, 8, 2000)                           # hours
x = rng.normal(size=t.size)                           # standardised centroid
syst = 1 + 1e-3*np.exp(-t/0.4) + 3e-4*(t-4)/8 + 6e-5*x
ecl = 1 + 150e-6*((t < 3.4) | (t > 4.5))              # 150 ppm eclipse in the middle
y = ecl*syst + rng.normal(0, 250e-6, t.size)

def fit(cols):
    M = np.column_stack(cols); c, *_ = np.linalg.lstsq(M, y, rcond=None); return y - M@c
shape = ((t < 3.4) | (t > 4.5)).astype(float)
for name, cols in [("no ramp", [np.ones_like(t), t, shape]),
                   ("ramp, no x", [np.ones_like(t), t, np.exp(-t/0.4), shape]),
                   ("ramp + x", [np.ones_like(t), t, np.exp(-t/0.4), x, shape])]:
    M = np.column_stack(cols); c, *_ = np.linalg.lstsq(M, y, rcond=None)
    print(f"{name:11s} depth ~ {1e6*c[-1]/c[0]:6.0f} ppm   residual rms {1e6*np.std(y-M@c):.0f} ppm")
```
Without the ramp column the depth comes out ~200 ppm instead of ~140: the wrong model doesn't
just fit worse, it **changes the depth**. That's why systematics are the
actual research problem, not a detail.

## 4. Is what's left white noise? The binned-RMS (Allan) test

If residuals are white, averaging n points shrinks their scatter by √n. If correlated ("red")
noise remains, the curve flattens and your error bars are too small.

```python
def binned_rms(r, sizes=(1, 2, 4, 8, 16, 32, 64, 128)):
    out = []
    for s in sizes:
        nb = len(r)//s
        out.append(np.std(r[:nb*s].reshape(nb, s).mean(axis=1), ddof=1))
    return np.array(sizes), np.array(out)

M = np.column_stack([np.ones_like(t), t, np.exp(-t/0.4), x, shape]); c, *_ = np.linalg.lstsq(M, y, rcond=None)
sizes, rms = binned_rms(y - M@c)
print("measured / white-expected at each bin size:", np.round(rms/(rms[0]/np.sqrt(sizes)), 2))
```
Ratios ≈ 1 = white. The project's β factor is this ratio at the eclipse timescale; β > 1.5 is a warning.

## 5. Choosing among models without fooling yourself
- Write a **menu** of physically motivated models (`linear+xy`, `exp+xy`, `exp+xyw`, ...).
- Score each with BIC; take the **simplest within ΔBIC 2** of the best.
- But keep terms you *know* are physically there (x, y), even if BIC says drop them: BIC
  judges one realisation and can't see that the depth's scatter tripled (the project's centroid
  lesson; learncode/05).
- Tune knobs (aperture size, model) with statistics that **can't see the eclipse** (e.g.
  high-frequency scatter), so tuning can't bias the depth.
- Prove it with **injection-recovery**: plant fake eclipses and check they come back unbiased, with
  scatter equal to the quoted error.

## 6. How this looks in the literature
TRAPPIST-1 b and c (Greene et al. 2023; Zieba et al. 2023): MIRI 15 µm eclipses, analysed with
exactly this toolbox (ramp + linear trend + decorrelation, MCMC), several independent pipelines
compared. Rocky Worlds' own data challenge was about the same thing: do different pipelines agree
on the same data?

## Readings
1. Greene et al. (2023), *Thermal emission from the Earth-sized exoplanet TRAPPIST-1 b using JWST*, Nature 618, 39 (arXiv:2303.14849). Short; read the methods.
2. Pont, Zucker & Queloz (2006), *The effect of red noise on planetary transit detection*, MNRAS 373, 231 (arXiv:astro-ph/0608597). Why white-noise error bars lie.
3. Bell et al. (2022), *Eureka!: An End-to-End Pipeline for JWST Time-Series Observations*, JOSS 7, 4503 (arXiv:2207.03585). The community pipeline RockyEye cross-checks against.

## Self-check (answers at the bottom)
1. What's the difference between a group, an integration and a segment?
2. Why does the exponential ramp column use raw time since the visit started, not centred time?
3. Your residuals' binned-RMS curve flattens at bin sizes near the eclipse duration. What does that mean for your error bar?
4. BIC says drop the x, y columns (ΔBIC = 3). You know the telescope jitters. What do you do, and how would you check you're right?
5. Why choose the aperture by high-frequency scatter rather than by "cleanest-looking eclipse"?

---
---

### Answers
1. Group: one non-destructive read. Integration: one ramp of groups → one image. Segment: a file holding hundreds of integrations.
2. So the ramp starts at exactly 1 at the beginning of the visit and decays from there; its
   timescale τ then has a direct physical meaning (settling time).
3. Correlated noise remains on the eclipse timescale, so the formal error is too small. Inflate
   it (β) or improve the systematics model.
4. Keep x, y (physically expected). Check with injection-recovery over many noise realisations:
   compare the scatter of recovered depths with and without x, y against the quoted error.
5. Choosing by the eclipse uses the answer to tune the method; noise that happens to deepen or
   shallow the eclipse would be selected. High-frequency scatter is blind to the eclipse.
