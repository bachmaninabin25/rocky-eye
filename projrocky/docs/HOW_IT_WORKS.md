# How RockyEye works

Reference notes on the whole system, from first principles, with the important code.
Code excerpts are from the real files, sometimes trimmed (marked `...`).

## 0. The idea in one paragraph

A rocky planet on a tight orbit glows in the infrared. When it passes *behind* its star (a
secondary eclipse) that glow disappears for an hour or two, and the total light drops by about
0.01%. The size of the drop tells us how hot the planet's dayside is. A bare rock cannot move
heat to its night side, so its dayside is as hot as physics allows; an atmosphere carries heat
away and the dayside is cooler, so the eclipse is shallower. The whole project is one long chain
that turns raw detector reads into that one number and then into a verdict. The philosophy:
**prove the chain on data where the answer is known before trusting it on data where it is not.**

```
 Kaggle (230 GB raw, attached)                         laptop (stamps only, ~2 GB)
 ─────────────────────────────                         ───────────────────────────
 *_uncal.fits ─s01─▶ list/size ─s02─▶ jwst stage 1-2 ─▶ *_stamp.npz ──import_stamps──┐
                                      (50-int chunks)   81x81 px per integration     │
                                                                                     ▼
   products/<config>/stage02/ ─s03─▶ light curve ─s05─▶ systematics models ─s06─▶ emcee
        ▲                            (aperture sum)     (ramp, drift, x, y,       │  chains
        │                                               width, sky)   s04 eclipse │  (HDF5)
   synth.py  (known eclipse)                                           shape ─────┘
        │                                                                 │
        └── validate.py / s08: plant fake eclipses, check they come back  ▼
                                                              s07 convergence checks
                                                                          │
                              s09: depth ─▶ brightness temperature ─▶ bare rock vs atmosphere
                                                                          │
                              s10: model comparison, benchmark vs published ─▶ REPORT.md
```

One module per ingredient, `rockyeye/sNN_*.py`. `pipeline.py` strings them into stages; the
`scripts/NN_*.py` files are thin command-line wrappers; every setting lives in `configs/*.yaml`.

| Config | What it is | Why |
|---|---|---|
| `synthetic.yaml` | fake stamps, eclipse of exactly 100 ppm | truth known today, runs on the laptop |
| `lhs1140b_sim.yaml` | STScI's simulated LHS 1140 b (9 eclipses) | the validation gate; official truth public after Nov 18, 2026 |
| `gj3929b.yaml` | real JWST data, 4 eclipses | the science target; locked until the gate passes |

---

## 1. Fetching data: `rockyeye/s01_fetch.py`

### First principles
A JWST observation arrives as *segments*: files of about 2 GB, each holding a few hundred
integrations. Before doing anything you need a list of them with sizes, because the answer to
"does this fit on my disk?" decides the whole design (it does not: 199 GB vs ~30 GB free).

### A file is a record, not a download
Everything downstream works with this small object. Listing never touches the big files.

```python
@dataclass
class RemoteFile:
    """One raw file we might download."""

    visit: str          # e.g. "eclipse3"
    segment: int        # 1-based segment number within the visit
    filename: str
    url: str
    size_bytes: int
    sha256: str | None  # None when the archive gives no checksum
    local_path: str
```

### Dry run first, always
`plan()` adds up sizes and compares them with the free disk. `scripts/01_fetch.py` prints this
and stops unless you pass `--download`.

```python
def plan(cfg: dict, visits: list[str] | None = None) -> dict:
    """Work out what a download would involve, without downloading anything."""
    files = list_files(cfg, visits)
    todo = [f for f in files if not Path(f.local_path).exists()]
    ...
    return {
        "n_files": len(files),
        "total_gb": sum(f.size_gb for f in files),
        "missing_gb": sum(f.size_gb for f in todo),
        "largest_file_gb": max((f.size_gb for f in files), default=0.0),
        "free_gb": free_gb(data_dir(cfg)),
        ...
    }
```

### Three guards around every download
Refuse if the disk would fill, resume instead of restarting, verify the checksum.

```python
    if free_gb(dest.parent) - need_gb < min_free_gb:      # guard 1: never fill the disk
        raise OSError(f"refusing to download {rf.filename}: needs {need_gb:.2f} GB and would "
                      f"leave less than {min_free_gb} GB free ({free_gb(dest.parent):.1f} GB free now)")
    ...
    for attempt in range(1, max_retries + 1):
        have = part.stat().st_size if part.exists() else 0   # guard 2: continue the .part file
        if have >= rf.size_bytes:
            break
        try:
            if rf.url.startswith("kaggle://"):
                _fetch_kaggle(rf, part)        # Kaggle's official client (no login for public data)
            else:
                _fetch_http(rf, part, have)    # HTTPS with a Range header
        except Exception as e:   # network errors of any flavour: wait and resume
            wait = min(300, 10 * 2 ** (attempt - 1))      # 10 s, 20 s, 40 s ... capped at 5 min
            ...
    if verify and rf.sha256:                               # guard 3: the bytes are the right bytes
        digest = sha256_of(part)
        if digest != rf.sha256:
            part.unlink()
            raise OSError(f"{rf.filename}: SHA-256 mismatch (file deleted, rerun to refetch)")
    os.replace(part, dest)                                 # only a complete, verified file gets the real name
```

### The same code runs on Kaggle without downloading
On Kaggle the dataset is attached as a read-only folder. One config key (`source.local_root`)
points `local_path` at it; `download()` then returns at once because the file "already exists",
and `is_input_file()` stops the pipeline from ever deleting it.

---

## 2. Calibration: `rockyeye/s02_calibrate.py`

### First principles
MIRI does not take pictures. During one *integration* it reads every pixel 45 times without
resetting ("up the ramp"). The slope of that ramp is the count rate. Stage 1 of the official
`jwst` pipeline fits those slopes (after correcting detector effects); stage 2 flat-fields them.
We call the official code and only choose its settings in the config.

| Product | Shape (simulated data) | Meaning |
|---|---|---|
| `uncal` | 271 integrations × 45 groups × 256 × 256 | raw reads |
| `rateints` | 271 × 256 × 256 | slope per integration (stage 1) |
| `calints` | 271 × 256 × 256 | flat-fielded, flux-calibrated (stage 2) |
| stamp | 271 × 81 × 81 | our cut-out around the star: all that later stages need |

### Memory is the real limit, so calibrate in chunks
Stage 1 holds several float copies of the ramp cube. A whole segment needed far more than 16 GB
and the laptop started swapping. Fifty integrations at a time need about 4 GB. A chunk is written
exactly like an official segment file.

```python
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            path = work / uncal.name.replace("_mirimage_uncal.fits", f"-c{k:02d}_mirimage_uncal.fits")
            hdus = [fits.PrimaryHDU(header=h[0].header.copy())]
            hdus[0].header["INTSTART"] = first + a          # which integrations of the exposure these are
            hdus[0].header["INTEND"] = first + b - 1
            for ext in h[1:]:
                if ext.name in ("SCI", "REFOUT"):
                    # .section reads just these integrations and applies the unsigned-16-bit
                    # offset (BZERO), so the chunk is written as proper uint16 again
                    hdu = fits.ImageHDU(data=ext.section[a:b], header=ext.header.copy(), name=ext.name)
                ...
```

That `.section` comment records a real bug: copying the raw bytes without the offset made every
count negative, and the saturation step then flagged 99.99% of pixels as bad.
`tests/test_02_calibrate.py::test_split_uncal_is_lossless` now guards it.

### One segment in, one small stamp out, big files deleted
The loop that makes the data fit on any disk: finished work is renamed into place atomically and
its config hash recorded, so a rerun skips it.

```python
    for chunk in chunks:
        ...
            rateints = run_stage1(chunk, work, cfg.get("stage1", {}).get("steps", {}))
        ...
            calints = run_stage2(rateints, work, cfg.get("stage2", {}).get("steps", {}))
        ...
        part = make_stamp(calints, centre, st["half_size"], guess=st.get("guess"))
        ...
        if chunk != uncal:
            chunk.unlink(missing_ok=True)                 # delete the chunk ...
        for product, kind in ((rateints, "rateints"), (calints, "calints")):
            if keep.get(kind, False):
                ...
            else:
                product.unlink(missing_ok=True)           # ... and the intermediate products
    ...
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **stamp, filename=rf.filename, visit=rf.visit, segment=rf.segment)
    os.replace(tmp, out)                                  # atomic: never a half-written stamp
    mark_cached(out, key)                                 # remember which settings made it
```

### Times are matched by integration number, never by position
A segment file can carry the time table of the whole exposure. Taking "the first N rows" gave
segment 2 an empty time axis. The fix:

```python
    if i0 is not None and i1 is not None:
        t = t_all[(number >= i0) & (number <= i1)]        # INTSTART..INTEND from the header
        if len(t) == n:
            return t
    if len(t_all) == n:
        return t_all
    raise ValueError(...)                                 # fail loudly rather than misalign time
```

### Why the jump step has two unusual settings
`configs/*.yaml`: `jump: {rejection_threshold: 6.0, minimum_sigclip_groups: 1000000}`. Six sigma
because the default four flags too many good reads on a bright star. The second switches off a
variant that compares integrations with each other: it was the memory hog, and it would make
results depend on the chunk size.

---

## 3. Photometry: `rockyeye/s03_photometry.py`

### First principles
"How bright is the star in this frame?" = add up the pixels inside a circle around it, after
subtracting the sky. Do that for every integration and you have a light curve. The eclipse is
100 parts per million, so anything that changes the sum by even 0.01% matters.

### Exact circle-pixel overlap
A pixel on the edge of the circle is partly inside. Counting it as all-in or all-out makes the
flux jump when the star moves by a thousandth of a pixel. So each pixel gets the exact fraction
of its area inside the circle, from one area function and inclusion-exclusion over four corners.

```python
def aperture_weights(shape: tuple[int, int], x0: float, y0: float, r: float) -> np.ndarray:
    ...
    xl = np.arange(nx) - 0.5 - x0                 # left edge of every pixel, relative to the centre
    yl = np.arange(ny) - 0.5 - y0
    X0, Y0 = np.meshgrid(xl, yl)
    X1, Y1 = X0 + 1.0, Y0 + 1.0
    return (_signed_area(X1, Y1, r) - _signed_area(X0, Y1, r)      # area(rectangle) =
            - _signed_area(X1, Y0, r) + _signed_area(X0, Y0, r))   # A(x1,y1) - A(x0,y1) - A(x1,y0) + A(x0,y0)
```

Test: the weights sum to πr² to 12 digits and match `photutils` to 1e-9.

### Repair before you add
Cosmic rays and bad pixels are replaced by that pixel's own recent history (a running median in
time). A pixel that deviates by more than 5 robust sigma from its own median is an outlier.

```python
        smooth = median_filter(np.nan_to_num(filled), size=(window, 1, 1), mode="nearest")  # per pixel, along time
        resid = filled - smooth
        mad = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0)     # robust sigma
    mad = np.where(mad > 0, mad, np.inf)
    outlier = np.abs(resid) > sigma * mad[None]
    replace = outlier | ~np.isfinite(cube)
    out = np.where(replace, smooth, cube)
```

### Choosing the aperture without peeking at the answer
Too small a circle loses light when the star jitters; too large adds sky noise. We try a grid of
radii and keep the one with the lowest *high-frequency* scatter. The running-median window
(101 points) is shorter than an eclipse, so this number cannot see the eclipse depth, and choosing
by it cannot bias the depth.

```python
def highpass_scatter(flux: np.ndarray, window: int = 101):
    f = flux / np.nanmedian(flux)
    resid = f - median_filter(f, size=window, mode="nearest")
    sig = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    return float(sig), np.abs(resid) < 5 * sig
```

The light curve also keeps what the systematics models need: centroid `x`, `y`, PSF widths `sx`,
`sy` and sky level `bg` for every integration.

---

## 4. The eclipse model: `rockyeye/s04_eclipse.py`

### First principles
The eclipse is pure geometry: how much of the planet's disk is hidden behind the star at time t.
The convention used everywhere:

```
flux(t) / F_star = 1 + depth × shape(t)        shape = 1 planet visible, 0 planet hidden
```

So `depth` is the planet-to-star flux ratio, and the *shape* does not depend on the depth.

### batman does the geometry, with fp = 1
```python
    p.t_secondary = t_ecl
    p.per = orbit.period
    p.rp = orbit.rp_rs
    p.a = orbit.a_rs
    p.inc = orbit.inc
    ...
    p.fp = 1.0                                   # planet/star flux ratio set to 1 ...
    m = batman.TransitModel(p, np.ascontiguousarray(t, dtype=float), transittype="secondary")
    return m.light_curve(p) - 1.0                # ... so batman's "1 + visible fraction" minus 1 is the shape
```

### Never trust one implementation
`eclipse_shape_analytic` computes the same thing from the overlap area of two circles and
nothing else. The test demands agreement to 1e-6, and that the eclipse lasts exactly as long as
the textbook contact-time formula says (`Orbit.duration_full`).

| Quantity | GJ 3929 b | LHS 1140 b |
|---|---|---|
| Period | 2.616 d | 24.737 d |
| Eclipse duration (first to last contact) | 1.19 h | 2.08 h |
| Expected bare-rock depth at 15 µm | ~140 ppm | ~75 ppm |

---

## 5. Systematics: `rockyeye/s05_systematics.py`

### First principles
The detector is not a perfect ruler. For the first hour it "settles" (a ramp of thousands of ppm,
30× the eclipse). The flux drifts slowly. It follows the star's position on the pixels, the PSF
width and the sky. A systematics model describes those so they can be told apart from the eclipse.

### Every model is a sum of basis vectors
```
S(t) = c0·1 + c1·time + c2·exp(-t/τ) + c3·x(t) + c4·y(t) + ...
```
The coefficients `c` enter linearly. Only τ is non-linear. This split is the design decision the
whole fit rests on (section 6).

```python
        tc = (th - th.mean()) / max(np.ptp(th), 1e-12)   # centred, unit-range time
        cols = [np.ones_like(th)]                         # baseline level
        if self.baseline != "const":
            cols.append(tc)                               # slow drift
        ...
        if self.baseline in ("exp", "exp2"):
            cols.append(np.exp(-th / np.exp(theta[0])))   # settling ramp; theta[0] = log(tau in hours)
        ...
        for c in self.aux_columns:
            cols.append(aux[c])                           # decorrelation: x, y, sx, sy, bg
        return np.column_stack(cols)
```

A model is named by a string: `exp+xy` = linear baseline + exponential ramp + centroid x and y.
Candidates compared on every visit: `linear+xy, poly2+xy, exp+xy, exp+xyw, exp+xywbg, exp2+xy`.

### Centroid terms are the floor, not an option (a lesson from the synthetic data)
The first synthetic run came out 3σ too deep. Switching ingredients off one at a time traced it to
pointing drift: the star wanders by a few milli-pixels, the flux follows it, and a model without
`x, y` absorbs that drift into the depth. Over 10 noise realisations the depth was *not biased on
average* (107 ± 24 ppm for a 100 ppm truth), but its scatter was 77 ppm while the quoted error
stayed ~35 ppm. BIC alone did not catch it (it preferred dropping `x, y` by 3-5 points), and
neither did β. With `x, y` always in: scatter 24 ppm, and nothing lost when there is no drift
(23 vs 23 ppm). The rule that transfers: **if a nuisance term is physically expected, keep it;
let the statistics choose only among the extras.**

### Is the noise white? The binned-RMS test
Average the residuals in bins of n points. White noise shrinks as 1/√n. If the curve flattens,
correlated ("red") noise remains and error bars are too small. `red_noise_beta` is the ratio of
measured to expected RMS at the eclipse timescale: β ≈ 1 is good, β > 1.5 is a warning.

---

## 6. Likelihood and MCMC: `rockyeye/s06_fit.py`

### First principles
The model for one visit is `flux = [1 + depth·shape] × [B·c] + noise`. A *likelihood* says how
probable the data are for given parameters; with Gaussian noise it is exp(−χ²/2σ²) times a
normalisation. *MCMC* draws parameter sets in proportion to likelihood × prior; the spread of the
draws is the error bar.

### Integrate out what you can, sample only what you must
Nine visits with seven coefficients each would be 60+ dimensions, too many for `emcee`. But the
coefficients are linear, and a Gaussian integral over linear parameters has a closed form:

```
ln L = −(N−K)/2 · ln(2πσ²) − χ²_min/(2σ²) − ½ ln det(MᵀM)
```

`χ²_min` is the least-squares residual; `M` is the design matrix with the eclipse multiplied in.
So the sampler explores only depth, eclipse time, and per visit a noise level and a ramp timescale.

```python
    def log_like(self, p: np.ndarray) -> float:
        total = 0.0
        for i, v in enumerate(self.visits):
            M = self.design(p, i)                         # eclipse-modulated basis for this visit
            try:
                _, chi2, logdet = solve_linear(v.flux, M) # least squares: the coefficients are solved, not sampled
            except np.linalg.LinAlgError:
                return -np.inf
            log_sigma = p[self._slices[i]][0]
            dof = v.n - M.shape[1]
            total += (-dof * (log_sigma + 0.5 * LOG2PI)
                      - 0.5 * chi2 * np.exp(-2 * log_sigma) - 0.5 * logdet)
        return total if np.isfinite(total) else -np.inf
```

`test_marginal_likelihood_matches_brute_force` integrates a coefficient numerically and gets the
same number. A joint fit is the same function: likelihoods of the visits add, the depth is shared.

### The eclipse shape is a lookup table
With fixed geometry the shape depends only on time from mid-eclipse. batman is called once on a
1-second grid; afterwards it is `np.interp`. A likelihood call takes 0.2 ms.

### Depth is allowed to be negative
Prior: uniform on [−500, 1000] ppm. Forbidding negative depths would push noise-level
measurements upward. Timing prior: a wide window around phase 0.5, because a slightly eccentric
orbit moves the eclipse (GJ 3929 b's is about 54 minutes early).

### Safe to kill and rerun
```python
        backend = emcee.backends.HDFBackend(str(backend_path))     # every step goes to disk
        ...
            fresh = backend.iteration == 0 or backend.shape != (n_walkers, self.ndim)
        ...
        else:
            start = None                                            # None = continue from the stored state
            log.info("resuming chain %s at step %d", backend_path.name, backend.iteration)
```

It stops by itself only when the chain passes the same tests the report applies: longer than 50
autocorrelation times, that estimate stable (within 10% between checks), and split R̂ ≤ 1.01.
`max_steps` is only a safety cap. (An earlier version stopped on length alone; the slow test then
caught chains that were "long enough" but whose R̂ was still 1.015.)

### Single visits borrow the eclipse time from the joint fit
One visit alone cannot pin down *when* a 100 ppm eclipse happened. `pipeline.stage_fit` runs the
joint fit first and gives the single-visit fits a timing prior of ±3σ around the joint value.

---

## 7. Convergence diagnostics: `rockyeye/s07_diagnostics.py`

### First principles
A chain can look fine and be wrong: too short, or stuck. Nothing proves convergence, so we use
several independent warning lights and require all of them off.

| Check | Threshold | Why that number |
|---|---|---|
| chain length / τ | ≥ 50 | below ~50 the estimate of τ itself is unreliable |
| split R̂ | ≤ 1.01 | sequences disagree by under 1% of the posterior width |
| effective samples | ≥ 1000 | 1σ quantiles good to a few percent |
| Geweke z (first 10% vs last 50%) | ≤ 3 | larger means it was still drifting |
| acceptance | 0.05 – 0.9, no walker < 0.02 | stuck or random-walking sampler |

τ, the *integrated autocorrelation time*, is how many steps make one independent sample.

### R̂: do the sequences agree with each other?
```python
    half = n_steps // 2
    seq = np.concatenate([chain[:half], chain[half:2 * half]], axis=1)  # each walker cut in two halves
    n = seq.shape[0]
    means = seq.mean(axis=0)
    within = seq.var(axis=0, ddof=1).mean(axis=0)      # scatter inside each sequence
    between = n * means.var(axis=0, ddof=1)            # scatter between sequence means
    var_plus = (n - 1) / n * within + between / n
    return np.sqrt(var_plus / np.where(within > 0, within, np.nan))   # -> 1 when all sample the same thing
```

The tests build deliberately bad chains (walkers at two different values, a chain 3τ long, a
chain still drifting, stuck walkers) and check that each is flagged.

---

## 8. Injection-recovery: `rockyeye/s08_injection.py`

### First principles
"Would we have seen an eclipse of depth D, and measured it right?" The only honest answer is to
plant one and look.

### Pixel level: dim the star in the frames
```python
    scale = (1.0 + depth * (shape - 1.0)).astype(np.float32)[:, None, None]   # 1 out of eclipse, 1-depth in eclipse
    sky = bg.astype(np.float32)[:, None, None]
    out["sci"] = sky + (stamps["sci"] - sky) * scale                          # only the star is scaled, not the sky
```
`pipeline.stage_validate` masks the real eclipse, plants eclipses of 0, 80 and 160 ppm in the
middle of the longest clean stretch beside it (a visit is only ~4 eclipse-durations long, so a fixed
offset would fall off the end of the data), reruns photometry and the MCMC, and passes only if
every injected depth (0 included) comes back within 2σ and the depth *scale* is preserved to 5%
((recovered − recovered at 0) / injected). Why 2σ and not 1σ: with honest error bars the truth
falls outside 1σ a third of the time, so a 1σ bar would fail a correct pipeline one run in three. This is the gate: `gj3929b.yaml` has
`requires_validation: configs/lhs1140b_sim.yaml` and stage 03 refuses to run until it passes.

### Light-curve level: hundreds of trials with real noise
New noise realisations come from the real residuals, cyclically shifted ("prayer bead"), which
keeps their correlations. Each trial uses a fast estimator: for small depths the eclipse is just
one more linear column.

```python
    s = fit.shape(v.t, t_ecl) - 1.0                 # 0 out of eclipse, -1 in eclipse
    M = np.column_stack([B, s])                     # systematics basis + eclipse column
    q, r = np.linalg.qr(M)
    c = np.linalg.solve(r, q.T @ flux)
    ...
    return 1e6 * c[-1] / base, 1e6 * np.sqrt(cov[-1, -1]) / base, chi2   # depth, formal error, chi2
```

Outputs: bias (recovered − injected), whether the scatter of recovered depths matches the quoted
error (`error_ratio` ≈ 1), and the fraction of trials detected at 3σ for each depth.

---

## 9. Physics: `rockyeye/s09_physics.py`

### First principles
```
depth = (Rp/Rs)² × B(T_planet) / I_star          (band-averaged over F1500W)
```
B is the Planck function. Solve for T_planet: the dayside *brightness temperature*. Energy
balance predicts what it should be (Cowan & Agol 2011):

```
T_day = T_star / √(a/Rs) × (1 − A)^¼ × (2/3 − 5/12·ε)^¼
```

| Case | A (albedo) | ε (redistribution) | GJ 3929 b |
|---|---|---|---|
| dark bare rock: hottest possible | 0 | 0 | 737 K |
| thick atmosphere, heat fully spread | 0 | 1 | 577 K |

```python
def dayside_temperature(teff, a_rs, albedo=0.0, eps=0.0):
    return np.asarray(teff) / np.sqrt(a_rs) * (1 - np.asarray(albedo)) ** 0.25 * (2 / 3 - 5 / 12 * np.asarray(eps)) ** 0.25
```

### The star matters as much as the planet
The depth is a ratio, so an error in the star's 15 µm brightness is an error in the planet's
temperature. Three options, all run: a blackbody at T_eff; a surface brightness from the star's
*measured* flux, radius and distance; a quoted model flux.

```python
    if mode == "flux":
        omega = (np.asarray(rstar_rsun) * RSUN / (np.asarray(dist_pc) * PC)) ** 2   # (R/d)^2
        return np.asarray(flux_mjy) * 1e-29 / (np.pi * omega)                        # F = pi * I * (R/d)^2
```

The measured flux comes from our own frames (`s03_photometry.absolute_flux_mjy`), using the
calibration programme's aperture and correction factor.

### The verdict is a ratio of probabilities
A scenario with no free parameters predicts a depth, with a width from the uncertain star and
orbit. Its *evidence* is the likelihood averaged over that prediction. The fit used a flat depth
prior, so a smoothed histogram (KDE) of the posterior samples *is* the likelihood.

```python
    like = depth_likelihood(depth_samples_ppm)             # KDE of the depth posterior
    ...
    for name, pred in predicted.items():
        z = float(np.mean(like(pred[:4000])))              # evidence = E_model[ L(depth) ]
        out[name] = {...
                     "tension_sigma": float((d_med - np.mean(pred)) / np.hypot(d_std, np.std(pred)))}
    z_ref = out[reference]["evidence"]
    for name in out:
        out[name]["bayes_factor_vs_ref"] = out[name]["evidence"] / z_ref if z_ref > 0 else np.nan
```

Bayes factor B vs "bare rock": |ln B| < 1 no real preference, 1–2.5 weak, 2.5–5 moderate, > 5
strong. Also reported: R = T_b / T_max (1 = bare rock), and a 95% upper limit on ε.

---

## 10. Model comparison and report: `rockyeye/s10_compare.py`, `pipeline.py`

### First principles
More parameters always fit better. BIC charges for them: `BIC = k·ln(N) − 2·ln L_max`. Lower is
better; differences under about 2 mean nothing.

### Ties go to the simpler model
```python
def choose_model(rows: list[dict], max_dbic: float = 2.0) -> dict:
    ok = [r for r in rows if r["dbic"] <= max_dbic]
    return min(ok, key=lambda r: (r["k"], r["bic"]))      # fewest parameters among the near-best
```

### Is there an eclipse at all?
`detection_bic` fits the same systematics with depth fixed to zero. ΔBIC > 10 = strong detection.
One trap, now tested: with no eclipse the timing parameter is meaningless, but it must still sit
inside its prior, or every trial point is rejected and the "no eclipse" model is scored at a
random ramp timescale (single-eclipse fits briefly reported ΔBIC ≈ 600 for 3σ signals).

### Never run the same thing twice
Each stage stores a hash of the config sections it depends on next to its output.

```python
def is_cached(path: Path, key: str) -> bool:
    """A product is fresh if it exists and its sidecar records the same config hash."""
    side = Path(str(path) + ".key")
    return path.exists() and side.exists() and side.read_text().strip() == key
```

Change the aperture list and photometry onward reruns; stamps do not. `stage_report` gathers
everything into `products/<config>/stage08/tables.md` and `summary.json`, which REPORT.md quotes.

---

## Safety nets

- **Real data are locked.** `pipeline.require_validation` blocks GJ 3929 b from stage 03 on
  until `products/lhs1140b_sim/validation/validation.json` says `passed: true`.
- **Raw data never land on the laptop.** Stages 01–02 run in the Kaggle notebooks, generated
  from the real modules by `scripts/build_kaggle_notebook.py` so the code cannot drift.
- **A long Kaggle run cannot be lost.** One stamp per segment, written as soon as it is done;
  failed segments are logged and skipped; the run stops itself after 9 h; a rerun with the
  previous output attached skips finished segments.
- **Imports are checked.** `scripts/import_stamps.py` counts segments against the manifest and
  compares the stage 1–2 config hash.
- **Atomic writes.** Stamps and JSON go to a temp name, then `os.replace`.
- **Testing without JWST.** 58 tests on synthetic data: no network, no reference files.
  `make test` (20 s), `make test-all` (adds a short MCMC).
- **Independent cross-checks.** Aperture weights vs photutils; eclipse shape vs circle geometry;
  marginal likelihood vs brute-force integration; photometry vs Eureka! on the same frames
  (`scripts/eureka_crosscheck.py`).
- **Truth slot.** `scripts/revalidate_truth.py` compares the simulated-data posterior with the
  official injected depth once it is public.

## Design principles worth remembering

- Prove the method where the answer is known before using it where it is not.
- Split linear from non-linear parameters; solve the linear ones exactly.
- Choose tuning knobs (aperture, model) with a statistic that cannot see the answer.
- Let the sampler go negative; a bounded prior biases weak signals.
- Every long job checkpoints, and "rerun" always means "continue".
- Keep the small product (stamps), delete the big ones; redoing cheap steps beats storing everything.
- One independent implementation of each critical formula, used only in tests.
- When an end-to-end check fails, switch ingredients off one at a time, then repeat over many
  noise realisations before deciding it is a bias: one realisation proves nothing.
- Config in YAML, hashed into the cache key: results always say which settings made them.
- A check that can fail loudly (raise) is worth more than one that warns.

## Glossary

| Term | Meaning |
|---|---|
| Secondary eclipse | planet passes behind the star; its light is blocked |
| Eclipse depth | planet flux / star flux, here in ppm (parts per million) |
| Integration / group | one exposure / one of its non-destructive reads |
| uncal, rateints, calints | raw reads / slope per integration / calibrated slope |
| Stamp | our 81×81 px cut-out per integration |
| Ramp (settling) | detector response drifting for the first hour of a visit |
| Decorrelation | removing flux changes that follow x, y, width or sky |
| Marginal likelihood | likelihood with nuisance parameters integrated out |
| τ (autocorrelation time) | steps per independent MCMC sample |
| R̂ | between-chain vs within-chain variance; 1 = agreement |
| BIC | fit quality with a penalty per parameter; lower is better |
| β (red-noise factor) | measured / white-noise-expected RMS at the eclipse timescale |
| Brightness temperature | temperature of a blackbody as bright as the planet in this band |
| ε, A | heat redistribution efficiency (0–1), Bond albedo |
| Bayes factor | ratio of the evidences of two scenarios |
| CRDS | STScI's server of calibration reference files |

## Try it yourself (5-minute exercises)

Run from `projrocky/` with the `rockyeye` environment active.

**1. See the eclipse shape and check its duration by hand.**
```python
import numpy as np
from rockyeye.s04_eclipse import Orbit, eclipse_shape
o = Orbit(period=2.6162644, t0=2460452.8997, a_rs=17.05, inc=89.3, rp_rs=0.0318)
te = o.predicted_eclipse_time(2460864.9)
t = te + np.linspace(-0.05, 0.05, 2001)
s = eclipse_shape(t, te, o)
print("hours with the planet (partly) hidden:", np.ptp(t[s < 1]) * 24, "formula:", o.duration_full() * 24)
```

**2. Redo the photometry of one synthetic visit with a different aperture.**
```python
from rockyeye.common import load_config
from rockyeye import s02_calibrate, s03_photometry
cfg = load_config("configs/synthetic.yaml")
stamps = s02_calibrate.load_visit_stamps(cfg, "eclipse1")
for r in (3.0, 5.0, 8.0):
    lc = s03_photometry.extract_light_curve(stamps, dict(cfg["photometry"], radius=r))
    print(r, "px ->", round(lc["scatter_best_ppm"]), "ppm per integration")
```

**3. Bare rock or atmosphere, by hand, for a depth you choose.**
```python
import numpy as np
from rockyeye import s09_physics as ph
band = ph.Band()
i_star = ph.stellar_intensity(band, "flux", flux_mjy=18.86, rstar_rsun=0.315, dist_pc=15.830)
for depth_ppm in (60, 118, 160):
    T = ph.temperature_from_depth(band, depth_ppm * 1e-6, 0.0318, i_star)
    print(depth_ppm, "ppm ->", round(float(T)), "K;  bare rock max:", round(float(ph.dayside_temperature(3369, 17.05))), "K")
```

**4. Break a chain and watch the diagnostics catch it.**
```python
import numpy as np
from rockyeye.s07_diagnostics import diagnose, format_report
rng = np.random.default_rng(0)
good = rng.normal(size=(4000, 16, 1))
bad = good + np.where(np.arange(16) < 8, 0, 2)[None, :, None]     # half the walkers sit somewhere else
print(format_report(diagnose(good, ["depth"])))
print(format_report(diagnose(bad, ["depth"])))
```
