# synth.py — the fake data with a known answer (`projrocky/rockyeye/synth.py`, 104 lines)

Tier 2: understand. Run by `scripts/make_synthetic.py` with `configs/synthetic.yaml`.

## Why it exists
The whole project rests on one rule: **prove the method where the answer is known before trusting
it where it isn't.** synth.py makes data where *you* chose the answer: an eclipse of exactly
100 ppm at a known time. If the pipeline gets 100 back (it got 117 ± 15, +1.1σ), photometry,
systematics, fitting, injection and physics are all verified together, on the laptop, with no
JWST download.

It is **not** a detector simulator. It skips stages 01–02 and writes the same stamp files stage 02
would produce (81×81 px cut-outs), so stages 03–10 can't tell the difference.

## What goes into each fake frame
`make_visit()` builds one visit, layer by layer:

| Layer | Code | Mimics |
|---|---|---|
| eclipse | `1 + depth * eclipse_shape(t, t_ecl, orbit)` | the planet, with a known depth and time |
| eclipse position | not centred, different per visit | real visits aren't perfectly centred either |
| settling ramp | `1 − amp·exp(−th/τ)`, amp and τ randomised per visit | detector warming up |
| drift | small random slope | slow trends |
| pointing | white jitter + slow random walk on x, y | telescope jitter and drift |
| pixel sensitivity | flux × (1 + s·x) | flux following position: the source of the centroid lesson |
| PSF | Airy pattern (`airy_psf`), shifted by jitter to first order | JWST's diffraction-limited star image |
| sky | constant + a small decaying term | background |
| noise | Gaussian with variance ∝ signal | photon noise |
| cosmic rays, bad pixels | random spikes, dead pixels (NaN) | detector junk the photometry must repair |

Noise and ramp sizes were matched to a real calibrated segment of the LHS 1140 b simulation
(~650 ppm per 13.5 s integration, a ramp of a few thousand ppm), so the test is realistic rather
than easy mode.

## The parts worth reading closely
```python
psf0 = airy_psf(size, half + 0.21, half + 0.39)       # star off pixel-centre on purpose
gx, gy = np.gradient(psf0, axis=1), np.gradient(psf0, axis=0)
for i in range(n):
    psf = psf0 - xy[i, 0] * gx - xy[i, 1] * gy       # first-order shift: fine for milli-pixel jitter
    sci[i] = star[i] * psf + sky[i]
```
A Taylor expansion instead of re-rendering the PSF every frame: fast, and accurate because the
jitter is tiny.

```python
sens = 1.0 + syn["x_sensitivity"] * (xy[:, 0] / syn["jitter_px"])
```
This one line is why the first synthetic run came out 3σ too deep: flux follows x, and a model
without an x term pushes that into the depth. It found a real flaw in the model selection
(learncode/05, "the big lesson").

`write_stamps()` splits each visit into the same segment files the manifest expects, writes
atomically, records the config hash, and returns the truth (eclipse time, ramp amplitude and
timescale per visit) so the report can compare against it.

## Can you explain it?
1. Why not just test on the simulated LHS 1140 b data instead?
   <details><summary>answer</summary>That needs 200 GB and Kaggle, and its truth isn't public
   until after Nov 18. Synthetic data is instant, local, and the truth is exact. It's the first
   rung; the STScI simulation is the second, real data the third.</details>
2. What *can't* synthetic data validate?
   <details><summary>answer</summary>Stages 01–02 (calibration), and any detector effect you didn't
   think to put in. That's what the STScI simulation is for: noise made by someone else.</details>
3. Why randomise the ramp and eclipse position per visit?
   <details><summary>answer</summary>So the pipeline can't get lucky with one fixed configuration;
   it must handle the variety real visits have.</details>
