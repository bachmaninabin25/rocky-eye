# 03 — Aperture photometry (`rockyeye/s03_photometry.py`, 229 lines) · Tier 2: understand + exercise

HOW_IT_WORKS §3 · exercise: `codecode/tier2/03_photometry/` (centroid and width)

## What it does
Turns stamps (81×81 px cut-outs, one per integration) into a **light curve**: the star's
brightness in every frame, plus the decorrelation columns 05 needs: centroid x, y, PSF widths
sx, sy, sky level bg.

## Four steps per frame
1. **Repair** bad pixels and cosmic rays with each pixel's own running median *in time*
   (a pixel deviating > 5 robust σ from its own history is replaced).
2. **Sky** from a ring (annulus, 16–30 px) around the star, sigma-clipped.
3. **Centre and width** from flux-weighted moments in a fixed 11×11 window (your exercise).
   Fixed so the window can't hop between pixels and add noise of its own.
4. **Aperture sum** with **exact** circle–pixel overlap weights.

## The key ideas
- **Exact overlap.** A pixel on the circle's edge is partly inside. All-in/all-out would make the
  flux jump when the star moves a thousandth of a pixel; 100 ppm signals can't survive that.
  The weights come from an analytic area function and inclusion–exclusion over the pixel's
  corners; they sum to πr² to 12 digits and match `photutils` to 1e-9.
- **Choose the aperture without peeking.** Try radii 3–8 px; keep the one with the lowest
  *high-frequency* scatter (residual after a 101-point running median). That window is shorter
  than the eclipse, so the criterion can't see the depth and can't bias it.
- **Written from scratch.** photutils is used only in tests, as an independent check.

## Can you explain it?
1. Why not just choose the aperture that gives the cleanest eclipse?
   <details><summary>answer</summary>That uses the answer to tune the method: the chosen
   aperture would favour noise that happens to deepen (or shallow) the eclipse. Biased.</details>
2. Why are x, y and the widths saved at all?
   <details><summary>answer</summary>The measured flux follows them (pointing jitter, PSF
   breathing). They become decorrelation columns in 05.</details>
3. Why repair pixels with a median *in time* rather than from neighbouring pixels?
   <details><summary>answer</summary>The PSF is sharp: neighbours have very different values.
   The same pixel a few frames earlier is the best predictor.</details>

## Exercise
`centroid_and_width`: ~15 lines. `python run.py check 03`.
