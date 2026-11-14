# 12 — Configs (`projrocky/configs/*.yaml`) · Tier 2: understand every value

These files aren't boilerplate: they hold the **physics and statistics choices**. "Why that
prior?" is a question about `gj3929b.yaml`. Goal: explain every line of it.

| Config | What | Why it exists |
|---|---|---|
| `synthetic.yaml` | fake stamps with an eclipse of exactly 100 ppm, 6 visits | truth known; runs on the laptop |
| `lhs1140b_sim.yaml` | STScI's simulated LHS 1140 b, 9 eclipses | the validation gate; official truth after Nov 18 |
| `gj3929b.yaml` | real JWST data, 4 eclipses (eclipse 4 split in two) | the science; locked until the gate passes |

## `gj3929b.yaml`, section by section
- **`requires_validation`**: stage 03 refuses to run until the sim validation passed. The
  "prove it where the answer is known" rule, enforced in code.
- **`source`**: Kaggle dataset + manifest. `visit_from_obsnum`: eclipse 4 was split into two
  exposures (frame-count limit) with a 16 min gap, so it becomes two fit units sharing one depth.
- **`stage1`**: `chunk_ints: 50` bounds memory (a whole segment needed far more than 16 GB).
  `jump: rejection_threshold 6.0` (default 4 flags too many good reads on a bright star),
  `minimum_sigclip_groups: 1000000` (switches off a variant that compares integrations: it was
  the memory hog and would make results depend on chunk size).
- **`photometry`**: candidate radii 3–8 px, sky ring 16–30 px, 11×11 centroid window, 5σ
  outlier cut.
- **`system`**: orbit from Xue et al. 2025, star from Kemmer et al. 2022 (T_eff 3369 ± 51 K,
  R 0.315 R☉, d 15.830 pc). Comments record why Beard et al. 2022 was *not* used. **Always cite
  the table.**
- **`fit`**:
  - `trim_start: 500`: drop the first integrations while the detector settles hardest.
  - `bin_size: 4`: average 4 points (speed; the eclipse is much longer than 4 integrations).
  - `candidate_models` all include `+xy` (the centroid lesson, see 05).
  - `priors`: depth −500…1000 ppm (negative allowed, see 06); `dt_min` −120…60 (the eclipse is
    known to come ~54 min early from slight eccentricity); τ 0.02–8 h.
  - `mcmc`: 64 walkers, up to 60k steps, stop at 50τ.
- **`injection`**: depths 0–300 ppm, 200 trials each.
- **`physics`**: three stellar models (measured flux, blackbody, SPHINX literature) and the
  scenario list: bare rock with A = 0 and 0.1, partial and full redistribution, plus three
  literature atmosphere models (Connors+ 2026).
- **benchmarks** (end of file): published depths to compare against, including the team's own
  HLSP: 142.5 ± 18.6 ppm for all four eclipses.

## Exercise (no code)
Pick five values in `gj3929b.yaml` and write one sentence each in your NOTES on what would
happen if you changed them. Example: "`bin_size: 1` → 4× slower fits, same depth within noise."
