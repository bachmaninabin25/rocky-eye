# learncode — notes on the real code, one ingredient at a time

`projrocky/docs/HOW_IT_WORKS.md` walks through the code. These notes are the
**learning layer** on top of it: what you must understand, what to rebuild,
what an interviewer will ask, and the lessons from the bugs found while building.

## Order

| # | Note | Tier | Time |
|---|---|---|---|
| — | `../learntopic/` days 1–3 | do first | 3 days |
| 04 | `04_eclipse.md` | **1 rebuild** | 2–3 days |
| 05 | `05_systematics.md` | **1 rebuild** | 2–3 days |
| 06 | `06_fit.md` | **1 rebuild** | 5–6 days (the hard one) |
| 07 | `07_diagnostics.md` | 2 exercise | 1 day |
| 08 | `08_injection.md` | 2 exercise | 1 day |
| 09 | `09_physics.md` | **1 rebuild** | 2–3 days |
| 03 | `03_photometry.md` | 2 exercise | 1 day |
| 12 | `12_configs.md` | 2 understand | half a day |
| 01 02 10 + | `tier3_plumbing.md`: fetch, calibrate, compare, pipeline, common, scripts | 3 skim | 2 days total |

## How to use a note
1. Read the note, with the matching HOW_IT_WORKS section open beside it.
2. Open the real file in `projrocky/rockyeye/` and read it top to bottom once.
3. Answer the **"Can you explain it?"** questions out loud, without looking. Answers are in a fold-out under each one.
4. Tier 1/2: go to `../codecode/` and rebuild.

## The whole chain in one breath
Raw detector reads → (01, 02) calibrated images, cut to small *stamps* around the star → (03)
brightness per frame = light curve, plus where the star sat and how wide it was → (05) a model
of the detector's drift built from those → (04) the shape of an eclipse → (06) fit both
together; MCMC gives depth ± error → (07) prove the chain converged → (08) prove we'd find a fake
eclipse → (09) depth → temperature → bare rock or atmosphere → (10) pick models fairly, compare
with published results.

## The rule behind every design choice
**Prove the method where the answer is known before trusting it where it isn't.** Synthetic data
(you made the eclipse), then STScI's simulation (they made it), then the real planet.
