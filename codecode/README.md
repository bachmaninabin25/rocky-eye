# codecode — where you rebuild RockyEye yourself

`projrocky/` is the reference. You never edit it. This folder holds **your**
version of the science, one ingredient at a time, with tests that tell you when
each piece is right.

## Setup (once)
```bash
cd ~/rockyproject/codecode
conda activate rockyeye
python run.py status          # everything says "todo": that's correct
```

## The loop, per ingredient
1. Read `../learncode/NN_<name>.md`.
2. Open the skeleton (e.g. `04_eclipse/eclipse.py`). Signatures and docstrings are given; every `TODO` body is yours.
3. `python run.py check 04` until it's green. Stuck for 20 min? Open `hints.md`, one tier at a time.
4. `python run.py demo --mine 04` runs the whole mini-pipeline with **your** piece plugged into the reference.
5. Commit: `git commit -m "04 eclipse: rebuilt"`. Your history is the proof you built it.

## What's in here

| Tier | Ingredient | Your file | Size | Check |
|---|---|---|---|---|
| 1 rebuild | 04 eclipse model | `04_eclipse/eclipse.py` | ~60 lines | `run.py check 04` |
| 1 rebuild | 05 systematics | `05_systematics/systematics.py` | ~80 | `run.py check 05` |
| 1 rebuild | 06 likelihood + MCMC | `06_fit/fit.py` | ~150 | `run.py check 06` (MCMC ~2 min; `--fast` skips it) |
| 1 rebuild | 09 physics verdict | `09_physics/physics.py` | ~100 | `run.py check 09` |
| 2 exercise | 03 photometry: centroid & width | `tier2/03_photometry/exercise.py` | ~15 | `run.py check 03` |
| 2 exercise | 07 diagnostics: split R-hat | `tier2/07_diagnostics/exercise.py` | ~10 | `run.py check 07` |
| 2 exercise | 08 injection | `tier2/08_injection/exercise.py` | ~20 | `run.py check 08` |

Tier 3 (01 fetch, 02 calibrate, 10 compare, pipeline, common) has notes only.

**Suggested order:** 04 → 05 → 06 → 09, with the Tier 2 exercises slotted in
where they fit (07 right after 06, 08 after 07, 03 any time).

## How the swapping works
`ROCKY_MINE` picks whose code runs: `run.py check 06` sets `ROCKY_MINE=06`, so
your `fit.py` runs on top of the **reference** 04 and 05. A bug in your 04
can't break your 06 test. `run.py demo --mine all` is the finish line: the
whole chain on your code only.

## The lean version differs from projrocky in a few deliberate ways
- 04: geometry only, no `batman` (the tests compare you *to* batman).
- 05: no `poly3` baseline.
- 06: no exposure breaks, no per-visit timing, no resumable HDF5 chain, fixed MCMC length instead of auto-stopping on convergence.
- 09: top-hat band instead of the real F1500W throughput curve.

That's the ~15% of the code that is the science. The rest of projrocky is
engineering around it (see `learncode/` Tier 3 notes).

## Files you don't touch
`_deps.py`, `_reference/`, `conftest.py`, `tier2/_load.py`, and `test_*.py`.
The tests are your spec. Read them before you start each ingredient.
