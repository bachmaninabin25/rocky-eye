# RockyEye

Independent reduction of the JWST/MIRI 15 µm eclipses of the rocky exoplanet **GJ 3929 b**
(Rocky Worlds DDT, program 9235). Question: **bare rock or atmosphere?**

**Result:** eclipse depth **126 ± 20 ppm**, dayside ~674 K. Thick atmospheres ruled out at ~4.5σ;
bare rock or a thin atmosphere fit. Agrees with the official team's 142 ± 19 ppm.
Details: `projrocky/REPORT.md`. Status and next steps: `HANDOFF.md`.

| Folder / file | What lives there |
|---|---|
| `projrocky/` | the full working pipeline, results and write-up (reference; I never edit it) |
| `learntopic/` | 3 days of background notes, independent of the code |
| `learncode/` | notes on the real code, ingredient by ingredient (start at `00_START_HERE.md`) |
| `codecode/` | skeletons + tests where I rebuild the science myself (`python run.py status`) |
| `coderesidue/` | notes on every remaining file: synth.py, the tests, scripts, setup |
| `HANDOFF.md` | final status, known issues, what's left |
| `PROGRESS.md` | my checklist: build phases A–D, then learning tiers |
| `SPEC.md` | the original build spec |
| `_to_delete/` | 1.7 GB of redundant files, safe to empty |
