# Tier 3: plumbing (skim, know what it does, don't rebuild)

These are about 85% of the lines and 0% of the science. Know what each does and why it's
built that way; that's enough for an interview.

## 01 fetch (`s01_fetch.py`, 325 lines) · HOW_IT_WORKS §1
- A file is a **record** (`RemoteFile`: visit, segment, size, sha256, path) before it's a download.
- **Dry run first**: `plan()` adds up sizes vs free disk (199 GB sim vs ~30 GB free decided the
  whole architecture: stream one file at a time, then move to Kaggle).
- **Three guards per download**: refuse if the disk would drop below `min_free_gb`; resume a
  `.part` file with exponential back-off; verify SHA-256, and only then rename into place.
- On Kaggle the dataset is a read-only folder; `local_root` points at it, and `is_input_file()`
  stops the pipeline from ever deleting it.
- **Say in an interview:** "I sized the data before writing the pipeline, and the size drove
  the design."

## 02 calibrate (`s02_calibrate.py`, 331 lines) · HOW_IT_WORKS §2
- MIRI reads each pixel 45 times per integration ("up the ramp"). The official `jwst` stage 1
  fits ramp slopes; stage 2 flat-fields. We only choose its settings.
- **Chunks of 50 integrations** (memory), each written as a valid mini segment file.
- **One segment in, one ~small stamp out**, everything big deleted; atomic writes; config hash
  recorded so reruns skip finished work.
- **Real bugs:** copying raw bytes without the FITS unsigned offset (BZERO) made every count
  negative → 99.99% of pixels "saturated". Times taken by row position instead of integration
  number gave segment 2 an empty time axis. Both now have regression tests.

## 10 compare (`s10_compare.py`, 337 lines) · HOW_IT_WORKS §10
- **BIC** = k ln N − 2 ln L. `choose_model`: among models within ΔBIC 2 of the best, take the
  one with the fewest parameters (ties → simpler).
- `detection_bic`: same model with depth = 0; ΔBIC > 10 = strong detection.
- `tension`, `benchmark_rows`, markdown tables, all the plots (raw curves, fit, phase-folded,
  Allan, corner, trace, depths vs published, injection, physics).

## pipeline + common (`pipeline.py` 536, `common.py` 137)
- `pipeline.py` strings the ingredients into stages: photometry → systematics choice → fits
  (joint first; single visits then get a timing prior ±3σ around the joint value) → injection →
  physics → validate → report.
- `require_validation` enforces the gate. `stage_report` writes `tables.md` + `summary.json`.
- `common.py`: config loading (YAML), paths, **cache keys** (hash of the config sections a
  stage depends on, stored beside the output; a mismatch reruns the stage), atomic JSON
  writes, free-disk check, logging, timer.

## Scripts, notebooks, Makefile
`scripts/NN_*.py` are thin CLI wrappers. `build_kaggle_notebook.py` *generates* the Kaggle
notebooks from the real modules so they can't drift apart. `import_stamps.py` checks segment
counts and the stage 1–2 config hash. The Makefile is shortcuts; read it once.

## Design principles worth quoting (from HOW_IT_WORKS)
- Every long job checkpoints; "rerun" always means "continue".
- Keep the small product (stamps), delete the big ones.
- A check that raises is worth more than one that warns.
- Config in YAML, hashed into the cache key: results always say which settings made them.
