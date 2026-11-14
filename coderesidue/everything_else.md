# Everything else — one line each, so nothing in the folder is a mystery

Not worth deep study. Know what each file is for.

## Top level (`~/rockyproject/`)
| File | What it is |
|---|---|
| `SPEC.md` | The original build instructions given to Fable. |
| `HANDOFF.md` | Where things stand and the exact commands to resume. Read this first after any break. |
| `PROGRESS.md` | Your checklist: build phases A–D, then learning tiers. |
| `README.md` | What each top-level folder is for. |
| `.gitignore` | Keeps raw JWST data, FITS files, caches and `__pycache__` out of git. |

## `projrocky/` docs and setup
| File | What it is |
|---|---|
| `README.md` | Map of projrocky: layout, how to run, where results land. |
| `REPORT.md` | The paper-style write-up. Methods filled in; results (§5.2 onward), conclusion and abstract are placeholders until the real run. |
| `docs/HOW_IT_WORKS.md` | Code walkthrough with real excerpts. The learncode notes sit on top of it. |
| `environment.yml` | The conda env `rockyeye`: Python 3.12, jwst 3.0.0, batman, emcee, corner, kagglehub, photutils (tests only), and so on. |
| `pytest.ini` | Test settings: the `slow` marker (MCMC) and the `data` marker (needs real products). |
| `Makefile` | Shortcuts: `test`, `test-all`, `dry-run`, `import`, `sim`, `validate`, `real`, `truth`, `notebooks`. Limits numerical libraries to 4 threads. |

## `projrocky/rockyeye/` extras
| File | What it is |
|---|---|
| `__init__.py` | Empty; makes `rockyeye` an importable package. |
| `refdata/JWST_MIRI.F1500W.dat` | The real F1500W filter throughput (wavelength in Ångström, response), from the SVO filter service. Used by 09 for band averages; your lean version uses a top-hat instead. |
| `synth.py` | See `synth.md` in this folder. |

## `projrocky/configs/`
| File | What it is |
|---|---|
| `synthetic.yaml`, `lhs1140b_sim.yaml`, `gj3929b.yaml` | Covered in `learncode/12_configs.md`. |
| `_probe.yaml`, `gj3929b_preview.yaml` | Moved to `_to_delete/` on 2026-11-14 (a debug config and a superseded preview run). |

## `projrocky/scripts/` (thin command-line wrappers around the modules)
| File | What it does |
|---|---|
| `01_fetch.py` | Lists raw files and sizes; downloads only with `--download`. |
| `02_calibrate.py` | Streams segments through jwst stage 1–2 locally (now replaced by the Kaggle notebooks). |
| `03_photometry.py` | Stamps → one light curve per visit. |
| `04_systematics.py` | Compares systematics models per visit with BIC and picks one. |
| `05_fit.py` | Joint emcee fit over all visits, then each visit alone; resumable. |
| `06_injection.py` | Injection-recovery grid: bias and sensitivity. |
| `07_physics.py` | Depth → brightness temperature → bare rock vs redistribution. |
| `08_report.py` | Gathers everything into `stage08/summary.json` and `tables.md`. |
| `validate.py` | **The gate.** Pixel-level injection (0/80/160 ppm) on the simulated frames; must pass before real data unlocks. Logic in learncode/08. |
| `revalidate_truth.py` | After Nov 18: compares the sim posterior with STScI's official injected depth (`make truth`). Maths tested in `test_truth_slot.py`. |
| `eureka_crosscheck.py` | Runs the independent Eureka! pipeline on the same frames and compares photometry (agreed to 59 ppm RMS). Needs its own conda env, `rockyeye-eureka`. |
| `make_synthetic.py` | Calls `synth.py` to write the fake known-answer stamps. |
| `import_stamps.py` | Brings Kaggle outputs in: copies stamps, checks every segment is present and that the stage 1–2 settings match. |
| `build_kaggle_notebook.py` | Generates the Kaggle notebooks *from the real modules*, so the notebook code can't drift from projrocky. |

## Notebooks
| File | What it is |
|---|---|
| `notebooks/kaggle_gj3929b.ipynb` | Stages 01–02 for the real data on Kaggle (25 segments, 32 GB attached). Saves a stamp per segment, resumes, stops at 9 h. |
| `notebooks/kaggle_lhs1140b_sim.ipynb` | Same for the simulation (106 segments, 199 GB). |
| `kaggle_kernels/` | Moved to `_to_delete/` on 2026-11-14: upload copies of the notebooks plus `kernel-metadata.json` (Kaggle CLI settings). The runs are finished. |

## Generated folders (not files you read)
`data/` manifests, benchmark files and the Eureka! cross-check output · `products/` every stage's
outputs, figures and tables · `logs/` run logs · `.pytest_cache/`, `__pycache__/` Python caches.
