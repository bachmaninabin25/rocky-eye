# RockyEye - every stage is cached and resumable; rerunning a target only does missing work.
# Stages 01-02 (raw ramps -> stamps) run on Kaggle (notebooks/); the laptop starts at stage 03.
PY ?= $(HOME)/miniforge3/envs/rockyeye/bin/python
SIM = configs/lhs1140b_sim.yaml
REAL = configs/gj3929b.yaml
SYN = configs/synthetic.yaml
# Use at most half of the CPU cores for numerical libraries.
export OMP_NUM_THREADS ?= 4
export VECLIB_MAXIMUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4

.PHONY: help test test-all dry-run notebooks import synthetic sim validate real truth all

help:
	@echo "make test        fast unit tests (no data needed)"
	@echo "make test-all    also the short MCMC test"
	@echo "make dry-run     list both datasets and their sizes; downloads nothing"
	@echo "make notebooks   regenerate the two Kaggle notebooks from the current code"
	@echo "make import FROM=<zip or folder>   bring Kaggle stamps into products/"
	@echo "make synthetic   end-to-end check on synthetic stamps with a known eclipse"
	@echo "make sim         stages 03-08 on the simulated LHS 1140 b stamps"
	@echo "make validate    pixel-level injection gate on the simulated stamps"
	@echo "make real        stages 03-08 on GJ 3929 b (refuses until 'validate' has passed)"
	@echo "make truth       compare the simulated-data result with the official truth (config slot)"

test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

dry-run:
	$(PY) scripts/01_fetch.py --config $(SIM)
	$(PY) scripts/01_fetch.py --config $(REAL)

notebooks:
	$(PY) scripts/build_kaggle_notebook.py

import:
	@test -n "$(FROM)" || (echo "usage: make import FROM=~/Downloads/<kaggle output zip or folder>"; exit 1)
	$(PY) scripts/import_stamps.py --from "$(FROM)"

# stages 03-07 for one config: $(call run_stages,<config>)
STAGES = 03_photometry 04_systematics 05_fit 06_injection 07_physics
run_stages = for s in $(STAGES); do $(PY) scripts/$$s.py --config $(1) || exit 1; done

synthetic:
	$(PY) scripts/make_synthetic.py --config $(SYN)
	$(call run_stages,$(SYN))
	$(PY) scripts/validate.py --config $(SYN)
	$(PY) scripts/08_report.py --config $(SYN)
	$(PY) scripts/revalidate_truth.py --config $(SYN)

sim:
	$(call run_stages,$(SIM))
	$(PY) scripts/08_report.py --config $(SIM)

validate:
	$(PY) scripts/validate.py --config $(SIM)
	$(PY) scripts/08_report.py --config $(SIM)

real:
	$(call run_stages,$(REAL))
	$(PY) scripts/08_report.py --config $(REAL)

truth:
	$(PY) scripts/revalidate_truth.py --config $(SIM)

all: sim validate real
