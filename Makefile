# NumFaith — one-command reproduction pipeline.
#
# Quick start (from a fresh clone):
#   python -m venv .venv && source .venv/bin/activate
#   make install
#   make all
#
# Use an explicit interpreter without activating a venv:
#   make all PYTHON=.venv/bin/python
#
# Hardware: CPU is enough. A single GPU (e.g. T4) only speeds up the transformer
#   detectors; nothing here requires one.
# Runtime: data / perturb / report take seconds. `make eval` is ~12-13 min on the
#   first run (two local models incl. one-time Hugging Face downloads), then seconds
#   on subsequent runs (raw detector outputs are cached under results/raw/).
#   The LLM-judge detector is skipped unless OPENAI_API_KEY is set.

PYTHON ?= python

.PHONY: help install data perturb eval report all test clean

help:
	@echo "NumFaith targets:"
	@echo "  install  - install dependencies and the numfaith package (editable)"
	@echo "  data     - load + normalise source QA into faithful trios"
	@echo "  perturb  - break trios into the labelled test set"
	@echo "  eval     - run detectors over the test set and score (cached)"
	@echo "  report   - build the results table and figures"
	@echo "  all      - data + perturb + eval + report"
	@echo "  test     - run the unit tests"
	@echo "  clean    - remove generated data and results (keeps .gitkeep)"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) scripts/01_build_dataset.py --skip-perturb

perturb:
	$(PYTHON) scripts/01_build_dataset.py --perturb-only

eval:
	$(PYTHON) scripts/02_run_detectors.py

report:
	$(PYTHON) scripts/03_make_report.py

all: data perturb eval report

test:
	$(PYTHON) -m pytest -q

clean:
	rm -f data/processed/*.jsonl
	rm -rf results/raw
	rm -f results/metrics.json
	rm -f results/tables/main_results.csv results/tables/main_results.md
	rm -f results/figures/headline.png results/figures/recall_by_type.png
