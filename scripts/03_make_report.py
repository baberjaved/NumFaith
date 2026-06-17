#!/usr/bin/env python3
"""Build the results table and headline figures from results/metrics.json.

Usage:
    python scripts/03_make_report.py [--config config/default.yaml]

Reads the metrics produced by scripts/02_run_detectors.py and writes the main
detectors x perturbation-types table (CSV + Markdown) to results/tables/ and the
shareable figures to results/figures/.
"""

from __future__ import annotations

import argparse

import yaml

from numfaith.report import make_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    out = make_report(config)
    print("table (csv):", out["table_csv"])
    print("table (md) :", out["table_md"])
    for fig in out["figures"]:
        print("figure     :", fig)


if __name__ == "__main__":
    main()
