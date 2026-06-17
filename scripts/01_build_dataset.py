#!/usr/bin/env python3
"""Build the NumFaith test set from the configured source dataset.

Usage:
    python scripts/01_build_dataset.py [--config config/default.yaml] [--skip-perturb]

Loads and normalises the source financial QA into faithful (source, question,
answer) trios, then runs the perturbation engine to assemble the full test set
(faithful originals + all labelled broken copies). Pass ``--skip-perturb`` to
only build the trios.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from numfaith.load import build_trios
from numfaith.perturb import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config (default: config/default.yaml)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip-perturb",
        action="store_true",
        help="Only build the faithful trios; do not run the perturbation engine.",
    )
    mode.add_argument(
        "--perturb-only",
        action="store_true",
        help="Skip building trios; perturb the existing trios.jsonl into the test set.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    trios = None
    if not args.perturb_only:
        trios = build_trios(config)
        print(f"source_dataset : {config['source_dataset']}")
        print(f"kept trios     : {len(trios)}")
        print(f"written to     : {config['paths']['trios']}")
        if args.skip_perturb:
            return

    rows = build_dataset(config, trios)
    counts = Counter(r["perturbation_type"] for r in rows)
    n_broken = sum(v for k, v in counts.items() if k != "none")
    print("\n-- test set --")
    print(f"faithful       : {counts['none']}")
    print(f"broken         : {n_broken}")
    for ptype in sorted(k for k in counts if k != "none"):
        print(f"  {ptype:14}: {counts[ptype]}")
    print(f"total rows     : {len(rows)}")
    print(f"written to     : {Path(config['paths']['testset'])}")


if __name__ == "__main__":
    main()
