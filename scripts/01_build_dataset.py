#!/usr/bin/env python3
"""Build the faithful trios from the configured source dataset.

Usage:
    python scripts/01_build_dataset.py [--config config/default.yaml]

Loads and normalises the source financial QA into faithful (source, question,
answer) trios and writes them to the configured ``trios`` path. Phase 4 will
extend this to also run the perturbation orchestrator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from numfaith.load import build_trios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config (default: config/default.yaml)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    trios = build_trios(config)

    out_path = Path(config["paths"]["trios"])
    print(f"source_dataset : {config['source_dataset']}")
    print(f"kept trios     : {len(trios)}")
    print(f"written to     : {out_path}")


if __name__ == "__main__":
    main()
