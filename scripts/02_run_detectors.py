#!/usr/bin/env python3
"""Run the enabled detectors over the NumFaith test set and score them.

Usage:
    python scripts/02_run_detectors.py [--config config/default.yaml]

Runs every detector in ``detectors.enabled`` over ``numfaith_testset.jsonl`` (caching
raw outputs under ``results/raw/`` so re-runs never recompute or re-pay), scores each
detector overall, by perturbation type, and by subtle-vs-gross, and writes
``results/metrics.json``.
"""

from __future__ import annotations

import argparse

import yaml

from numfaith.evaluate import evaluate


def _fmt(metric: dict, *keys: str) -> str:
    return "  ".join(f"{k}={metric[k]:.3f}" for k in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    metrics = evaluate(config)

    print(f"test set: {metrics['n_faithful']} faithful + {metrics['n_broken']} broken\n")
    for name, m in metrics["detectors"].items():
        if m.get("status") != "ok":
            print(f"== {name} ==  [{m.get('status')}] {m.get('reason') or ''}\n")
            continue
        print(f"== {name} ==")
        print(f"  overall   {_fmt(m['overall'], 'precision', 'recall', 'f1', 'balanced_accuracy')}")
        print(f"  faithful false-positive rate: {m['faithful_fpr']:.3f}")
        print("  recall by perturbation type:")
        for t, tm in sorted(m["by_type"].items()):
            print(f"    {t:14} recall={tm['recall']:.3f}  bal_acc={tm['balanced_accuracy']:.3f}  (n_broken={tm['n'] - metrics['n_faithful']})")
        if m["by_magnitude"]:
            mags = "  ".join(f"{k}={v['recall']:.3f}" for k, v in m["by_magnitude"].items())
            print(f"  number_swap recall by magnitude: {mags}")
        print()

    print("written to results/metrics.json")


if __name__ == "__main__":
    main()
