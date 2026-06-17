"""Build the results table and the headline chart from results/metrics.json.

Framed around the study's finding: off-the-shelf faithfulness detectors *over-flag*
faithful financial answers — they label faithful and broken answers at nearly the same
rate, giving ~chance balanced accuracy and no numeric-vs-prose or subtle-vs-gross gap.
The headline figure makes that over-flagging visible (recall on broken vs false-positive
rate on faithful); a second figure shows no perturbation type clears the false-positive
baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Stable perturbation-type order for table columns and the x-axis.
_TYPES = ["number_swap", "date_shift", "unit_currency", "direction_flip", "entity_swap"]


def load_metrics(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _ok_detectors(metrics: dict) -> dict:
    return {n: m for n, m in metrics["detectors"].items() if m.get("status") == "ok"}


def build_table(metrics: dict) -> pd.DataFrame:
    """One row per scored detector: overall metrics, faithful FPR, per-type & magnitude recall."""
    rows = {}
    for name, m in _ok_detectors(metrics).items():
        o = m["overall"]
        row = {
            "precision": o["precision"],
            "recall": o["recall"],
            "f1": o["f1"],
            "balanced_accuracy": o["balanced_accuracy"],
            "faithful_fpr": m["faithful_fpr"],
        }
        for t in _TYPES:
            row[f"recall_{t}"] = m["by_type"].get(t, {}).get("recall")
        for mag in ("subtle", "gross"):
            row[f"recall_{mag}"] = m["by_magnitude"].get(mag, {}).get("recall")
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index").round(3)


def _write_markdown(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    cols = list(df.columns)
    lines = [
        "| detector | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for name, row in df.iterrows():
        vals = ["" if pd.isna(row[c]) else f"{row[c]:.3f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    if note:
        lines += ["", note]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_headline(metrics: dict, path: Path) -> None:
    """Recall-on-broken vs false-positive-rate-on-faithful, per detector."""
    dets = _ok_detectors(metrics)
    names = list(dets)
    recall = [dets[n]["overall"]["recall"] for n in names]
    fpr = [dets[n]["faithful_fpr"] for n in names]
    bal = [dets[n]["overall"]["balanced_accuracy"] for n in names]

    x = range(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], recall, w, label="Recall on broken (caught)", color="#4C72B0")
    b2 = ax.bar([i + w / 2 for i in x], fpr, w, label="False-positive rate on faithful (over-flag)", color="#C44E52")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9)

    ax.axhline(0.5, ls="--", lw=1, color="gray")
    ax.text(len(names) - 0.5, 0.51, "chance", color="gray", fontsize=8, ha="right")
    for i, b in zip(x, bal):
        ax.text(i, 1.04, f"balanced acc = {b:.2f}", ha="center", fontsize=9, color="#333")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate")
    ax.set_title("Off-the-shelf detectors flag faithful answers nearly as often as broken ones")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.005,
             "NumFaith / FinanceBench — high recall but near-equal false-positive rate ⇒ detectors barely discriminate.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_recall_by_type(metrics: dict, path: Path) -> None:
    """Per-type recall, with each detector's faithful false-positive rate as a baseline line."""
    dets = _ok_detectors(metrics)
    names = list(dets)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
    x = range(len(_TYPES))
    n = len(names)
    w = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, name in enumerate(names):
        vals = [dets[name]["by_type"].get(t, {}).get("recall", 0.0) for t in _TYPES]
        offs = [i + (j - (n - 1) / 2) * w for i in x]
        ax.bar(offs, vals, w, label=f"{name} recall", color=colors[j % len(colors)])
        ax.axhline(dets[name]["faithful_fpr"], ls="--", lw=1.2, color=colors[j % len(colors)],
                   label=f"{name} faithful FPR")

    ax.set_xticks(list(x))
    ax.set_xticklabels(_TYPES, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall (fraction of broken caught)")
    ax.set_title("No perturbation type is caught much above the false-positive baseline")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_report(config: dict) -> dict:
    """Generate the results table (CSV + MD) and figures from metrics.json."""
    results_dir = Path(config["paths"]["results_dir"])
    metrics = load_metrics(results_dir / "metrics.json")

    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = build_table(metrics)
    df.to_csv(tables_dir / "main_results.csv", index_label="detector")
    skipped = [n for n, m in metrics["detectors"].items() if m.get("status") != "ok"]
    note = f"_Skipped detectors: {', '.join(skipped)}._" if skipped else None
    _write_markdown(df, tables_dir / "main_results.md", note)

    plot_headline(metrics, figures_dir / "headline.png")
    plot_recall_by_type(metrics, figures_dir / "recall_by_type.png")

    return {
        "table_csv": str(tables_dir / "main_results.csv"),
        "table_md": str(tables_dir / "main_results.md"),
        "figures": [str(figures_dir / "headline.png"), str(figures_dir / "recall_by_type.png")],
    }
