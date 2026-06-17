"""Run detectors over the test set, score, and aggregate results.

Positive class is ``unfaithful``: a detector "fires" (true positive) when it labels a
broken row unfaithful. The 126 faithful originals are the shared negatives, so we also
report the **false-positive rate on faithful rows** — without it, per-type catch rates
are misleading (a detector that flags everything has high recall but is useless).

Raw detector outputs are cached to ``results/raw/<detector>.jsonl`` keyed by the answer
content, so re-runs never recompute or re-pay for API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from numfaith.detectors import get_detectors

# Verdicts that count as a real prediction (others — skipped/error — are excluded).
_USABLE = {"faithful", "unfaithful"}


def _row_key(row: dict) -> str:
    """Stable cache key: id repeats across faithful+broken, so disambiguate by answer."""
    raw = f"{row['id']}\x00{row['answer']}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def run_detector(detector, rows: list[dict], cache_path: Path) -> dict:
    """Return ``{row_key: result}`` for every row, using/extending the on-disk cache."""
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                cache[rec["key"]] = rec["result"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            key = _row_key(row)
            if key in cache:
                continue
            result = detector.detect(row["source_text"], row["question"], row["answer"])
            cache[key] = result
            fh.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")

    return {_row_key(row): cache[_row_key(row)] for row in rows}


def _score(y_true: list[int], y_pred: list[int]) -> dict:
    """Precision/recall/F1/balanced-accuracy for the unfaithful (=1) positive class."""
    return {
        "n": len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def _pred(result: dict) -> int | None:
    """1 if the detector said unfaithful, 0 if faithful, None if skipped/errored."""
    v = result.get("verdict")
    return None if v not in _USABLE else int(v == "unfaithful")


def _evaluate_detector(results: dict, rows: list[dict]) -> dict:
    """Build overall / per-type / per-magnitude metrics for one detector."""
    # Drop rows the detector couldn't score (e.g. all-skipped without an API key).
    scored = [(row, _pred(results[_row_key(row)])) for row in rows]
    scored = [(row, p) for row, p in scored if p is not None]
    if not scored:
        sample = results[_row_key(rows[0])]
        return {"status": sample.get("verdict", "skipped"), "reason": sample.get("info")}

    faithful = [(r, p) for r, p in scored if r["label"] == "faithful"]
    broken = [(r, p) for r, p in scored if r["label"] == "unfaithful"]
    yt = [int(r["label"] == "unfaithful") for r, _ in scored]
    yp = [p for _, p in scored]

    out = {
        "status": "ok",
        "overall": _score(yt, yp),
        "faithful_fpr": (sum(p for _, p in faithful) / len(faithful)) if faithful else None,
        "by_type": {},
        "by_magnitude": {},
    }

    types = sorted({r["perturbation_type"] for r, _ in broken})
    for t in types:
        slice_rows = faithful + [(r, p) for r, p in broken if r["perturbation_type"] == t]
        yt_t = [int(r["label"] == "unfaithful") for r, _ in slice_rows]
        yp_t = [p for _, p in slice_rows]
        out["by_type"][t] = _score(yt_t, yp_t)

    for mag in ("subtle", "gross"):
        mag_rows = faithful + [
            (r, p) for r, p in broken
            if r["perturbation_type"] == "number_swap" and r.get("magnitude") == mag
        ]
        if any(r["label"] == "unfaithful" for r, _ in mag_rows):
            yt_m = [int(r["label"] == "unfaithful") for r, _ in mag_rows]
            yp_m = [p for _, p in mag_rows]
            out["by_magnitude"][mag] = _score(yt_m, yp_m)

    return out


def evaluate(config: dict, rows: list[dict] | None = None) -> dict:
    """Run every enabled detector over the test set and write ``results/metrics.json``."""
    if rows is None:
        with open(config["paths"]["testset"], encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]

    results_dir = Path(config["paths"]["results_dir"])
    raw_dir = results_dir / "raw"

    metrics = {
        "n_faithful": sum(r["label"] == "faithful" for r in rows),
        "n_broken": sum(r["label"] == "unfaithful" for r in rows),
        "detectors": {},
    }
    for detector in get_detectors(config):
        results = run_detector(detector, rows, raw_dir / f"{detector.name}.jsonl")
        metrics["detectors"][detector.name] = _evaluate_detector(results, rows)

    out_path = results_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
