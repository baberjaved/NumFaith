"""Load and normalise source financial QA into faithful (source, question, answer) trios.

The output of this module is a list of *faithful trios* — examples whose answer is
correct and grounded in ``source_text`` — written to ``data/processed/trios.jsonl``.
These are the clean inputs the perturbation engine (Phase 4) breaks.

Normalised schema (one JSON object per line):

    {"id", "source_text", "question", "answer", "source_dataset"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import regex as re

# Matches a number (incl. currency/percent/thousands separators) or a temporal
# token (4-digit year, quarter like "Q3", or an English month name). Used to keep
# only answers that contain something the perturbation engine can break.
_NUMBER_OR_DATE = re.compile(
    r"""
      \$?\d[\d,]*(?:\.\d+)?%?        # 12  1,234.5  $3.4  5.1%
    | \bQ[1-4]\b                      # Q1..Q4
    | \b(?:19|20)\d{2}\b              # 1999..2099
    | \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)
        (?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _has_number_or_date(text: str) -> bool:
    """True if ``text`` contains a number or date — i.e. it is breakable."""
    return bool(text) and _NUMBER_OR_DATE.search(text) is not None


def _join_evidence(evidence: Any) -> str:
    """Join FinanceBench ``evidence`` passages into a single grounding string.

    ``evidence`` is a list of dicts carrying an ``evidence_text`` field; we tolerate
    bare strings and alternate key names defensively in case the schema shifts.
    """
    if evidence is None:
        return ""
    if isinstance(evidence, str):
        return evidence.strip()

    passages: list[str] = []
    for item in evidence:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("evidence_text") or item.get("text") or ""
        else:
            text = ""
        text = (text or "").strip()
        if text:
            passages.append(text)
    return "\n\n".join(passages)


def load_financebench(hf_path: str, split: str) -> list[dict]:
    """Load FinanceBench from Hugging Face and map it to the normalised schema.

    ``source_text`` is built from the grounding ``evidence`` only. The ``justification``
    field is intentionally excluded: it explains how the answer was derived and can leak
    the answer, which would undermine the Phase 4 safety check.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_path, split=split)
    trios: list[dict] = []
    for row in ds:
        trios.append(
            {
                "id": str(row.get("financebench_id")),
                "source_text": _join_evidence(row.get("evidence")),
                "question": (row.get("question") or "").strip(),
                "answer": (row.get("answer") or "").strip(),
                "source_dataset": "financebench",
            }
        )
    return trios


# Registry mapping a config ``source_dataset`` name to its loader. Add new loaders
# here to support more sources without touching ``build_trios``.
LOADERS: dict[str, Callable[..., list[dict]]] = {
    "financebench": load_financebench,
}


def build_trios(config: dict) -> list[dict]:
    """Build faithful trios per ``config`` and write them to ``trios.jsonl``.

    Drops rows with empty grounding or non-breakable answers, applies the
    ``max_examples`` cap (deterministically, preserving source order), writes the
    result, and returns the kept rows.
    """
    name = config["source_dataset"]
    if name not in LOADERS:
        raise ValueError(
            f"Unknown source_dataset {name!r}; known: {sorted(LOADERS)}"
        )

    source_cfg = config.get("source", {}).get(name, {})
    raw = LOADERS[name](
        hf_path=source_cfg["hf_path"],
        split=source_cfg.get("hf_split", "train"),
    )

    kept = [
        t
        for t in raw
        if t["source_text"].strip()
        and t["answer"]
        and _has_number_or_date(t["answer"])
    ]

    max_examples = config.get("max_examples")
    if max_examples:
        kept = kept[:max_examples]

    out_path = Path(config["paths"]["trios"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for trio in kept:
            fh.write(json.dumps(trio, ensure_ascii=False) + "\n")

    return kept
