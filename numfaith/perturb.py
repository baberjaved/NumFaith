"""The answer-breaking engine: controlled, auto-labelled perturbations of faithful answers.

Each perturbation obeys the same contract:

1. Find a target in the answer (a number, date, unit, direction word, or entity).
2. Pick a replacement that changes the meaning.
3. **Safety check** — the replacement must NOT appear in ``source_text``; otherwise the
   "broken" answer might still be supportable and the ``unfaithful`` label would be wrong.
4. Return a row labelled ``unfaithful`` recording ``perturbation_type`` and
   ``(original, replacement)``.

Grammaticality is preserved *by construction*: replacements are always in-kind
(number→number, year→year, direction word→antonym, …). All randomness flows through a
single seeded ``random.Random`` so the test set is reproducible.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import regex as re

# Relative change separating a "subtle" numeric edit from a "gross" one.
DEFAULT_SUBTLE_GROSS_THRESHOLD = 0.10

# A number: optional currency prefix, a digit core (with optional thousands commas and
# decimals), and an optional percent suffix. e.g. $1,577.00  5.1%  153.2  2018
_NUMBER_TOKEN = re.compile(
    r"(?P<prefix>[$€£]?)"
    r"(?P<core>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>%?)"
)


# --------------------------------------------------------------------------- helpers


def _in_source(value, source_text: str) -> bool:
    """True if ``value`` occurs in ``source_text`` (the safety gate).

    Case-insensitive. When ``value`` begins/ends with a digit, matches that abut another
    digit are ignored so e.g. ``5`` is not considered present inside ``1500``.
    """
    v = str(value).strip().lower()
    if not v:
        return True  # an empty replacement is never safe
    s = source_text.lower()
    start = s.find(v)
    while start != -1:
        end = start + len(v)
        before = s[start - 1] if start > 0 else ""
        after = s[end] if end < len(s) else ""
        left_ok = not (v[0].isdigit() and before.isdigit())
        right_ok = not (v[-1].isdigit() and after.isdigit())
        if left_ok and right_ok:
            return True
        start = s.find(v, start + 1)
    return False


def _make_row(
    trio: dict,
    new_answer: str,
    ptype: str,
    original: str,
    replacement: str,
    magnitude: Optional[str] = None,
) -> dict:
    """Build an ``unfaithful`` test-set row from a trio and a single edit."""
    return {
        "id": trio["id"],
        "source_text": trio["source_text"],
        "question": trio["question"],
        "answer": new_answer,
        "label": "unfaithful",
        "perturbation_type": ptype,
        "original": original,
        "replacement": replacement,
        "magnitude": magnitude,
        "source_dataset": trio.get("source_dataset"),
    }


def _reformat(value: float, prefix: str, core: str, suffix: str) -> str:
    """Render ``value`` like the matched token (same prefix/suffix, decimals, commas)."""
    decimals = len(core.split(".")[1]) if "." in core else 0
    has_commas = "," in core
    s = f"{value:.{decimals}f}" if decimals else str(int(round(value)))
    if has_commas:
        intpart, dot, frac = s.partition(".")
        neg = intpart.startswith("-")
        intpart = intpart.lstrip("-")
        s = ("-" if neg else "") + f"{int(intpart):,}" + (dot + frac if dot else "")
    return f"{prefix}{s}{suffix}"


def _apply(answer: str, match: re.Match, new_token: str) -> str:
    """Splice ``new_token`` in place of ``match`` within ``answer``."""
    return answer[: match.start()] + new_token + answer[match.end() :]


# ---------------------------------------------------------------------- perturbations


def perturb_number(
    trio: dict, rng: random.Random, threshold: float = DEFAULT_SUBTLE_GROSS_THRESHOLD
) -> list[dict]:
    """Swap a number for a different one, emitting a subtle and a gross variant.

    Both variants edit the *same* chosen number so they are directly comparable; the
    ``magnitude`` field records which bucket each falls in.
    """
    answer, source = trio["answer"], trio["source_text"]
    matches = list(_NUMBER_TOKEN.finditer(answer))
    if not matches:
        return []

    m = rng.choice(matches)
    prefix, core, suffix = m.group("prefix"), m.group("core"), m.group("suffix")
    v = float(core.replace(",", ""))
    decimals = len(core.split(".")[1]) if "." in core else 0
    token = m.group(0)

    def _subtle() -> list[float]:
        cands = [v * (1 + rng.choice([-1, 1]) * rng.uniform(0.03, threshold)) for _ in range(8)]
        if decimals == 0:  # small integers need an absolute bump to actually change
            cands += [v + d for d in (1, -1, 2, 3, -2)]
        return cands

    def _gross() -> list[float]:
        cands = []
        for _ in range(8):
            factor = rng.uniform(2.0, 10.0)
            cands.append(v * factor if rng.random() < 0.5 else v / factor)
        cands += [v * 10, v * 100]
        return cands

    rows: list[dict] = []
    for magnitude, candidates in (("subtle", _subtle()), ("gross", _gross())):
        for nv in candidates:
            new_token = _reformat(nv, prefix, core, suffix)
            if new_token == token or _in_source(new_token, source):
                continue
            rows.append(
                _make_row(trio, _apply(answer, m, new_token), "number_swap", token, new_token, magnitude)
            )
            break
    return rows


# Registry: maps a config perturbation name to its function. Extended as types are added.
PERTURBATIONS: dict[str, Callable[..., list[dict]]] = {
    "number_swap": perturb_number,
}
