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

import json
import random
from pathlib import Path
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


_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_QUARTER = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_MONTHS_FULL = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_RE = re.compile(
    r"\b(?:" + "|".join(_MONTHS_FULL + ["Sept"] + _MONTHS_ABBR) + r")\b", re.IGNORECASE
)


def _recase(word: str, like: str) -> str:
    """Return ``word`` cased like ``like`` (UPPER / Title / lower)."""
    if like.isupper():
        return word.upper()
    if like[:1].isupper():
        return word.capitalize()
    return word.lower()


def _month_candidates(token: str, rng: random.Random) -> list[str]:
    low = token.lower().rstrip(".")
    is_abbr = len(low) <= 4
    pool = _MONTHS_ABBR if is_abbr else _MONTHS_FULL
    idx = next(
        (i for i, (a, f) in enumerate(zip(_MONTHS_ABBR, _MONTHS_FULL))
         if low == f.lower() or low.startswith(a.lower())),
        None,
    )
    others = [i for i in range(12) if i != idx]
    rng.shuffle(others)
    return [_recase(pool[i], token) for i in others]


def _date_candidates(kind: str, token: str, m: re.Match, rng: random.Random) -> list[str]:
    if kind == "year":
        y = int(token)
        deltas = [-1, 1, -2, 2, -3, 3]
        rng.shuffle(deltas)
        return [str(y + d) for d in deltas if 1900 <= y + d <= 2099]
    if kind == "quarter":
        q = int(m.group(1))
        others = [i for i in (1, 2, 3, 4) if i != q]
        rng.shuffle(others)
        return [token[0] + str(i) for i in others]  # preserve Q/q case
    return _month_candidates(token, rng)


def perturb_date(trio: dict, rng: random.Random) -> list[dict]:
    """Shift a year, quarter, or month to a different in-kind value."""
    answer, source = trio["answer"], trio["source_text"]
    targets = (
        [("year", m) for m in _YEAR.finditer(answer)]
        + [("quarter", m) for m in _QUARTER.finditer(answer)]
        + [("month", m) for m in _MONTH_RE.finditer(answer)]
    )
    if not targets:
        return []
    rng.shuffle(targets)
    for kind, m in targets:
        token = m.group(0)
        for new_token in _date_candidates(kind, token, m, rng):
            if new_token == token or _in_source(new_token, source):
                continue
            return [_make_row(trio, _apply(answer, m, new_token), "date_shift", token, new_token)]
    return []


_SCALE_RE = re.compile(r"\b(?:thousand|million|billion|trillion)s?\b", re.IGNORECASE)
_SCALE_SWAP = {
    "thousand": "million", "million": "billion", "billion": "million", "trillion": "billion",
}
_CURRENCY_WORD_RE = re.compile(r"\b(?:dollars?|euros?|pounds?|USD|EUR|GBP)\b", re.IGNORECASE)
_CURRENCY_WORD_SWAP = {
    "dollar": "euro", "euro": "pound", "pound": "dollar",
    "usd": "eur", "eur": "gbp", "gbp": "usd",
}
_CURRENCY_SYM_RE = re.compile(r"[$€£]")
_CURRENCY_SYM_SWAP = {"$": "€", "€": "£", "£": "$"}


def _swap_word(token: str, mapping: dict) -> str:
    """Map a (possibly plural, any-case) word via ``mapping``, preserving plural + case."""
    low = token.lower()
    plural = low.endswith("s") and low[:-1] in mapping
    base = low[:-1] if plural else low
    repl = mapping[base] + ("s" if plural else "")
    return _recase(repl, token)


def perturb_unit_currency(trio: dict, rng: random.Random) -> list[dict]:
    """Swap a scale word (million↔billion) or currency (symbol or word)."""
    answer, source = trio["answer"], trio["source_text"]
    edits: list[tuple[re.Match, str]] = []
    for m in _SCALE_RE.finditer(answer):
        edits.append((m, _swap_word(m.group(0), _SCALE_SWAP)))
    for m in _CURRENCY_WORD_RE.finditer(answer):
        edits.append((m, _swap_word(m.group(0), _CURRENCY_WORD_SWAP)))
    for m in _CURRENCY_SYM_RE.finditer(answer):
        edits.append((m, _CURRENCY_SYM_SWAP[m.group(0)]))
    if not edits:
        return []
    rng.shuffle(edits)
    for m, new_token in edits:
        token = m.group(0)
        if new_token == token or _in_source(new_token, source):
            continue
        return [_make_row(trio, _apply(answer, m, new_token), "unit_currency", token, new_token)]
    return []


_DIRECTION_PAIRS = [
    ("rose", "fell"), ("rise", "fall"), ("increased", "decreased"), ("increase", "decrease"),
    ("up", "down"), ("gain", "loss"), ("gains", "losses"), ("gained", "lost"),
    ("grew", "shrank"), ("growth", "decline"), ("higher", "lower"), ("improved", "worsened"),
    ("expanded", "contracted"), ("positive", "negative"), ("above", "below"),
    ("strengthened", "weakened"), ("outperformed", "underperformed"),
]
_DIRECTION_SWAP: dict[str, str] = {}
for _a, _b in _DIRECTION_PAIRS:
    _DIRECTION_SWAP[_a] = _b
    _DIRECTION_SWAP[_b] = _a
_DIRECTION_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_DIRECTION_SWAP, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def perturb_direction(trio: dict, rng: random.Random) -> list[dict]:
    """Flip a direction word to its antonym (rose↔fell, increased↔decreased, …)."""
    answer, source = trio["answer"], trio["source_text"]
    matches = list(_DIRECTION_RE.finditer(answer))
    if not matches:
        return []
    rng.shuffle(matches)
    for m in matches:
        token = m.group(0)
        repl = _recase(_DIRECTION_SWAP[token.lower()], token)
        if repl == token or _in_source(repl, source):
            continue
        return [_make_row(trio, _apply(answer, m, repl), "direction_flip", token, repl)]
    return []


_METRICS = [
    "free cash flow", "operating cash flow", "capital expenditure", "gross margin",
    "operating margin", "operating income", "net income", "net sales", "gross profit",
    "total assets", "total liabilities", "total debt", "cash flow", "revenue", "EBITDA",
]
_COMPANIES = [
    "3M", "Apple", "Amazon", "Microsoft", "Pfizer", "Johnson & Johnson", "PepsiCo",
    "Coca-Cola", "CVS Health", "Amcor", "Costco", "Verizon", "Nike", "Oracle", "Walmart",
    "Adobe", "Netflix", "Boeing", "Lockheed Martin", "General Mills", "Kraft Heinz",
]


def _phrase_re(pool: list[str]) -> re.Pattern:
    alts = "|".join(re.escape(p) for p in sorted(pool, key=len, reverse=True))
    return re.compile(r"\b(?:" + alts + r")\b", re.IGNORECASE)


_METRIC_RE = _phrase_re(_METRICS)
_COMPANY_RE = _phrase_re(_COMPANIES)


def _recase_phrase(alt: str, like: str) -> str:
    if alt.isupper():  # acronym (e.g. EBITDA) — leave as-is
        return alt
    if like[:1].isupper():
        return alt[:1].upper() + alt[1:]
    return alt


def perturb_entity(trio: dict, rng: random.Random) -> list[dict]:
    """Swap a financial metric or company name for a different one absent from source."""
    answer, source = trio["answer"], trio["source_text"]
    for regex, pool, preserve_case in (
        (_METRIC_RE, _METRICS, True),
        (_COMPANY_RE, _COMPANIES, False),
    ):
        matches = list(regex.finditer(answer))
        rng.shuffle(matches)
        for m in matches:
            token = m.group(0)
            alts = [x for x in pool if x.lower() != token.lower()]
            rng.shuffle(alts)
            for alt in alts:
                repl = _recase_phrase(alt, token) if preserve_case else alt
                if _in_source(repl, source):
                    continue
                return [_make_row(trio, _apply(answer, m, repl), "entity_swap", token, repl)]
    return []


# Registry: maps a config perturbation name to its function.
PERTURBATIONS: dict[str, Callable[..., list[dict]]] = {
    "number_swap": perturb_number,
    "date_shift": perturb_date,
    "unit_currency": perturb_unit_currency,
    "direction_flip": perturb_direction,
    "entity_swap": perturb_entity,
}


# ------------------------------------------------------------------------- orchestrator


def _faithful_row(trio: dict) -> dict:
    """The unperturbed original, carried into the test set as a ``faithful`` negative."""
    return {
        "id": trio["id"],
        "source_text": trio["source_text"],
        "question": trio["question"],
        "answer": trio["answer"],
        "label": "faithful",
        "perturbation_type": "none",
        "original": None,
        "replacement": None,
        "magnitude": None,
        "source_dataset": trio.get("source_dataset"),
    }


def build_dataset(config: dict, trios: Optional[list[dict]] = None) -> list[dict]:
    """Run all enabled perturbations over every trio and write the full test set.

    Emits each faithful original plus every labelled broken variant to
    ``config['paths']['testset']`` and returns the rows.
    """
    if trios is None:
        with open(config["paths"]["trios"], encoding="utf-8") as fh:
            trios = [json.loads(line) for line in fh]

    pcfg = config.get("perturbations", {})
    rng = random.Random(pcfg.get("seed", 42))
    enabled = pcfg.get("enabled", list(PERTURBATIONS))
    threshold = pcfg.get("subtle_gross_threshold", DEFAULT_SUBTLE_GROSS_THRESHOLD)

    rows: list[dict] = []
    for trio in trios:
        rows.append(_faithful_row(trio))
        for name in enabled:
            fn = PERTURBATIONS.get(name)
            if fn is None:
                continue
            if name == "number_swap":
                rows.extend(fn(trio, rng, threshold))
            else:
                rows.extend(fn(trio, rng))

    out_path = Path(config["paths"]["testset"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
