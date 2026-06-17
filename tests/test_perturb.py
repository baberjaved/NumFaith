"""Unit tests for the perturbation engine (numfaith/perturb.py).

The auto-labelling is the part reviewers scrutinise most, so these tests lock down
the contract: each perturbation changes the answer, the safety check rejects
replacements present in the source, in-kind shape is preserved, labels/metadata are
correct, and everything is deterministic under a fixed seed.
"""

from __future__ import annotations

import random

import regex as re

from numfaith.perturb import (
    DEFAULT_SUBTLE_GROSS_THRESHOLD,
    PERTURBATIONS,
    _faithful_row,
    _in_source,
    build_dataset,
    perturb_date,
    perturb_direction,
    perturb_entity,
    perturb_number,
    perturb_unit_currency,
)

SEED = 42


def _trio(answer: str, source_text: str = "unrelated source text with no overlap") -> dict:
    return {
        "id": "t1",
        "source_text": source_text,
        "question": "q?",
        "answer": answer,
        "source_dataset": "test",
    }


def rng() -> random.Random:
    return random.Random(SEED)


# --------------------------------------------------------------- safety gate (_in_source)


def test_in_source_present_and_absent():
    assert _in_source("decreased", "revenue decreased sharply") is True
    assert _in_source("increased", "revenue decreased sharply") is False


def test_in_source_digit_boundary():
    # a bare "5" must NOT count as present inside a larger number
    assert _in_source("5", "capex was 1500 last year") is False
    # but a real standalone match does count
    assert _in_source("5", "there were 5 segments") is True


def test_in_source_percent_and_empty():
    assert _in_source("5%", "gross margin was 5% this year") is True
    assert _in_source("", "anything") is True  # empty replacement is never safe


# ------------------------------------------------------ each perturbation changes the answer


def test_number_swap_changes_answer():
    rows = perturb_number(_trio("Revenue was $1,577.00 in the period."), rng())
    assert rows
    assert all(r["answer"] != "Revenue was $1,577.00 in the period." for r in rows)


def test_date_shift_changes_answer():
    rows = perturb_date(_trio("Results improved in FY2018."), rng())
    assert rows and rows[0]["answer"] != "Results improved in FY2018."


def test_unit_currency_changes_answer():
    rows = perturb_unit_currency(_trio("Net income was $5.2 million."), rng())
    assert rows and rows[0]["answer"] != "Net income was $5.2 million."


def test_direction_flip_changes_answer():
    rows = perturb_direction(_trio("Revenue increased year over year."), rng())
    assert rows and rows[0]["answer"] != "Revenue increased year over year."


def test_entity_swap_changes_answer():
    rows = perturb_entity(_trio("Net income reached a record."), rng())
    assert rows and rows[0]["answer"] != "Net income reached a record."


def test_perturbations_with_no_target_emit_nothing():
    blank = _trio("Everything remained qualitatively stable.")
    assert perturb_number(blank, rng()) == []
    assert perturb_date(blank, rng()) == []
    # a bare number has no direction word or entity to swap
    assert perturb_direction(_trio("42"), rng()) == []
    assert perturb_entity(_trio("42"), rng()) == []


# ------------------------------------------------------------------ safety check at engine level


def test_direction_flip_blocked_when_antonym_in_source():
    # the only flip ("increased" -> "decreased") is unsafe because the source contains it
    trio = _trio("Sales increased.", source_text="the segment decreased during the year")
    assert perturb_direction(trio, rng()) == []


def test_no_emitted_replacement_appears_in_source():
    trios = [
        _trio("Revenue was $1,577.00 in FY2018.", "Filing for fiscal 2018; capex 1,577."),
        _trio("Margin rose 5.1% as net income grew.", "net income discussion only"),
        _trio("Net income was 3.4 billion dollars.", "reported in local currency"),
        _trio("Microsoft increased revenue.", "Apple and peers competed; revenue noted"),
    ]
    r = rng()
    for trio in trios:
        for fn in (perturb_number, perturb_date, perturb_unit_currency, perturb_direction, perturb_entity):
            for row in fn(trio, r):
                assert not _in_source(row["replacement"], row["source_text"]), row


# -------------------------------------------------------------------- shape / grammaticality


def test_number_swap_preserves_currency_and_decimals():
    rows = perturb_number(_trio("Capex was $1,577.00."), rng())
    for r in rows:
        assert re.fullmatch(r"\$[\d,]+\.\d{2}", r["replacement"]), r["replacement"]


def test_number_swap_preserves_percent_suffix():
    rows = perturb_number(_trio("Gross margin was 5.1% in the period."), rng())
    assert rows and all(r["replacement"].endswith("%") for r in rows)


def test_date_shift_year_stays_a_year():
    rows = perturb_date(_trio("Results improved in FY2018."), rng())
    assert rows
    assert re.search(r"FY(?:19|20)\d{2}", rows[0]["answer"])
    assert rows[0]["replacement"] != "2018"


def test_date_shift_quarter_stays_a_quarter():
    rows = perturb_date(_trio("Sales peaked in Q3."), rng())
    assert rows and rows[0]["replacement"] in {"Q1", "Q2", "Q4"}


def test_direction_flip_uses_antonym():
    rows = perturb_direction(_trio("Revenue increased."), rng())
    assert rows and rows[0]["replacement"] == "decreased"


def test_unit_currency_scale_swap():
    rows = perturb_unit_currency(_trio("It was 5 million units."), rng())
    assert rows and rows[0]["replacement"] == "billion"


# ------------------------------------------------------------------------ labels / metadata


def test_broken_row_metadata():
    rows = perturb_date(_trio("Closed in FY2018."), rng())
    r = rows[0]
    assert r["label"] == "unfaithful"
    assert r["perturbation_type"] == "date_shift"
    assert r["original"] and r["replacement"]
    assert r["magnitude"] is None  # only number_swap carries a magnitude


def test_number_swap_emits_subtle_and_gross():
    rows = perturb_number(_trio("Revenue was $1,577.00."), rng())
    mags = {r["magnitude"] for r in rows}
    assert mags == {"subtle", "gross"}
    assert all(r["perturbation_type"] == "number_swap" for r in rows)


def test_faithful_row_metadata():
    r = _faithful_row(_trio("Revenue was $1,577.00."))
    assert r["label"] == "faithful"
    assert r["perturbation_type"] == "none"
    assert r["original"] is None and r["replacement"] is None and r["magnitude"] is None


# ----------------------------------------------------------------------------- determinism


def test_perturbation_is_deterministic():
    trio = _trio("Revenue was $1,577.00 in FY2018.")
    assert perturb_number(trio, rng()) == perturb_number(trio, rng())
    assert perturb_date(trio, rng()) == perturb_date(trio, rng())


# --------------------------------------------------------------------- build_dataset (orchestrator)


def _config(tmp_path, enabled=None) -> dict:
    return {
        "paths": {"testset": str(tmp_path / "testset.jsonl")},
        "perturbations": {
            "seed": SEED,
            "enabled": enabled or list(PERTURBATIONS),
            "subtle_gross_threshold": DEFAULT_SUBTLE_GROSS_THRESHOLD,
        },
    }


def test_build_dataset_emits_one_faithful_per_trio_and_is_safe(tmp_path):
    trios = [
        _trio("Revenue was $1,577.00 in FY2018.", "fiscal 2018 filing"),
        _trio("Net income rose 5.1% to 3.4 billion dollars.", "income statement"),
        _trio("Margins were qualitatively stable.", "no figures"),
    ]
    rows = build_dataset(_config(tmp_path), trios)

    faithful = [r for r in rows if r["label"] == "faithful"]
    broken = [r for r in rows if r["label"] == "unfaithful"]
    assert len(faithful) == len(trios)
    assert (tmp_path / "testset.jsonl").exists()
    for r in broken:
        assert r["perturbation_type"] != "none"
        assert not _in_source(r["replacement"], r["source_text"])


def test_build_dataset_respects_enabled_list(tmp_path):
    trios = [_trio("Revenue rose 5.1% in FY2018.", "income statement only")]
    rows = build_dataset(_config(tmp_path, enabled=["date_shift"]), trios)
    types = {r["perturbation_type"] for r in rows}
    assert types <= {"none", "date_shift"}


def test_build_dataset_is_deterministic(tmp_path):
    trios = [_trio("Revenue was $1,577.00 in FY2018.", "fiscal 2018 filing")]
    a = build_dataset(_config(tmp_path / "a"), trios)
    b = build_dataset(_config(tmp_path / "b"), trios)
    assert a == b
