"""Uniform wrappers around off-the-shelf faithfulness detectors.

Every detector exposes the same interface::

    detector.detect(source, question, answer) -> {
        "detector": <name>,
        "verdict":  "faithful" | "unfaithful" | "skipped" | "error",
        "score":    float | None,   # UNFAITHFULNESS in [0, 1]; higher = more likely hallucinated
        "info":     <optional note, e.g. a skip/error reason>,
    }

``score`` is standardised as *unfaithfulness* so the evaluation harness (Phase 7) can
treat ``unfaithful`` as the positive class uniformly across detectors.

Each detector loads its model lazily (heavy imports happen inside methods, never at
module import) and degrades gracefully on its own: if a detector's dependency or model
fails to load it returns a ``skipped``/``error`` result instead of raising, so one
broken detector never takes down the others.
"""

from __future__ import annotations

from typing import Callable, Optional

FAITHFUL = "faithful"
UNFAITHFUL = "unfaithful"
SKIPPED = "skipped"
ERROR = "error"


def _result(
    detector: str,
    verdict: str,
    score: Optional[float],
    info: Optional[str] = None,
) -> dict:
    return {"detector": detector, "verdict": verdict, "score": score, "info": info}


class LettuceDetectDetector:
    """LettuceDetect — a pretrained ModernBERT span-level hallucination detector.

    The model flags hallucinated spans in the answer given the source as context. We
    convert that span output to a binary verdict: any flagged span ⇒ ``unfaithful``,
    with the score set to the strongest span confidence.
    """

    name = "lettucedetect"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._detector = None

    def _ensure_model(self):
        if self._detector is None:
            from lettucedetect.models.inference import HallucinationDetector

            self._detector = HallucinationDetector(
                method="transformer", model_path=self.model_path
            )
        return self._detector

    def detect(self, source: str, question: str, answer: str) -> dict:
        try:
            model = self._ensure_model()
            spans = model.predict(
                context=[source], question=question, answer=answer, output_format="spans"
            )
        except Exception as exc:  # missing dep, download failure, etc.
            return _result(self.name, SKIPPED, None, f"{type(exc).__name__}: {exc}")

        if spans:
            confidence = max(float(s.get("confidence", 1.0)) for s in spans)
            return _result(self.name, UNFAITHFUL, confidence)
        return _result(self.name, FAITHFUL, 0.0)


# Maps a config detector name to a factory taking its per-detector settings dict.
DETECTOR_FACTORIES: dict[str, Callable[[dict], object]] = {
    "lettucedetect": lambda cfg: LettuceDetectDetector(
        model_path=cfg.get("model_path", "KRLabsOrg/lettucedect-base-modernbert-en-v1")
    ),
}
