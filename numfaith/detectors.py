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

import json
import os
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


class NLIDetector:
    """A DeBERTa-MNLI entailment baseline: does the answer entail from the source?

    Premise = source, hypothesis = answer. A low entailment probability means the
    answer is not supported. ``score`` is ``1 - P(entailment)`` and the verdict is
    ``unfaithful`` when ``P(entailment)`` falls below ``threshold``.
    """

    name = "nli"

    def __init__(self, model: str, threshold: float = 0.5):
        self.model_name = model
        self.threshold = threshold
        self._tok = None
        self._model = None
        self._entail_idx = None

    def _ensure_model(self):
        if self._model is None:
            import torch  # noqa: F401  (imported for availability; used in detect)
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            # Locate the "entailment" class index robustly across label naming schemes.
            label2id = {k.lower(): v for k, v in self._model.config.label2id.items()}
            self._entail_idx = next(
                (v for k, v in label2id.items() if "entail" in k), None
            )
            if self._entail_idx is None:
                raise ValueError(f"No entailment label in {self._model.config.label2id}")
        return self._model

    def detect(self, source: str, question: str, answer: str) -> dict:
        try:
            import torch

            model = self._ensure_model()
            inputs = self._tok(
                source, answer, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            p_entail = float(probs[self._entail_idx])
        except Exception as exc:
            return _result(self.name, SKIPPED, None, f"{type(exc).__name__}: {exc}")

        verdict = UNFAITHFUL if p_entail < self.threshold else FAITHFUL
        return _result(self.name, verdict, 1.0 - p_entail)


_JUDGE_SYSTEM = (
    "You are a strict faithfulness grader for retrieval-augmented answers. Given a "
    "SOURCE passage, a QUESTION, and an ANSWER, decide whether every claim in the "
    "ANSWER is fully supported by the SOURCE. If any number, date, entity, or direction "
    "is unsupported or contradicted, the answer is unfaithful. Respond with ONLY a JSON "
    'object: {"verdict": "faithful" | "unfaithful", "confidence": <0..1>} where '
    "confidence is your probability that the answer is UNFAITHFUL."
)


class LLMJudgeDetector:
    """An LLM acting as the expensive-but-accurate upper-reference detector.

    Reads its API key from the environment; when the key is absent it returns a
    ``skipped`` result rather than failing, so the rest of the suite still runs.
    """

    name = "llm_judge"

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ[self.api_key_env])
        return self._client

    def detect(self, source: str, question: str, answer: str) -> dict:
        if not os.getenv(self.api_key_env):
            return _result(self.name, SKIPPED, None, f"no API key in ${self.api_key_env}")
        try:
            client = self._ensure_client()
            user = f"SOURCE:\n{source}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            verdict = UNFAITHFUL if str(data.get("verdict")).lower() == UNFAITHFUL else FAITHFUL
            conf = data.get("confidence")
            score = float(conf) if conf is not None else (1.0 if verdict == UNFAITHFUL else 0.0)
        except Exception as exc:
            return _result(self.name, ERROR, None, f"{type(exc).__name__}: {exc}")
        return _result(self.name, verdict, score)


# Maps a config detector name to a factory taking its per-detector settings dict.
DETECTOR_FACTORIES: dict[str, Callable[[dict], object]] = {
    "lettucedetect": lambda cfg: LettuceDetectDetector(
        model_path=cfg.get("model_path", "KRLabsOrg/lettucedect-base-modernbert-en-v1")
    ),
    "nli": lambda cfg: NLIDetector(
        model=cfg.get("model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
        threshold=cfg.get("threshold", 0.5),
    ),
    "llm_judge": lambda cfg: LLMJudgeDetector(
        model=cfg.get("model", "gpt-4.1-mini"),
        api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
    ),
}


def get_detectors(config: dict) -> list:
    """Instantiate the detectors listed under ``config['detectors']['enabled']``."""
    dcfg = config.get("detectors", {})
    detectors = []
    for name in dcfg.get("enabled", []):
        factory = DETECTOR_FACTORIES.get(name)
        if factory is None:
            continue
        detectors.append(factory(dcfg.get(name, {})))
    return detectors
