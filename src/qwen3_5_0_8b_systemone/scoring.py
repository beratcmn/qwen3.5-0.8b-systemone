from __future__ import annotations

import json
import math
from collections.abc import Sequence

from .contract import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("at least one logit is required")
    peak = max(logits)
    exponents = [math.exp(value - peak) for value in logits]
    total = sum(exponents)
    return [value / total for value in exponents]


def confidence(probabilities: Sequence[float]) -> float:
    if len(probabilities) == 1:
        return 1.0
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def _description(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def answer_from_logits(question: Question, logits: Sequence[float]) -> Answer:
    probabilities = softmax(logits)
    if isinstance(question, ChoiceQuestion):
        names = list(question.criteria)
        if len(logits) != len(names):
            raise ValueError("Choice logit count does not match its options")
        distribution = dict(zip(names, probabilities, strict=True))
        winner = max(range(len(names)), key=lambda index: probabilities[index])
        return ChoiceAnswer(
            choice=names[winner],
            probabilities=distribution,
            confidence=confidence(probabilities),
        )

    if isinstance(question, ScoreQuestion):
        if len(logits) != len(question.criteria):
            raise ValueError("Score logit count does not match its levels")
        keys = [str(index) for index in range(len(question.criteria))]
        return ScoreAnswer(
            score=sum(
                index * probability for index, probability in enumerate(probabilities)
            ),
            legend={
                key: _description(value)
                for key, value in zip(keys, question.criteria, strict=True)
            },
            probabilities=dict(zip(keys, probabilities, strict=True)),
            confidence=confidence(probabilities),
        )

    if isinstance(question, NoulQuestion):
        if len(logits) != 2:
            raise ValueError("Noul requires true and false logits")
        return NoulAnswer(noul=probabilities[0])

    raise TypeError(f"unsupported question type: {type(question).__name__}")
