from __future__ import annotations

import base64
import binascii
import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

Structured = str | list[JsonValue] | dict[str, JsonValue]


def _assert_finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numbers must be finite")
    if isinstance(value, list):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoulCriteria(StrictModel):
    true: Structured | None = None
    false: Structured | None = None

    _finite = field_validator("true", "false")(_assert_finite)


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Structured
    criteria: NoulCriteria | None = None

    _finite = field_validator("instructions")(_assert_finite)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Structured
    criteria: dict[str, Structured | None]

    _finite = field_validator("instructions", "criteria")(_assert_finite)

    @field_validator("criteria")
    @classmethod
    def validate_criteria(
        cls, criteria: dict[str, Structured | None]
    ) -> dict[str, Structured | None]:
        if not 1 <= len(criteria) <= 255:
            raise ValueError("Choice criteria must contain 1 to 255 options")
        if any(not name for name in criteria):
            raise ValueError("Choice option names must not be empty")
        return criteria


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Structured
    criteria: list[Structured]

    _finite = field_validator("instructions", "criteria")(_assert_finite)

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, criteria: list[Structured]) -> list[Structured]:
        if not 2 <= len(criteria) <= 10:
            raise ValueError("Score criteria must contain 2 to 10 levels")
        return criteria


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")
]


class Attachment(StrictModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    data_base64: str

    @field_validator("data_base64")
    @classmethod
    def validate_data(cls, data: str) -> str:
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("data_base64 must be valid base64") from error
        if len(decoded) > 4 * 1024 * 1024:
            raise ValueError("decoded image must not exceed 4 MiB")
        return data

    def decoded(self) -> bytes:
        return base64.b64decode(self.data_base64, validate=True)


class SystemOneRequest(StrictModel):
    model: Literal["qwen3.5-0.8b-systemone"]
    state: Structured
    questions: dict[str, Question]
    attachments: list[Attachment] = Field(default_factory=list, max_length=4)

    _finite = field_validator("state")(_assert_finite)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: dict[str, Question]) -> dict[str, Question]:
        if not 1 <= len(questions) <= 64:
            raise ValueError("questions must contain 1 to 64 entries")
        if any(not name for name in questions):
            raise ValueError("question ids must not be empty")
        return questions

    @model_validator(mode="after")
    def unique_attachment_ids(self) -> "SystemOneRequest":
        ids = [attachment.id for attachment in self.attachments]
        if len(ids) != len(set(ids)):
            raise ValueError("attachment ids must be unique")
        return self


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class SystemOneResponse(StrictModel):
    model: Literal["qwen3.5-0.8b-systemone"] = "qwen3.5-0.8b-systemone"
    answers: dict[str, Answer]
    usage: Usage
