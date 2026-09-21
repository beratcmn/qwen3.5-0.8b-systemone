import asyncio
import math

import httpx
import pytest
from pydantic import ValidationError

from qwen3_5_0_8b_systemone.app import app
from qwen3_5_0_8b_systemone.contract import (
    ChoiceQuestion,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from qwen3_5_0_8b_systemone.engine import get_engine
from qwen3_5_0_8b_systemone.scoring import answer_from_logits


def test_mixed_contract_and_answer_math() -> None:
    request = SystemOneRequest.model_validate(
        {
            "model": "qwen3.5-0.8b-systemone",
            "state": {"message": "Ödeme başarısız", "attempts": 3},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"engineering": None, "billing": {"covers": "charges"}},
                },
                "severity": {
                    "type": "score",
                    "instructions": {"question": "How severe?"},
                    "criteria": ["Low", {"meaning": "High"}],
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            },
        }
    )
    choice = answer_from_logits(request.questions["route"], [math.log(9), 0])
    assert choice.choice == "engineering"
    assert choice.probabilities == pytest.approx({"engineering": 0.9, "billing": 0.1})
    assert choice.confidence == pytest.approx(
        1 - (-(0.9 * math.log(0.9) + 0.1 * math.log(0.1))) / math.log(2)
    )

    score = answer_from_logits(request.questions["severity"], [0, math.log(3)])
    assert score.score == pytest.approx(0.75)
    assert score.legend == {"0": "Low", "1": '{"meaning":"High"}'}

    noul = answer_from_logits(request.questions["urgent"], [math.log(4), 0])
    assert noul.noul == pytest.approx(0.8)


def test_choice_and_score_limits() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(type="choice", instructions="pick", criteria={})
    ChoiceQuestion(
        type="choice",
        instructions="pick",
        criteria={str(index): None for index in range(255)},
    )
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            type="choice",
            instructions="pick",
            criteria={str(index): None for index in range(256)},
        )
    with pytest.raises(ValidationError):
        ScoreQuestion(type="score", instructions="rate", criteria=["only one"])


def test_api_rejects_duplicate_keys_and_serves_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = SystemOneResponse(
        answers={}, usage=Usage(input_tokens=1, output_tokens=0)
    )
    monkeypatch.setattr(get_engine(), "evaluate", lambda request: expected)
    payload = {
        "model": "qwen3.5-0.8b-systemone",
        "state": "hello",
        "questions": {"ok": {"type": "noul", "instructions": "Is this a greeting?"}},
    }

    async def check() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post("/v1/systemone", json=payload)
            assert response.status_code == 200
            assert response.json() == expected.model_dump(mode="json")

            duplicate = '{"model":"qwen3.5-0.8b-systemone","state":"x","state":"y","questions":{}}'
            response = await client.post(
                "/v1/systemone",
                content=duplicate,
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 422
            assert "duplicate JSON key" in response.json()["detail"]

    asyncio.run(check())


def test_connect_four_demo_is_served() -> None:
    async def check() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/demos/connect-four")
            assert response.status_code == 200
            assert "System One · Connect Four" in response.text
            assert 'fetch("/v1/systemone"' in response.text

    asyncio.run(check())


def test_rejects_nonfinite_nested_number() -> None:
    payload = {
        "model": "qwen3.5-0.8b-systemone",
        "state": {"bad": float("inf")},
        "questions": {"ok": {"type": "noul", "instructions": "Anything?"}},
    }
    with pytest.raises(ValidationError, match="finite"):
        SystemOneRequest.model_validate(payload)
