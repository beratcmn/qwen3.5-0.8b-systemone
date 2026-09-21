import base64
import io
import os

import pytest
import torch
from PIL import Image

from qwen3_5_0_8b_systemone.contract import SystemOneRequest
from qwen3_5_0_8b_systemone.engine import get_engine
from qwen3_5_0_8b_systemone.scoring import answer_from_logits

pytestmark = pytest.mark.skipif(
    os.getenv("SYSTEMONE_INTEGRATION") != "1",
    reason="set SYSTEMONE_INTEGRATION=1 to run the real Qwen GPU checks",
)


def _assert_close(left: object, right: object) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_close(left[key], right[key])
    elif isinstance(left, float) and isinstance(right, float):
        assert left == pytest.approx(right, abs=2e-2)
    else:
        assert left == right


def _request() -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "model": "qwen3.5-0.8b-systemone",
            "state": "Checkout is unavailable in production and there is no workaround.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "engineering": "Outages",
                        "billing": "Charges",
                        "sales": "Purchases",
                    },
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Does this need immediate attention?",
                },
            },
        }
    )


def test_cached_branch_matches_full_prompt() -> None:
    engine = get_engine()
    engine.load()
    request = _request()
    shared = {
        key: value.to(engine.device)
        for key, value in engine._shared_inputs(request, []).items()
    }
    branch = engine._compile("route", request.questions["route"])
    suffix = torch.tensor([branch.input_ids], device=engine.device)
    full_ids = torch.cat([shared["input_ids"], suffix], dim=1)
    full_mask = torch.ones_like(full_ids)

    engine.model.model.rope_deltas = None
    with torch.inference_mode():
        full = engine.model.model(
            input_ids=full_ids, attention_mask=full_mask, return_dict=True
        )
        prefix = engine.model.model(**shared, use_cache=True, return_dict=True)
    cached = engine._evaluate_batch(
        [branch], prefix.past_key_values, shared["attention_mask"], None
    )[0]
    candidate_ids = torch.tensor(branch.candidate_token_ids, device=engine.device)
    direct = torch.mv(
        engine.model.lm_head.weight.index_select(0, candidate_ids).float(),
        full.last_hidden_state[0, -1].float(),
    ).cpu()
    cached_answer = answer_from_logits(branch.question, cached)
    direct_answer = answer_from_logits(branch.question, direct.tolist())
    assert cached_answer.choice == direct_answer.choice
    _assert_close(cached_answer.probabilities, direct_answer.probabilities)


def test_batched_questions_match_isolated_questions() -> None:
    engine = get_engine()
    request = _request()
    original_batch_size = engine.batch_size
    try:
        engine.batch_size = 1
        isolated = engine.evaluate(request)
        engine.batch_size = 4
        batched = engine.evaluate(request)
    finally:
        engine.batch_size = original_batch_size
    _assert_close(batched.model_dump(), isolated.model_dump())


def test_native_image_path() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (230, 20, 20)).save(buffer, format="PNG")
    request = SystemOneRequest.model_validate(
        {
            "model": "qwen3.5-0.8b-systemone",
            "state": "Inspect attachment `swatch`.",
            "attachments": [
                {
                    "id": "swatch",
                    "media_type": "image/png",
                    "data_base64": base64.b64encode(buffer.getvalue()).decode(),
                }
            ],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What is the dominant color?",
                    "criteria": {"red": None, "blue": None, "green": None},
                }
            },
        }
    )
    answer = get_engine().evaluate(request).answers["color"]
    assert answer.type == "choice"
    assert answer.choice == "red"
    assert answer.probabilities["red"] > 0.8
