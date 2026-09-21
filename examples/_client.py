from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from typing import Any

URL = os.getenv("SYSTEMONE_URL", "http://127.0.0.1:8000/v1/systemone")


def post(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code}: {error.read().decode()}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Cannot reach {URL}: {error.reason}") from error
    validate(result)
    return result


def validate(result: dict[str, Any]) -> None:
    assert result["model"] == "qwen3.5-0.8b-systemone"
    assert result["usage"]["output_tokens"] == 0
    for answer in result["answers"].values():
        if answer["type"] == "noul":
            assert 0 <= answer["noul"] <= 1
            continue
        probabilities = answer["probabilities"]
        assert math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)
        assert all(0 <= probability <= 1 for probability in probabilities.values())
        assert 0 <= answer["confidence"] <= 1
        if answer["type"] == "choice":
            assert answer["choice"] == max(probabilities, key=probabilities.get)
        elif answer["type"] == "score":
            expected = sum(
                int(level) * probability for level, probability in probabilities.items()
            )
            assert math.isclose(answer["score"], expected, abs_tol=1e-6)


def show(result: dict[str, Any]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
