from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx


QUESTIONS = [
    {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Charges, refunds, invoices, or payments.",
            "technical": "Product failures or account access.",
            "sales": "Purchases, upgrades, or pricing.",
        },
    },
    {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": [
            "Can wait.",
            "Low priority.",
            "Normal priority.",
            "Urgent.",
            "Immediate action required.",
        ],
    },
    {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
    },
]


def request_with(question_count: int) -> dict[str, object]:
    return {
        "model": "qwen3.5-0.8b-systemone",
        "state": (
            "I was billed twice for my subscription. "
            "Please refund the duplicate charge."
        ),
        "questions": {
            f"q{index}": QUESTIONS[index % len(QUESTIONS)]
            for index in range(question_count)
        },
    }


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the HTTP API with a repeatable short-text workload."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 5, 10, 50])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=12)
    args = parser.parse_args()

    results = []
    with httpx.Client(base_url=args.url, timeout=120) as client:
        client.get("/healthz").raise_for_status()
        for question_count in args.questions:
            payload = request_with(question_count)
            for _ in range(args.warmups):
                client.post("/v1/systemone", json=payload).raise_for_status()

            timings = []
            response = None
            for _ in range(args.runs):
                started = time.perf_counter()
                response = client.post("/v1/systemone", json=payload)
                response.raise_for_status()
                timings.append((time.perf_counter() - started) * 1000)

            assert response is not None
            median = statistics.median(timings)
            results.append(
                {
                    "questions": question_count,
                    "median_ms": round(median, 2),
                    "p95_ms": round(percentile_95(timings), 2),
                    "questions_per_second": round(
                        question_count / (median / 1000), 2
                    ),
                    "input_tokens": response.json()["usage"]["input_tokens"],
                }
            )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
