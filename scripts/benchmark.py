from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qwen3_5_0_8b_systemone.contract import SystemOneRequest
from qwen3_5_0_8b_systemone.engine import get_engine


def request_with(question_count: int, state_repetitions: int) -> SystemOneRequest:
    state = " ".join(
        ["Checkout is unavailable in production and there is no workaround."]
        * state_repetitions
    )
    questions = {
        f"route_{index}": {
            "type": "choice",
            "instructions": "Which team should handle this incident?",
            "criteria": {
                "engineering": "Outages and deployment failures",
                "billing": "Charges and refunds",
                "sales": "Purchases and upgrades",
            },
        }
        for index in range(question_count)
    }
    return SystemOneRequest.model_validate(
        {
            "model": "qwen3.5-0.8b-systemone",
            "state": state,
            "questions": questions,
        }
    )


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--state-repetitions", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    engine = get_engine()
    engine.load()
    request = request_with(args.questions, args.state_repetitions)
    original_batch_size = engine.batch_size
    results = []
    try:
        independent = [
            request.model_copy(update={"questions": {question_id: question}})
            for question_id, question in request.questions.items()
        ]
        for item in independent:
            engine.evaluate(item)
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for item in independent:
                engine.evaluate(item)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
        results.append(
            {
                "questions": args.questions,
                "mode": "independent_requests",
                "median_ms": statistics.median(timings),
                "p95_ms": percentile_95(timings),
                "questions_per_second": args.questions
                / (statistics.median(timings) / 1000),
                "peak_vram_mib": torch.cuda.max_memory_allocated() / 1_048_576,
            }
        )
        for batch_size in args.batch_sizes:
            engine.batch_size = batch_size
            engine.evaluate(request)
            timings = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.runs):
                torch.cuda.synchronize()
                started = time.perf_counter()
                engine.evaluate(request)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
            results.append(
                {
                    "questions": args.questions,
                    "mode": "shared_prefill",
                    "batch_size": batch_size,
                    "median_ms": statistics.median(timings),
                    "p95_ms": percentile_95(timings),
                    "questions_per_second": args.questions
                    / (statistics.median(timings) / 1000),
                    "peak_vram_mib": torch.cuda.max_memory_allocated() / 1_048_576,
                }
            )
    finally:
        engine.batch_size = original_batch_size
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
