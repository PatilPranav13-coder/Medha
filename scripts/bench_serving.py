"""Measure first-token latency and output speed at several local server loads."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from pathlib import Path

from evals.capability.runner import make_brain
from backend.medha_brain import GenerationParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--brain", choices=("openai_compat", "medha"), default="openai_compat")
    parser.add_argument("--out", default="reports/serving.json")
    parser.add_argument("--requests-per-worker", type=int, default=3)
    args = parser.parse_args()
    if args.requests_per_worker < 1:
        parser.error("--requests-per-worker must be positive")
    brain = make_brain(args.brain, args.model)
    brain.warmup()

    def sample(_: int) -> dict[str, float | str]:
        started = time.perf_counter()
        first = None
        chunks = []
        try:
            for chunk in brain.generate_stream(
                [{"role": "user", "content": "Explain how to check a simple engineering calculation."}],
                GenerationParams(max_tokens=128, temperature=0.2, top_p=0.9),
            ):
                if first is None:
                    first = time.perf_counter() - started
                chunks.append(chunk)
            elapsed = max(0.000001, time.perf_counter() - started)
            return {"status": "ok", "first_token_seconds": first or elapsed,
                    "tokens_per_second_estimate": (len("".join(chunks)) / 4) / elapsed,
                    "elapsed_seconds": elapsed}
        except Exception as exc:
            return {"status": type(exc).__name__, "first_token_seconds": 0.0,
                    "tokens_per_second_estimate": 0.0, "elapsed_seconds": time.perf_counter() - started}

    data = {"model": args.model, "brain": args.brain, "concurrency": {}}
    for concurrency in (1, 2, 4):
        count = concurrency * args.requests_per_worker
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(sample, range(count)))
        elapsed = max(0.000001, time.perf_counter() - started)
        successful = [row for row in results if row["status"] == "ok"]
        first_tokens = [float(row["first_token_seconds"]) for row in successful]
        rates = [float(row["tokens_per_second_estimate"]) for row in successful]
        data["concurrency"][str(concurrency)] = {
            "requests": count, "errors": count - len(successful),
            "wall_seconds": round(elapsed, 6),
            "time_to_first_token_p50_seconds": round(statistics.median(first_tokens), 6) if first_tokens else None,
            "time_to_first_token_p95_seconds": round(sorted(first_tokens)[max(0, math.ceil(0.95 * len(first_tokens)) - 1)], 6) if first_tokens else None,
            "tokens_per_second_mean": round(statistics.mean(rates), 3) if rates else 0.0,
        }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote serving measurements to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
