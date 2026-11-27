"""Run original capability tasks against the configured replaceable brain."""
from __future__ import annotations

import json
import csv
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.medha_brain import Brain, GenerationParams, build_brain
from evals.capability.bank import load_tasks
from evals.capability.checkers import check_answer

SYSTEM = "You are Medha, a helpful, honest assistant. Follow the user instructions. Do not invent facts."


def _wilson(passed: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = passed / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _generate(brain: Brain, messages: list[dict[str, str]]) -> tuple[str, float, float]:
    started = time.perf_counter()
    first_token: float | None = None
    chunks = []
    for chunk in brain.generate_stream(messages, GenerationParams(max_tokens=512, temperature=0.0, top_p=1.0)):
        if first_token is None:
            first_token = time.perf_counter() - started
        chunks.append(chunk)
    elapsed = max(0.000001, time.perf_counter() - started)
    return "".join(chunks), first_token if first_token is not None else elapsed, elapsed


def _rss_peak_mb() -> float | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 * 1024) if value > 10_000_000 else value / 1024
    except (ImportError, AttributeError, OSError):
        return None


def run_suite(brain: Brain, tasks: list[dict] | None = None) -> dict[str, Any]:
    task_list = tasks or load_tasks()
    results = []
    review_samples = []
    latencies: list[float] = []
    token_rates: list[float] = []
    for task in task_list:
        messages = [{"role": "system", "content": SYSTEM}]
        total_latency, first_token, generated = 0.0, None, ""
        error = None
        try:
            if task["checker"] == "multi_turn":
                for turn in task["turns"]:
                    messages.append({"role": "user", "content": turn})
                    generated, initial, elapsed = _generate(brain, messages)
                    total_latency += elapsed
                    first_token = initial if first_token is None else first_token
                    messages.append({"role": "assistant", "content": generated})
                passed = generated.strip() == task["expected"]
                reason = "" if passed else "final answer did not retain the scripted details"
            else:
                messages.append({"role": "user", "content": task["prompt"]})
                generated, first_token, total_latency = _generate(brain, messages)
                passed, reason = check_answer(task, generated)
        except Exception as exc:
            passed, reason, error = False, "brain or checker raised an error", type(exc).__name__
        total_latency = max(0.0, total_latency)
        latencies.append(total_latency)
        token_rates.append((len(generated) / 4) / max(total_latency, 0.000001))
        results.append({
            "id": task["id"], "category": task["category"], "passed": bool(passed),
            "latency_seconds": round(total_latency, 6), "time_to_first_token_seconds": round(first_token or 0.0, 6),
            "estimated_tokens_per_second": round(token_rates[-1], 3),
            "error_type": error, "checker_note": reason,
        })
        if task["category"] == "summarise_rewrite_translate":
            review_samples.append({"task_id": task["id"], "prompt": task["prompt"],
                                   "model_output": generated, "rating_1_to_5": "", "notes": ""})
    categories = sorted({task["category"] for task in task_list})
    summary = {}
    for category in categories:
        rows = [row for row in results if row["category"] == category]
        passed = sum(row["passed"] for row in rows)
        low, high = _wilson(passed, len(rows))
        summary[category] = {"passed": passed, "total": len(rows), "rate": passed / len(rows), "wilson_95": [low, high]}
    ordered = sorted(latencies)
    def percentile(p: float) -> float:
        if not ordered:
            return 0.0
        index = max(0, math.ceil(p * len(ordered)) - 1)
        return ordered[index]
    return {
        "task_count": len(results), "categories": summary, "latency_seconds": {
            "p50": percentile(0.50), "p95": percentile(0.95), "mean": statistics.mean(latencies) if latencies else 0.0,
        },
        "estimated_tokens_per_second_mean": statistics.mean(token_rates) if token_rates else 0.0,
        "peak_rss_mb": _rss_peak_mb(), "results": results, "review_samples": review_samples[:30],
    }


def _markdown(model: str, report: dict[str, Any], category_bar: dict[str, float]) -> str:
    lines = [f"# Capability report: {model}", "", f"Tasks run: {report['task_count']}", "", "| Category | Pass rate | 95% Wilson interval | Gate |", "|---|---:|---:|---:|"]
    for category, values in report["categories"].items():
        threshold = category_bar.get(category)
        gate = "n/a" if threshold is None else ("PASS" if values["rate"] >= threshold else "BELOW BAR")
        low, high = values["wilson_95"]
        lines.append(f"| {category} | {values['passed']}/{values['total']} ({values['rate']:.1%}) | {low:.1%}–{high:.1%} | {gate} |")
    latency = report["latency_seconds"]
    lines += ["", f"Latency: p50 {latency['p50']:.3f}s, p95 {latency['p95']:.3f}s; estimated output {report['estimated_tokens_per_second_mean']:.2f} tokens/s.", f"Peak process RSS: {report['peak_rss_mb']} MB (null means unavailable).", "", "Scores are measured only for the run represented by this report. The 95% Wilson interval is descriptive and does not establish parity with another product."]
    return "\n".join(lines) + "\n"


def write_report(model: str, output_dir: str | Path, report: dict[str, Any]) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    bar = json.loads(Path(__file__).with_name("bar.json").read_text(encoding="utf-8"))
    report["model"] = model
    report["thresholds"] = bar
    (directory / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "report.md").write_text(_markdown(model, report, bar), encoding="utf-8")
    with (directory / "human_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id", "prompt", "model_output", "rating_1_to_5", "notes"))
        writer.writeheader()
        writer.writerows(report.get("review_samples", []))
    comparison = []
    for path in sorted(directory.parent.glob("*/metrics.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("model"):
            comparison.append(record)
    lines = ["# Model comparison", "", "| Model | " + " | ".join(bar) + " |", "|---|" + "---:|" * len(bar)]
    for record in comparison:
        scores = [f"{record.get('categories', {}).get(category, {}).get('rate', 0):.1%}" for category in bar]
        lines.append("| " + str(record["model"]) + " | " + " | ".join(scores) + " |")
    (directory.parent / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def make_brain(provider: str, model: str | None) -> Brain:
    settings = Settings.from_env()
    settings = replace(settings, brain_provider=provider, brain_model=model or settings.brain_model)
    if provider == "stub":
        settings = replace(settings, brain_model="stub")
    settings.validate()
    return build_brain(settings)
