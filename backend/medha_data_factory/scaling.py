"""Target planning and progress accounting for Medha's 100M-example programme."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_TARGET = 100_000_000


def load_target_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_target_plan(plan)
    return plan


def _validate_percent_mix(name: str, values: dict[str, Any]) -> None:
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{name} must be a non-empty object.")
    for key, value in values.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer percentage.")
    if sum(values.values()) != 100:
        raise ValueError(f"{name} must add up to exactly 100; got {sum(values.values())}.")


def validate_target_plan(plan: dict[str, Any]) -> None:
    if plan.get("plan_version") != "1.0.0":
        raise ValueError("Unsupported 100M plan version.")
    target = plan.get("target_unique_examples")
    if type(target) is not int or target != EXPECTED_TARGET:
        raise ValueError(f"target_unique_examples must be exactly {EXPECTED_TARGET}.")
    _validate_percent_mix("language_primary_mix_percent", plan.get("language_primary_mix_percent"))
    _validate_percent_mix("task_type_mix_percent", plan.get("task_type_mix_percent"))
    _validate_percent_mix("split_percent", plan.get("split_percent"))
    if type(plan.get("full_stack_min_examples")) is not int or not 0 <= plan["full_stack_min_examples"] <= target:
        raise ValueError("full_stack_min_examples must be between zero and the total target.")
    if type(plan.get("quality_min_score")) is not int or not 0 <= plan["quality_min_score"] <= 100:
        raise ValueError("quality_min_score must be an integer from 0 to 100.")
    fraction = plan.get("synthetic_max_fraction")
    if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError("synthetic_max_fraction must be between 0 and 1.")
    if type(plan.get("target_shard_examples")) is not int or plan["target_shard_examples"] <= 0:
        raise ValueError("target_shard_examples must be a positive integer.")
    milestones = plan.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("At least one cumulative milestone is required.")
    previous = 0
    for milestone in milestones:
        value = milestone.get("cumulative_examples") if isinstance(milestone, dict) else None
        if type(value) is not int or not previous < value <= target:
            raise ValueError("Milestones must be strictly increasing positive counts within the target.")
        previous = value
    if previous != target:
        raise ValueError("The final milestone must equal the total target.")


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _quota_table(target: int, percentages: dict[str, int], observed: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    unknown = sorted(set(observed) - set(percentages))
    if unknown:
        raise ValueError("Unknown quota buckets: " + ", ".join(unknown))
    result: dict[str, dict[str, int | float]] = {}
    for key, percentage in percentages.items():
        actual = _count(observed.get(key, 0), f"observed.{key}")
        quota = target * percentage // 100
        result[key] = {
            "target": quota,
            "actual": actual,
            "remaining": max(0, quota - actual),
            "completion_percent": round((actual / quota) * 100, 2) if quota else 100.0,
        }
    return result


def build_progress_report(plan: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    """Compare observed counters against the approved target plan."""
    validate_target_plan(plan)
    total = _count(counts.get("unique_examples", 0), "unique_examples")
    if total > plan["target_unique_examples"]:
        raise ValueError("unique_examples cannot exceed the 100M target.")
    language_counts = counts.get("language_primary") or {}
    task_counts = counts.get("task_types") or {}
    split_counts = counts.get("splits") or {}
    if not all(isinstance(item, dict) for item in (language_counts, task_counts, split_counts)):
        raise ValueError("language_primary, task_types and splits must be objects.")
    full_stack = _count(counts.get("full_stack_examples", 0), "full_stack_examples")
    quality_passed = _count(counts.get("quality_passed_examples", 0), "quality_passed_examples")
    licence_approved = _count(counts.get("licence_approved_examples", 0), "licence_approved_examples")
    if quality_passed > total or licence_approved > total or full_stack > total:
        raise ValueError("Sub-counts cannot exceed unique_examples.")

    language = _quota_table(plan["target_unique_examples"], plan["language_primary_mix_percent"], language_counts)
    tasks = _quota_table(plan["target_unique_examples"], plan["task_type_mix_percent"], task_counts)
    splits = _quota_table(plan["target_unique_examples"], plan["split_percent"], split_counts)
    warnings: list[str] = []
    if total and sum(_count(v, f"language_primary.{k}") for k, v in language_counts.items()) != total:
        warnings.append("Primary-language counts do not sum to unique_examples.")
    if total and sum(_count(v, f"task_types.{k}") for k, v in task_counts.items()) != total:
        warnings.append("Task-type counts do not sum to unique_examples.")
    if total and sum(_count(v, f"splits.{k}") for k, v in split_counts.items()) != total:
        warnings.append("Split counts do not sum to unique_examples.")
    if quality_passed < total:
        warnings.append("Some counted examples have not passed the quality gate.")
    if licence_approved < total:
        warnings.append("Some counted examples do not yet have approved source-rights evidence.")
    if full_stack < min(total, plan["full_stack_min_examples"]):
        warnings.append("Full-stack coverage is below its minimum target.")

    target = plan["target_unique_examples"]
    return {
        "plan_version": plan["plan_version"],
        "target_unique_examples": target,
        "unique_examples": total,
        "remaining_examples": target - total,
        "overall_completion_percent": round(total / target * 100, 4),
        "full_stack": {
            "minimum_target": plan["full_stack_min_examples"],
            "actual": full_stack,
            "remaining": max(0, plan["full_stack_min_examples"] - full_stack),
        },
        "language_primary_quotas": language,
        "task_type_quotas": tasks,
        "split_quotas": splits,
        "quality_passed_examples": quality_passed,
        "quality_pass_rate_percent": round(quality_passed / total * 100, 2) if total else None,
        "licence_approved_examples": licence_approved,
        "licence_approval_rate_percent": round(licence_approved / total * 100, 2) if total else None,
        "warnings": warnings,
        "status": "target_reached" if total == target and not warnings else "in_progress",
    }
