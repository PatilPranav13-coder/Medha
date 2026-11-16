"""Evidence-based quality scoring before SFT examples are admitted to training."""
from __future__ import annotations

from typing import Any

MINIMUM_QUALITY_SCORE = 85


def score_sft_candidate(
    candidate: dict[str, Any],
    review: dict[str, Any],
    repository_index: dict[str, Any],
) -> dict[str, Any]:
    """Score a proposed SFT example. Mandatory evidence gates cannot be bypassed."""
    dimensions: dict[str, int] = {}
    blockers: list[str] = []
    notes: list[str] = []

    rights_ok = repository_index.get("usage_status") in {"approved", "approved_with_attribution"}
    dimensions["source_rights"] = 15 if rights_ok else 0
    if not rights_ok:
        blockers.append("source rights are not approved")

    provenance_fields = (
        repository_index.get("repository_url"),
        repository_index.get("revision"),
        repository_index.get("license_evidence_url"),
        repository_index.get("reviewer"),
        repository_index.get("reviewed_at"),
        candidate.get("candidate_id"),
    )
    provenance_ok = all(bool(value) for value in provenance_fields)
    dimensions["provenance_completeness"] = 15 if provenance_ok else 0
    if not provenance_ok:
        blockers.append("repository, revision, licence evidence, reviewer, date, or candidate ID is missing")

    for field, weight, label in (
        ("human_reviewed", 10, "human_review"),
        ("factuality_pass", 10, "factuality_review"),
        ("safety_reviewed", 10, "safety_review"),
    ):
        passed = review.get(field) is True
        dimensions[label] = weight if passed else 0
        if not passed:
            blockers.append(f"{field} must be true")

    prompt = str(candidate.get("prompt") or "").strip()
    answer = str(review.get("assistant_answer") or "").strip()
    content_score = 0
    if len(prompt) >= 20 and len(answer) >= 40:
        content_score = 5
    if len(prompt) >= 80:
        content_score += 3
    if len(answer) >= 300:
        content_score += 2
    dimensions["task_and_answer_specificity"] = min(10, content_score)
    if not prompt or not answer:
        blockers.append("task prompt and reviewed answer are required")
    elif len(prompt) < 20 or len(answer) < 40:
        blockers.append("task prompt or answer is too short to be a useful training example")

    if review.get("requires_execution", True):
        verification = review.get("verification_report") or {}
        verification_ok = (
            verification.get("status") == "passed"
            and verification.get("exit_code") == 0
            and verification.get("network_enabled") is False
            and verification.get("source_mount_read_only") is True
            and verification.get("repository_code_executed") is True
        )
        dimensions["isolated_verification"] = 20 if verification_ok else 0
        if not verification_ok:
            blockers.append("isolated verification must pass with network disabled and a read-only source mount")
        execution_quality_ok = review.get("build_passed") is True and int(review.get("tests_passed") or 0) >= 1
        dimensions["build_and_tests"] = 10 if execution_quality_ok else 0
        if not execution_quality_ok:
            blockers.append("a successful build and at least one passing test are required")
    else:
        verification_rationale = str(review.get("verification_rationale") or "").strip()
        test_rationale = str(review.get("test_coverage_rationale") or "").strip()
        dimensions["isolated_verification"] = 20 if len(verification_rationale) >= 50 else 0
        dimensions["build_and_tests"] = 10 if len(test_rationale) >= 50 else 0
        if len(verification_rationale) < 50:
            blockers.append("non-executable task needs a reviewer verification rationale")
        if len(test_rationale) < 50:
            blockers.append("non-executable task needs a testability or evidence rationale")

    score = sum(dimensions.values())
    eligible = score >= MINIMUM_QUALITY_SCORE and not blockers
    tier = "gold" if eligible and score >= 95 else ("accepted" if eligible else "rejected")
    if blockers:
        notes.extend("Rejected: " + item for item in blockers)
    return {
        "score": score,
        "minimum_score": MINIMUM_QUALITY_SCORE,
        "tier": tier,
        "eligible": eligible,
        "dimensions": dimensions,
        "blockers": blockers,
        "notes": notes,
    }
