from __future__ import annotations

from typing import Any

from .core import validate_source_policy


def adapt_swegym_row(
    row: dict[str, Any],
    dataset_source: dict[str, Any],
    repository_clearance: dict[str, Any],
) -> dict[str, Any]:
    """Convert a SWE-Gym task to a review candidate, never straight to training.

    Dataset rights and the underlying task repository rights must both be reviewed.
    The patch is not considered verified until applied at base_commit and tests pass.
    """
    validate_source_policy(dataset_source)
    validate_source_policy(repository_clearance)
    required = ("instance_id", "problem_statement", "patch", "repo", "base_commit")
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError("SWE-Gym row is missing: " + ", ".join(missing))
    repo_url = f"https://github.com/{row['repo']}"
    if repository_clearance["repository_url"].rstrip("/").lower() != repo_url.lower():
        raise PermissionError("Repository clearance does not match the repository in this SWE-Gym row.")
    base_commit = str(row["base_commit"])
    if len(base_commit) not in (40, 64) or any(c not in "0123456789abcdefABCDEF" for c in base_commit):
        raise ValueError("SWE-Gym base_commit must be a full Git commit SHA.")
    return {
        "candidate_id": f"swe-gym:{row['instance_id']}",
        "repository_id": row["repo"], "repository_url": repo_url,
        "base_commit": base_commit, "task_type": "debugging",
        "prompt": (
            "Fix the issue in the existing repository. Work from the exact base commit, "
            "understand the surrounding code, make the smallest correct change, and preserve behaviour.\n\n"
            f"Repository: {row['repo']}\nBase commit: {base_commit}\n\nIssue:\n{row['problem_statement']}"
        ),
        "solution_patch": row["patch"], "test_patch": row.get("test_patch"),
        "expected_tests": {
            "FAIL_TO_PASS": row.get("FAIL_TO_PASS") or [],
            "PASS_TO_PASS": row.get("PASS_TO_PASS") or [],
        },
        "dataset_split": row.get("split", "train"),
        "training_eligible": False,
        "status": "requires_base_checkout_patch_and_sandbox_verification",
        "verification": {
            "base_revision_reconstructed": False, "solution_patch_applied": False,
            "test_patch_applied": False, "fail_to_pass_passed": False,
            "pass_to_pass_preserved": False, "build_passed": False,
            "human_reviewed": False,
        },
        "provenance": {
            "dataset": dataset_source["repository_url"],
            "dataset_license": dataset_source["declared_license"],
            "dataset_revision": dataset_source.get("revision"),
            "dataset_row_id": str(row["instance_id"]),
            "source_repository": repo_url,
            "source_repository_license": repository_clearance["underlying_license"],
            "source_repository_license_evidence": repository_clearance["license_evidence_url"],
            "source_base_commit": base_commit,
        },
        "promotion_rule": "Require base checkout, patch application, FAIL_TO_PASS, PASS_TO_PASS, build and human review.",
    }
