from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from .core import excluded_reason, secret_type, detect_language, validate_source_policy

API_ROOT = "https://api.github.com"


class GitHubAPIError(RuntimeError):
    pass


def _request_json(url: str, timeout: int = 20) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Medha-Data-Factory/0.3.0",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(10_000_001)
            if len(body) > 10_000_000:
                raise GitHubAPIError("GitHub API response exceeded the 10 MB safety limit.")
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", errors="replace")
        raise GitHubAPIError(f"GitHub API returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GitHubAPIError(f"GitHub API request failed: {exc}") from exc


def discover_repositories(query: str, limit: int = 20, allowed_licenses: set[str] | None = None) -> list[dict[str, Any]]:
    """Search public GitHub repository metadata only. Discovery never downloads code."""
    if not query.strip():
        raise ValueError("A GitHub search query is required.")
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    encoded = urllib.parse.urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": limit})
    data = _request_json(f"{API_ROOT}/search/repositories?{encoded}")
    allowed = {item.lower() for item in allowed_licenses} if allowed_licenses else None
    results = []
    for repo in data.get("items", []):
        license_obj = repo.get("license") or {}
        spdx = license_obj.get("spdx_id") or "NOASSERTION"
        if allowed is not None and spdx.lower() not in allowed:
            continue
        results.append({
            "repository": repo.get("full_name"),
            "url": repo.get("html_url"),
            "description": repo.get("description"),
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "language": repo.get("language"),
            "license_spdx_id": spdx,
            "license_url": license_obj.get("url"),
            "archived": repo.get("archived", False),
            "size_kb": repo.get("size", 0),
            "default_branch": repo.get("default_branch"),
            "pushed_at": repo.get("pushed_at"),
            "metadata_only": True,
            "requires_human_license_review": True,
        })
    return results


def _request_bytes(url: str, timeout: int = 30, max_bytes: int = 2_000_000) -> bytes:
    headers = {"User-Agent": "Medha-Data-Factory/0.3.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith(API_ROOT):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise GitHubAPIError(f"File exceeded the {max_bytes}-byte download limit.")
            return body
    except urllib.error.HTTPError as exc:
        raise GitHubAPIError(f"Download failed with HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GitHubAPIError(f"Download failed: {exc}") from exc


def download_selected_repository(
    repository: str,
    source: dict[str, Any],
    destination: Path,
    *,
    max_files: int = 1000,
    max_file_bytes: int = 256_000,
    max_total_bytes: int = 25_000_000,
) -> dict[str, Any]:
    """Download bounded text/source files from one explicitly approved repository."""
    validate_source_policy(source)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must be owner/name.")
    if source["repository_url"].rstrip("/").lower() != f"https://github.com/{repository}".lower():
        raise PermissionError("Repository URL does not match the reviewed source manifest.")
    if (max_files < 1 or max_files > 5000 or max_file_bytes < 1 or
            max_file_bytes > 2_000_000 or max_total_bytes < 1 or max_total_bytes > 200_000_000):
        raise ValueError("Download limits exceed safe configured bounds.")

    metadata = _request_json(f"{API_ROOT}/repos/{repository}")
    if metadata.get("private") or metadata.get("archived") or metadata.get("disabled"):
        raise PermissionError("Private, archived or disabled repositories are not ingested.")
    reported = ((metadata.get("license") or {}).get("spdx_id") or "NOASSERTION")
    if reported != source["declared_license"] or reported in {"NOASSERTION", "Other"}:
        raise PermissionError(f"GitHub reports licence {reported!r}, manifest declares {source['declared_license']!r}.")

    pinned = source.get("revision")
    if pinned:
        revision = str(pinned).strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            raise ValueError("revision must be a full Git commit SHA.")
    else:
        branch = metadata.get("default_branch")
        if not branch:
            raise GitHubAPIError("Repository metadata has no default branch.")
        branch_path = urllib.parse.quote(branch, safe="/")
        ref = _request_json(f"{API_ROOT}/repos/{repository}/git/ref/heads/{branch_path}")
        revision = (ref.get("object") or {}).get("sha")
        if not revision:
            raise GitHubAPIError("Branch reference did not return a commit SHA.")

    commit = _request_json(f"{API_ROOT}/repos/{repository}/git/commits/{revision}")
    tree_sha = (commit.get("tree") or {}).get("sha")
    if not tree_sha:
        raise GitHubAPIError("Pinned commit did not return its root tree SHA.")
    tree = _request_json(f"{API_ROOT}/repos/{repository}/git/trees/{tree_sha}?recursive=1")
    if tree.get("truncated"):
        raise GitHubAPIError("Git tree is truncated; refusing an incomplete repository snapshot.")

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Download destination must be empty: {destination}")

    total = 0
    downloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in tree.get("tree", []):
        if item.get("type") != "blob" or item.get("mode") == "120000":
            continue
        raw_path = item.get("path", "")
        pure = PurePosixPath(raw_path)
        if not raw_path or pure.is_absolute() or ".." in pure.parts:
            skipped.append({"path": raw_path, "reason": "unsafe_path"})
            continue
        relative = Path(*pure.parts)
        reason = excluded_reason(relative)
        if reason:
            skipped.append({"path": raw_path, "reason": reason})
            continue
        if detect_language(relative) is None:
            skipped.append({"path": raw_path, "reason": "unsupported_extension"})
            continue
        announced = int(item.get("size") or 0)
        if announced <= 0 or announced > max_file_bytes:
            skipped.append({"path": raw_path, "reason": "file_size_limit_or_empty"})
            continue
        if len(downloaded) >= max_files:
            skipped.append({"path": raw_path, "reason": "max_files_limit_reached"})
            break
        if total + announced > max_total_bytes:
            skipped.append({"path": raw_path, "reason": "max_total_bytes_limit_reached"})
            break
        raw_url = f"https://raw.githubusercontent.com/{repository}/{revision}/{urllib.parse.quote(raw_path, safe='/')}"
        content = _request_bytes(raw_url, max_bytes=max_file_bytes)
        if len(content) != announced:
            skipped.append({"path": raw_path, "reason": "size_mismatch"})
            continue
        if b"\x00" in content:
            skipped.append({"path": raw_path, "reason": "binary_content"})
            continue
        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            skipped.append({"path": raw_path, "reason": "non_utf8_content"})
            continue
        detected_secret = secret_type(decoded)
        if detected_secret:
            skipped.append({"path": raw_path, "reason": "secret_pattern_detected", "secret_type": detected_secret})
            continue
        target = (destination / relative).resolve()
        if destination not in target.parents:
            skipped.append({"path": raw_path, "reason": "path_escapes_destination"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        total += len(content)
        downloaded.append({"path": raw_path, "bytes": len(content), "blob_sha": item.get("sha")})

    return {
        "repository": repository,
        "repository_url": metadata.get("html_url"),
        "revision": revision,
        "reported_spdx_license": reported,
        "declared_license": source["declared_license"],
        "destination": str(destination),
        "files_downloaded": len(downloaded),
        "bytes_downloaded": total,
        "files_skipped": len(skipped),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "network_used": True,
        "code_executed": False,
        "note": "Only selected text/source files were fetched. GitHub repository licence metadata is not proof of per-file rights.",
    }
