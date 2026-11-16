from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def verify_in_docker(
    repo_path: Path,
    image: str,
    command: str,
    *,
    timeout_seconds: int = 180,
    memory_limit: str = "1g",
    cpus: str = "1",
    pids_limit: int = 128,
) -> dict[str, Any]:
    """Run an explicitly supplied test command in a disposable no-network container.

    This opt-in action is never called by ingestion. Dependencies must already be
    present in the local image because the container has no network connection.
    """
    root = repo_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not image.strip() or not command.strip():
        raise ValueError("A local runtime image and explicit test command are required.")
    if not 1 <= timeout_seconds <= 1800:
        raise ValueError("timeout_seconds must be between 1 and 1800.")
    if not 16 <= pids_limit <= 512:
        raise ValueError("pids_limit must be between 16 and 512.")
    script = "mkdir -p /work/project && cp -R /input/. /work/project/ && cd /work/project && " + command
    args = [
        "docker", "run", "--rm", "--network=none", "--read-only",
        f"--memory={memory_limit}", f"--cpus={cpus}", f"--pids-limit={pids_limit}",
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--tmpfs", "/work:rw,nosuid,nodev,size=768m,uid=65534,gid=65534,mode=700",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=128m,uid=65534,gid=65534,mode=700",
        "--mount", f"type=bind,src={root},dst=/input,readonly",
        "--user", "65534:65534", "--workdir", "/work",
        "--entrypoint", "/bin/sh", image, "-lc", script,
    ]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode, "stdout": result.stdout[-20000:],
            "stderr": result.stderr[-20000:], "image": image, "command": command,
            "timeout_seconds": timeout_seconds, "network_enabled": False,
            "source_mount_read_only": True, "repository_code_executed": True,
        }
    except FileNotFoundError as exc:
        raise RuntimeError("Docker was not found. Install Docker Desktop/Engine before verification.") from exc
    except subprocess.TimeoutExpired as exc:
        def tail(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")[-20000:]
            return str(value or "")[-20000:]
        return {
            "status": "timeout", "exit_code": None,
            "stdout": tail(exc.stdout), "stderr": tail(exc.stderr),
            "image": image, "command": command, "timeout_seconds": timeout_seconds,
            "network_enabled": False, "source_mount_read_only": True,
            "repository_code_executed": True,
        }
