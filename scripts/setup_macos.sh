#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Medha local setup (macOS / Linux) ==="

if [[ ! -f "backend/requirements.txt" ]]; then
  echo "ERROR: backend/requirements.txt not found. Run this from a cloned Medha repository." >&2
  exit 1
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "ERROR: Python 3.11 is required. Install Python 3.11, then rerun this script." >&2
  exit 1
fi

PYTHON="$(command -v python3.11)"
echo "Using Python: $("$PYTHON" --version)"

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON" -m venv .venv
else
  echo "Existing .venv found; reusing it."
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt

echo "Running Medha tests..."
.venv/bin/python -m unittest backend.test_main backend.test_data_factory -v

echo "Checking Medha data-factory CLI..."
.venv/bin/python -m backend.medha_data_factory --help

echo "Setup complete."
echo "Activate the environment with: source .venv/bin/activate"
echo "Example: python -m backend.medha_data_factory inspect --repo ."
echo "This sets up the data factory only. It does not download the 100M corpus or train model weights."
