$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host ""
Write-Host "=== Medha local setup (Windows) ===" -ForegroundColor Cyan

$Requirements = Join-Path $RepoRoot "backend\requirements.txt"
if (-not (Test-Path $Requirements)) {
    throw "backend\requirements.txt was not found. Run this script from a cloned Medha repository."
}

$PythonLauncher = Get-Command "py" -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    $PythonPath = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $PythonPath) {
        throw "Python 3.11 was not found. Install Python 3.11, enable the Python Launcher, then run this script again."
    }
    $PythonPath = $PythonPath.Trim()
} else {
    $PythonCommand = Get-Command "python" -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python was not found. Install Python 3.11 and make sure it is on PATH."
    }
    $PythonPath = (& python -c "import sys; print(sys.executable)" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not run Python. Check your Python installation."
    }
    $Version = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($Version -ne "3.11") {
        throw "This project CI uses Python 3.11. Detected Python $Version. Install Python 3.11 or use the py launcher."
    }
    $PythonPath = $PythonPath.Trim()
}

Write-Host "Using Python: $PythonPath" -ForegroundColor Green

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    & $PythonPath -m venv (Join-Path $RepoRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
} else {
    Write-Host "Existing .venv found; reusing it."
}

Write-Host "Installing dependencies..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host "Running Medha tests..."
& $VenvPython -m unittest backend.test_main backend.test_data_factory -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Copy the error output and share it for troubleshooting." }

Write-Host "Checking the Medha data-factory CLI..."
& $VenvPython -m backend.medha_data_factory --help
if ($LASTEXITCODE -ne 0) { throw "The Medha CLI check failed." }

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "To use Medha from this repository, run:"
Write-Host "  .\.venv\Scripts\python.exe -m backend.medha_data_factory --help"
Write-Host "Example: inspect the repository:"
Write-Host "  .\.venv\Scripts\python.exe -m backend.medha_data_factory inspect --repo ."
Write-Host "This sets up the data factory only. It does not download the 100M corpus or train model weights."
