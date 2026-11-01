# Medha backend

Medha is a programming tutor and app-building assistant. The backend uses a local Ollama model and can retrieve public developer documentation for each question.

## Run locally

Install Ollama, then download the default coding model:

```powershell
ollama pull qwen2.5-coder:7b
```

From the repository root, install the Python dependencies and start the API:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn main:app --app-dir backend --reload
```

Open http://127.0.0.1:8000. The API defaults to `http://127.0.0.1:11434` for Ollama. Configure another local Ollama URL or model with `MEDHA_OLLAMA_URL` and `MEDHA_OLLAMA_MODEL`.

## What the public lookup does

When “Search public sources” is enabled, Medha searches MDN Web Docs, Microsoft Learn, and Stack Overflow, then gives the retrieved excerpts to the local model and shows links with the answer. It also adds official SwiftUI, Android, Windows, Flutter, or React Native documentation for the selected target. The question text is sent to those public search services. Turn the checkbox off to keep the question out of those services.

Retrieved material is used at answer time (retrieval-augmented generation), not added to model weights. This keeps answers linked to their sources and avoids claiming that Medha has been fine-tuned on a corpus. Stack Overflow references include author and license attribution.

## Platform guidance

- iPhone, iPad, and Mac: SwiftUI
- Android: Kotlin and Jetpack Compose
- Windows: WinUI 3/.NET
- Shared app across iOS, iPadOS, Android, macOS, and Windows: Flutter

Medha teaches programming concepts or gives app-building steps based on the selected mode and target. It does not run, sign, or publish generated apps.
