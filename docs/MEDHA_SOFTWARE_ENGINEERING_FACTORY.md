# Medha Software Engineering Data Factory

The factory prepares data for Medha's software-engineering capabilities. It is a data operation, not a model fine-tuning run.

## Pipeline

1. **Discover:** search public GitHub repository metadata; no source files are downloaded.
2. **Approve:** maintain a human-reviewed manifest with dataset and underlying repository/file rights. Example entries start on hold.
3. **Collect:** harvest one approved repository at a time, with a pinned commit SHA, file count/size limits and path checks.
4. **Clean:** filter binaries, build/dependency directories, sensitive filenames and common secret patterns; hash and deduplicate text/code.
5. **Index:** detect languages and frameworks; retain paths, commit and licence provenance. Generate task candidates for architecture explanation, code explanation and test generation.
6. **Verify:** run only an explicitly supplied test command in a disposable, no-network Docker container with a read-only source mount.
7. **Export:** create pretraining records from approved source code, SFT records after human review and successful verification, preference pairs from non-tied comparisons, and sealed test-only evaluation records.

## Install and run from the repository root

```bash
python -m pip install -r backend/requirements.txt

# 1. Inspect an existing local clone without execution or network access.
python -m backend.medha_data_factory inspect --repo /path/to/project

# 2. Discover GitHub metadata only. No repository source is downloaded.
python -m backend.medha_data_factory discover --query 'topic:fastapi stars:>100 language:Python' --limit 20

# 3. Copy repository_sources.example.json and complete a real human rights review.
#    All template entries start with usage_status=hold; ingestion must stay blocked
#    until review, evidence, attribution and exact scope are filled.
python -m backend.medha_data_factory ingest \
  --repo /path/to/already-cloned/project \
  --manifest /path/to/reviewed_sources.json \
  --source-id owner/repository \
  --out data/software_engineering/owner-repository

# Or use bounded GitHub harvesting after explicit approval:
python -m backend.medha_data_factory harvest \
  --repo owner/repository \
  --manifest /path/to/reviewed_sources.json \
  --source-id owner/repository \
  --out data/software_engineering/owner-repository \
  --max-files 500 --max-total-bytes 15000000

# Optional test execution. Requires a trusted local runner image with dependencies pre-installed.
python -m backend.medha_data_factory verify \
  --repo /path/to/project --image medha-python-test:local \
  --test-command 'python -m unittest' \
  --out data/software_engineering/verification.json
```

## Four dataset formats

- **Pretraining:** `pretraining_train.jsonl`, `pretraining_validation.jsonl`, `pretraining_test.jsonl`. Every row validates against the Medha pretraining schema and carries file-level hash/provenance.
- **SFT:** task candidates are written to `task_candidates.jsonl` and are not training examples. Use `promote-sft` only after a reviewed answer plus passing build/tests and isolated verification.
- **Preference:** `make-preference` converts one explicit human comparison into chosen/rejected. Strength must be between −3 and +3; ties are rejected. Example input is in the training formats guide.
- **Evaluation:** `make-evaluation` creates records with `split=test`. Evaluation prompts, answers and hidden tests must remain sealed and must never be added to training.

## Source rights

Discovery metadata is not licence approval. Repository-level licence badges may not cover every file, dependency, copied component or upstream dataset. Each source registry entry needs a reviewer, date, licence evidence URL, scope, underlying licence and attribution terms. Entries in `software_engineering_sources.json` intentionally start on hold.

## Safety and leakage controls

- No code runs during discovery, download, inspection, deduplication or record export.
- Harvest is explicitly requested for one repository and uses immutable commit SHA, limits, path checks and secret scanning.
- Symlinks, private/archived repositories, binaries, dependency/build folders and likely secret files are rejected.
- Docker verification is opt-in, no-network, non-root, read-only-source, resource-limited and time-limited.
- Repositories are assigned deterministic 90/5/5 splits as indivisible groups.
- Exact duplicate handling prefers test, then validation, then train and writes a provenance ledger.
- No source licence is automatically approved. Unknown/held sources are blocked.
- The data factory prepares training records but does not fine-tune Medha or change model weights.

## Tests and CI

```bash
python -m unittest backend.test_main backend.test_data_factory -v
python -m backend.medha_data_factory --help
```

GitHub Actions runs these checks on factory-related pushes and pull requests.
