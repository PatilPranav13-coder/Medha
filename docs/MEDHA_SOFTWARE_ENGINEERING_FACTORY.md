# Medha Software Engineering Data Factory

The companion package `Medha_Software_Engineering_Data_Factory.zip` is the first implementation of a local-first data operation for improving Medha's software-engineering skills.

## Capability coverage

- Multi-language code: Python, JavaScript, TypeScript, Java, C, C++, Swift, Kotlin, C#, Go, Rust, Ruby, PHP, Dart, Scala, SQL and more.
- App development: frontend, backend, full-stack and mobile source structures.
- Engineering workflows: debugging, refactoring, performance improvements, automated tests and CI verification.
- Repository understanding: manifests, dependencies, README, source-file map, test locations, API contracts and database files.
- Architecture: review prompts for module boundaries, API/database design, data flow, failure modes and test strategy.

## Factory stages

1. Discover public GitHub repository metadata without downloading source.
2. Require a human-reviewed source/licence manifest before harvesting.
3. Harvest one repository at a time with hard file-count and byte limits.
4. Filter binaries, dependency/build directories, common secrets and unsafe paths.
5. Tag languages, frameworks and engineering domains; preserve commit/file provenance and SHA-256 hashes.
6. Deduplicate exact content globally and keep repository groups together across train/validation/test.
7. Create repository-understanding and architecture-review task candidates. These are not training-eligible until reviewed.
8. Run an explicitly requested test command in a restricted, no-network Docker container.
9. Promote code-change answers into SFT only when human review, factuality/safety checks and verification evidence pass.

## Local commands

After extracting the companion package into the Medha repository root:

```bash
python -m backend.medha_data_factory discover --query 'topic:fastapi stars:>100 language:Python' --limit 20

python -m backend.medha_data_factory inspect --repo /path/to/already-cloned/repository

python -m backend.medha_data_factory harvest \
  --repo owner/repository \
  --manifest datasets/medha_dataset_starter_pack/config/repository_sources.json \
  --source-id owner/repository \
  --out data/software_engineering/owner-repository

python -m unittest backend.test_main backend.test_data_factory -v
```

The example manifest is deliberately held. A person must check the repository's licence, upstream code/data rights, attribution and exact revision before setting an approved status.

## Important boundary

This factory creates candidate corpora and audited training records. It does not download all of The Stack, does not execute code during ingestion, and does not modify model weights. Fine-tuning Medha and demonstrating coding improvements require a model training job and an independent, sealed evaluation suite. The chat endpoint remains a placeholder until a model inference backend is connected.

## Current implementation status

The package contains the Python factory, GitHub metadata discovery and bounded file harvesting, repository ingestion, SWE-Gym task adapter, Docker verification helper, exact deduplication, JSON Schema checks, review-gated SFT promotion, documentation and synthetic tests. Its local validation run completed 13 passing tests.
