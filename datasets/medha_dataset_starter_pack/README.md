# Medha dataset starter pack

This folder contains the canonical data formats and ingestion policy for Medha. The complete ZIP is available in the ChatGPT response; this GitHub folder carries the source documentation and JSON Schemas so they can be versioned with the codebase.

## Four independent data products
- **Pretraining:** high-quality raw text/code/math, with domain tags and provenance.
- **SFT:** validated user/assistant or multi-turn examples.
- **Preference:** chosen/rejected answers with separate 0–4 quality dimensions.
- **Evaluation:** sealed, independently reviewed tasks with rubrics and hidden tests.

## Files
- `docs/TRAINING_RECORD_FORMATS.md` — field-level guidance and examples.
- `docs/DATASET_CATALOG.md` — schemas and recommended roles for all eight selected sources.
- `docs/LICENCE_AND_PROVENANCE.md` — rights and attribution controls.
- `docs/TRAINING_PIPELINE.md` — bounded sampling, quality filters, dedupe and split rules.
- `schemas/*.schema.json` — strict JSON Schemas.

**No source dataset is copied here.** All ingestion must be bounded and explicit. Do not bulk-download the source corpora automatically. The Stack v2 remains on hold pending gated access, Software Heritage/INRIA conditions and file-level licence controls.
