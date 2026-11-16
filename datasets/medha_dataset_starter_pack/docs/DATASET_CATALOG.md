# Medha dataset catalogue

Observed Hugging Face cards/viewers reviewed 2026-11-16. Re-check exact features at the pinned revision before production ingest.

| Dataset | Observed schema | Medha use | Licence / access review |
|---|---|---|---|
| FineWeb-Edu — https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | text, id, dump, url, date, file_path, language, language_score, token_count, score, int_score | Educational pretraining | ODC-By 1.0 + CommonCrawl terms; underlying web-page content rights need separate review. Start with sample-10BT and bounded streaming. |
| FineWeb — https://huggingface.co/datasets/HuggingFaceFW/fineweb | text, id, dump, url, date, file_path, language, language_score, token_count | Broad pretraining backfill | ODC-By 1.0 + CommonCrawl terms; same underlying content caveat. Current viewer reports ~25.9B rows / 54.8 TB. Choose one crawl config, not default. |
| Proof-Pile-2 — https://huggingface.co/datasets/EleutherAI/proof-pile-2 | text, meta (JSON string; source-specific schema) | Mathematics, science, formal methods, mathematical code | It preserves the licences of underlying components; no umbrella licence. Components include ArXiv, OpenWebMath and AlgebraicStack. Viewer warns arbitrary Python code is required; do not enable remote code by default. |
| NuminaMath-CoT — https://huggingface.co/datasets/AI-MO/NuminaMath-CoT | source, problem, solution, messages | Maths SFT | Dataset card declares Apache-2.0; card describes PDF/forum sources, OCR, translation, synthetic data. Review source rights and validate answers independently. |
| OASST1 — https://huggingface.co/datasets/OpenAssistant/oasst1 | message_id, parent_id, user_id, created_date, text, role, lang, review_count, review_result, deleted, rank, synthetic, model_name, detoxify, message_tree_id, tree_state, emojis, labels | Conversation SFT | Apache-2.0. This is a message graph, not flat prompt/response rows. Reconstruct by tree; filter deleted, spam and PII. |
| HelpSteer2 — https://huggingface.co/datasets/nvidia/HelpSteer2 | prompt, response, helpfulness, correctness, coherence, complexity, verbosity | Preference / reward modelling | CC-BY-4.0. 21,362 samples; official train 20,324 and validation 1,038. Keep five attributes separate and keep validation out of training. |
| IndicCorp v2 — https://huggingface.co/datasets/ai4bharat/IndicCorpV2 | text; 23 language-specific splits | Indic-language pretraining | Dataset card states CC-0 for datasets produced by the authors; check underlying source/content rights. Sample per-language split and preserve language code. |
| The Stack v2 — https://huggingface.co/datasets/bigcode/the-stack-v2 | Gated; inspect actual feature names only after access | Code pretraining (conditional) | HOLD: bulk download requires Software Heritage/INRIA agreement; obey Software Heritage model-training principles, each source repository licence/attribution and validated removal updates. |

## Sampling policy

1. Read card, licence, terms, citation and schema.
2. Use `streaming=True` and hard row caps (25–100 for first inspection).
3. FineWeb-Edu: `sample-10BT`; FineWeb: one explicit crawl.
4. Proof-Pile-2: inspect components individually; retain raw `meta`.
5. Numina: stratify by `source`; check solutions.
6. OASST1: sample whole `message_tree_id` groups.
7. HelpSteer2: group by normalized prompt and preserve official split.
8. IndicCorpV2: sample each selected language directory separately.
9. The Stack v2: no sampling until access and licence gates pass.

## Cross-source overlap

FineWeb-Edu is derived from FineWeb. Deduplicate both globally by normalized text hash and crawl/source URL; prefer the educationally filtered copy when appropriate. Proof-Pile-2 may also overlap web/math corpora. Retain a duplicate audit ledger listing all source locations and licences even when only one canonical record is kept.
