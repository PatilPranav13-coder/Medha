# Medha canonical training record formats

Use UTF-8 JSONL: exactly one complete JSON object per line. Preserve the unmodified source row separately; these are canonical Medha records, not the original source schemas.

## Pretraining — `schemas/pretraining.schema.json`

Required: `schema_version`, `record_id`, `text`, `content_type`, `domain_tags`, `language`, `provenance`.

Example:
```json
{"schema_version":"1.0.0","record_id":"fineweb-edu:000001","text":"A MOSFET uses an electric field to control current through a semiconductor channel.","content_type":"text","domain_tags":["semiconductors.devices","physics.electromagnetism"],"language":"en","quality_score":4.2,"source_metadata":{"dump":"CC-MAIN-2025-05","url":"https://example.org/page"},"provenance":{"source_dataset":"HuggingFaceFW/fineweb-edu","source_revision":"PINNED_COMMIT","source_config":"sample-10BT","source_split":"train","source_record_id":"source-row-id","declared_license":"ODC-By-1.0","underlying_license":"review-required","usage_status":"conditional","content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pipeline_version":"git-sha","transformation_version":"1.0.0","source_metadata":{"dump":"CC-MAIN-2025-05"}}}
```

Chunk by semantic boundaries; keep document ID, chunk index, source URL, crawl and hash. Preserve code indentation and math syntax.

## Supervised fine-tuning — `schemas/sft.schema.json`

Use a `messages` array with `system`, `user`, `assistant` (and only intentionally used `tool`) roles. Supported task types include conversation, instruction, math_solution, code_generation, debugging, architecture_design and research_synthesis.

Example:
```json
{"schema_version":"1.0.0","record_id":"numina:000001","messages":[{"role":"user","content":"Solve the problem ..."},{"role":"assistant","content":"Verified solution ..."}],"task_type":"math_solution","domain_tags":["mathematics.algebra"],"quality_checks":{"human_reviewed":true,"factuality_pass":true,"tests_passed":null,"safety_reviewed":true,"notes":["Checked independently"]},"synthetic":false,"source_metadata":{"source":"olympiads"},"provenance":{"source_dataset":"AI-MO/NuminaMath-CoT","source_revision":"PINNED_COMMIT","source_config":null,"source_split":"train","source_record_id":"row-id","declared_license":"Apache-2.0","underlying_license":"review-required","usage_status":"conditional","content_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","pipeline_version":"git-sha","transformation_version":"1.0.0"}}
```

For OASST1, reconstruct complete root-to-leaf thread paths using `message_tree_id`, `parent_id`, and `message_id`; never split sibling messages from one tree across partitions. For Numina, map problem to user and validated solution to assistant; retain source category and check every answer.

## Preference — `schemas/preference.schema.json`

Store the same prompt with a preferred and rejected response. Preserve five independent HelpSteer2 dimensions: helpfulness, correctness, coherence, complexity and verbosity (each 0–4). Do not trade correctness away for verbosity. Pair only responses to the same normalized prompt; keep ambiguous comparisons out of DPO labels.

Example:
```json
{"schema_version":"1.0.0","record_id":"helpsteer2:pair-001","prompt_messages":[{"role":"user","content":"Explain a cache-coherence protocol."}],"chosen":"A technically correct explanation ...","rejected":"An incomplete explanation ...","scores":{"chosen":{"helpfulness":4,"correctness":4,"coherence":4,"complexity":3,"verbosity":3},"rejected":{"helpfulness":2,"correctness":1,"coherence":3,"complexity":2,"verbosity":1}},"preference_reason":"Chosen response is more accurate and useful.","domain_tags":["computer_architecture.cache_coherence"],"provenance":{"source_dataset":"nvidia/HelpSteer2","source_revision":"PINNED_COMMIT","source_config":"preference","source_split":"train","source_record_id":"prompt-group-id","declared_license":"CC-BY-4.0","underlying_license":"CC-BY-4.0","usage_status":"approved_with_attribution","content_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","pipeline_version":"git-sha","transformation_version":"1.0.0"}}
```

The official HelpSteer2 validation split is for validation, never preference training. Group by prompt before splitting.

## Evaluation — `schemas/evaluation.schema.json`

Evaluation items contain a sealed prompt, reference answer, weighted rubric, category and `split: test`. Coding tasks should include executable hidden tests or expected artifacts; STEM tasks need units, assumptions and independently checked calculations; research tasks need evidence and citation criteria.

Example:
```json
{"schema_version":"1.0.0","task_id":"medha-chip-001","category":"semiconductors","prompt_messages":[{"role":"user","content":"Explain the role of a MOS gate dielectric and one reliability failure mode."}],"reference_answer":"Verified reference answer...","rubric":[{"criterion":"Physical accuracy","weight":0.7,"description":"Correctly explains field coupling and isolation."},{"criterion":"Reliability","weight":0.3,"description":"Identifies a real failure mechanism."}],"expected_artifacts":[],"hidden_tests":[],"difficulty":"foundation","split":"test","provenance":{"source_dataset":"medha-curated-eval","source_revision":"git-sha","source_split":"test","source_record_id":"task-001","declared_license":"internal-original","underlying_license":"internal-original","usage_status":"approved","content_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","pipeline_version":"git-sha","transformation_version":"1.0.0"}}
```

## Provenance required in every record

Keep dataset ID, immutable revision, config/subset, split, source row/document ID, source URL/path, declared and underlying licence, usage status, content SHA-256, transform/pipeline version, acquisition time, selection seed and attribution note. Unknown licences must be marked `unknown` and blocked from training until reviewed.

## Data separation

- Pretraining corpus: no evaluation prompt/reference content.
- SFT: train/validation/test by thread or problem-family group.
- Preference: chosen/rejected pairs from same prompt; official validation stays out of training.
- Evaluation: sealed test set only; never expose reference answers or hidden tests to training jobs.
