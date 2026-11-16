# Medha: 100-million-example software-engineering programme

## Objective

Build Medha into a capable software-engineering assistant that can understand, design, implement, test, debug, review, secure, optimise, and maintain software across Python, JavaScript, TypeScript, Java, C++, Swift, and multi-language full-stack repositories.

The approved planning target is **100,000,000 unique engineering examples**. This is a target, not a statement that the corpus has been collected. The current data factory is still an early prototype: it can inspect approved repositories, create pretraining records and task candidates, and promote reviewed work under gates. The plan and counters do not download data, create model weights, or train Medha.

## What counts as one example

A raw source file is not automatically one useful supervised example. The target unit should normally contain:

- a clear engineering task or code-understanding objective;
- enough code, repository, issue, API, or system context to solve it;
- a reviewed answer, reference implementation, patch, or engineering decision;
- relevant tests or an explicit rationale for non-executable review;
- actual build/test evidence when execution is required;
- language, framework, task, difficulty, and project-scope labels;
- source URL, pinned revision, licence evidence, content hashes, and transformation provenance.

Pretraining code/text records remain a separate data product. Do not inflate the 100M supervised-example counter by counting each file, chunk, generated paraphrase, or duplicate as an independent engineering task.

## Target allocation

### Primary language mix

| Primary bucket | Target | Share |
|---|---:|---:|
| Python | 20,000,000 | 20% |
| JavaScript | 15,000,000 | 15% |
| TypeScript | 15,000,000 | 15% |
| Java | 12,000,000 | 12% |
| C++ | 12,000,000 | 12% |
| Swift | 8,000,000 | 8% |
| Multi-language projects | 18,000,000 | 18% |
| **Total** | **100,000,000** | **100%** |

Each example gets one primary bucket for accounting, even when it contains several languages.

### Task mix

| Engineering task family | Target | Share |
|---|---:|---:|
| Implementation and feature delivery | 18,000,000 | 18% |
| Debugging and root-cause analysis | 16,000,000 | 16% |
| Testing and validation | 12,000,000 | 12% |
| Repository comprehension | 10,000,000 | 10% |
| Refactoring and maintainability | 8,000,000 | 8% |
| Backend, APIs, and databases | 8,000,000 | 8% |
| Frontend, mobile, and accessibility | 8,000,000 | 8% |
| Architecture and system design | 7,000,000 | 7% |
| Performance, security, and reliability | 7,000,000 | 7% |
| DevOps, delivery, and observability | 4,000,000 | 4% |
| Documentation and maintenance | 2,000,000 | 2% |
| **Total** | **100,000,000** | **100%** |

### Full-stack coverage

At least **20,000,000 examples** should be tagged as full-stack or cross-layer work. This is a cross-cutting minimum, not an additional bucket: full-stack examples also count once in the primary-language and task-family dimensions.

### Split policy

- Train: 90,000,000
- Validation: 5,000,000
- Sealed test/evaluation: 5,000,000

Split by repository, issue lineage, benchmark family, and near-duplicate cluster before deriving chunks or tasks. Keep test prompts, reference answers, patches, and hidden tests out of training.

## What Medha must learn

The companion software-engineering curriculum defines eight progressive levels:

1. Programming foundations in the six requested languages.
2. Testing, types, debugging, code quality, and review.
3. Repository navigation and issue-level engineering.
4. Frontend, backend, APIs, databases, and full-stack delivery.
5. Swift/iOS, Android, and cross-platform mobile work.
6. Performance, concurrency, security, and reliability.
7. CI/CD, containers, deployment, and observability.
8. Architecture, migrations, technical trade-offs, and technical leadership.

Every task should reinforce the engineering loop: understand requirements, inspect context, plan, make a minimal change, add tests, run checks, investigate failures, iterate, and report exactly what was verified.

## Quality gate

The SFT promotion threshold is **85/100**. The scorer measures source rights, provenance completeness, human review, factuality review, safety review, isolated verification, build/tests (or explicit non-executable evidence), and task/answer specificity.

A score cannot override a failed hard gate. A task is rejected when rights are held or unknown, provenance is incomplete, human/factuality/safety review is missing, an executable solution lacks passing isolated verification, or the prompt/answer is too weak. A model-generated answer is not ground truth merely because it is fluent.

Synthetic examples are capped at 15% of the target, must be labelled and traceable, and need independent review and verification. Never generate evaluation examples from training tasks or leak benchmark answers into the training mixture.

## Scaling architecture

The present factory is single-repository and bounded. Reaching 100M requires a separate distributed collection and training system:

1. **Source registry:** licence and terms review, per-repository/file clearance, source revision pinning, attribution, takedown/update tracking.
2. **Discovery queue:** metadata-only discovery; prioritise requested language, task, framework, and project gaps.
3. **Partitioned workers:** idempotent jobs by source/revision/language; bounded fetches; retries; checkpoints; dead-letter queue; immutable run IDs.
4. **Processing:** secret and sensitive-file scanning, binary/dependency exclusion, language/framework detection, normalisation, exact and near-duplicate detection, repository lineage grouping.
5. **Task construction:** derive candidate tasks from code, issues, tests, docs, commits, and API contracts; candidates are not training records until reviewed.
6. **Verification:** disposable no-network execution, pinned dependencies, resource/time limits, test/build logs, patch application and regression checks.
7. **Quality and rights gate:** only eligible records enter the training pool; keep rejected records and reasons for audit without retaining disallowed content.
8. **Sharded storage:** immutable compressed shards (target 100,000 examples per shard), manifests, hashes, source lineage, split assignment, and reproducible dataset versions.
9. **Training:** separate pretraining, SFT, preference optimisation, and sealed evaluation; run controlled ablations and holdout tests before scaling model size.
10. **Monitoring:** quality, pass rates, language/task coverage, duplicates, licence status, contamination, storage, tokens, and cost per accepted example.

Do not push a giant raw corpus into Git. Keep code, schemas, manifests, and small test fixtures in the repository; store approved large shards in versioned object storage or an approved dataset registry.

## Milestones

| Stage | Cumulative accepted examples | Exit condition |
|---|---:|---|
| Proof of pipeline | 10,000 | quality scoring, provenance, split isolation, and tests work end to end |
| Repeatable collection | 100,000 | multiple approved sources and reproducible runs |
| Multi-source scale | 1,000,000 | all target languages represented; quality and licence audits |
| Distributed collection | 10,000,000 | partitioned workers, checkpointing, sharded storage, cost/quality dashboards |
| Full target | 100,000,000 | coverage quotas met, quality and legal audits passed, sealed evaluations preserved |

Do not skip a milestone merely because a raw row counter reaches its number. The counter should include only unique examples that satisfy the programme's rights and quality rules.

## Current state

At the time this plan was added, the repository contained the prototype data factory and held source templates; it did **not** contain 100M examples. No external repository corpus has been collected by this planning change, and Medha's model weights have not been updated. The next practical milestone is to validate 10,000 accepted examples end to end, then measure real throughput, storage, review effort, and training costs before raising collection volume.
