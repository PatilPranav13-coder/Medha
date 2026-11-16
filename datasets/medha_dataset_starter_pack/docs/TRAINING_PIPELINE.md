# Medha data preparation pipeline

## 1. Inspect and sample
- Never use non-streaming loads during exploration.
- Inspect all dataset cards, configs, split metadata and exact source revisions.
- Use streaming and a strict row cap (25–100 initially; 1,000 maximum for this exploration script).
- Do not set `trust_remote_code=True` automatically. Proof-Pile-2's viewer warns that the loader executes arbitrary Python code.
- The Stack v2 stays blocked pending access, SWH/INRIA conditions and file-level licence rules.

## 2. Filter
- All text: empty/short rows, extraction boilerplate, malformed encoding, spam, low language confidence, PII and junk.
- Code: secrets, credentials, malware, binaries, generated/minified/vendor files and licences outside an approved allow-list. Retain path, repo, commit and licence provenance.
- Math/STEM: invalid OCR/LaTeX, incomplete questions, incorrect solutions, missing units/assumptions; use symbolic/numeric checks and human review.
- Conversation: deleted/spam/PII labels, invalid role sequences, broken parent links, unsafe material, low review quality.
- Preference: same normalized prompt only; keep five HelpSteer2 dimensions; avoid ambiguous pairs.

## 3. Deduplicate globally
Normalize Unicode NFC and line endings. Exact dedupe with SHA-256. Near-dedupe with MinHash/SimHash or embeddings at document/paragraph/file level. Preserve code indentation and math. Group cross-dataset duplicates, especially FineWeb with FineWeb-Edu. Keep one canonical copy plus a duplicate audit ledger containing every source and licence. Evaluation wins: remove any duplicated training copy rather than leaking an eval task.

## 4. Build separate outputs
- Pretraining: text/code/math documents; use pretraining schema.
- SFT: OASST1 full conversation paths and checked NuminaMath problem/solution pairs.
- Preference: HelpSteer2 same-prompt response comparisons; preserve all scores.
- Evaluation: sealed tasks with references/rubrics and hidden coding tests; independently curated or separately licensed.

## 5. Split policy
- Pretraining: optional 98/2 document-level holdout for exploratory perplexity only; it is not a capability benchmark.
- SFT: 90/5/5 by conversation tree or problem-family group; do not split sibling records.
- Preference: retain official HelpSteer2 train and validation partitions; validation never enters training; group by prompt.
- Evaluation: sealed test only; no prompt/reference/test code in pretraining, SFT or preference data.

## 6. Quality gates
Every record validates against schema and has provenance/hash. Hold/unreviewed licences stay out. Code passes syntax and secret scans; executable examples run sandboxed. Math is independently verified. Preference labels have a real margin and no correctness regression. Evaluation leakage scan and dedupe report must pass. Log row counts, rejection reasons, source/licence mix, languages, domains, duplicate ratios and reviewer sign-off.
