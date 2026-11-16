# Licence and provenance controls

This is an engineering risk register, not legal advice. Re-check terms for the exact revision and intended use (research, commercial training, redistribution and model release can have different implications).

| Source | Declared licence / terms | Medha decision |
|---|---|---|
| FineWeb-Edu | ODC-By 1.0 + CommonCrawl Terms of Use | Conditional. Preserve ODC attribution. ODC-By covers database rights and does not automatically clear independent rights in every web page's content. |
| FineWeb | ODC-By 1.0 + CommonCrawl Terms of Use | Conditional for the same reason. |
| Proof-Pile-2 | Underlying component licences remain unchanged | Component-level approval only; parse meta and hold unknown/restricted components. |
| NuminaMath-CoT | Apache-2.0 declared by card | Conditional; source material includes external PDFs/forums and synthetic examples. Check underlying rights and independently validate solutions. |
| OASST1 | Apache-2.0 | Dataset layer is permissive; still perform PII, safety, spam and privacy filtering. |
| HelpSteer2 | CC-BY-4.0 | Attribution required; keep provider, licence URL and transformation notes. |
| IndicCorpV2 | Card says CC-0 for datasets created by the authors | Conditional until underlying data-source and privacy checks pass. Retain language provenance. |
| The Stack v2 | Hugging Face says `other`; gated terms + original per-repository licences | HOLD. Bulk access requires Software Heritage/INRIA agreement; comply with original code licences, attribution, SWH training principles and ongoing takedown updates. |

## Mandatory provenance

Each canonical record must include dataset repository, exact immutable revision, config/subset, split, source row/file ID, source URL/path, crawl or commit date, declared dataset licence, underlying content/code licence (or `unknown`), raw source metadata, SHA-256 of normalized content, transformation version, filter result, duplicate cluster ID, selection seed, acquisition timestamp and Medha pipeline commit.

Do not store HF tokens, private credentials, contributor email addresses or unnecessary personal data in manifests.

## Release gates

- G0: dataset card, terms, citation and schema inspected.
- G1: bounded sample streamed; no full download.
- G2: rights review assigns approved / conditional / hold / excluded.
- G3: quality, language, PII, toxicity, spam, malware and secrets checks pass.
- G4: immutable revision and complete provenance recorded.
- G5: global dedupe and group-safe split checks pass.
- G6: audit report, attribution and deletion/takedown process approved.

ODC attribution notice: “This database includes material from [dataset], made available under ODC-By 1.0: https://opendatacommons.org/licenses/by/1-0/ . Source content may be subject to additional terms and rights.”

HelpSteer2 attribution: “HelpSteer2 by NVIDIA, licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Modified for Medha by [describe transformation].”
