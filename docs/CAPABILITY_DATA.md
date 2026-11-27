# Capability evaluation data

`evals/capability/bank.py` contains 210 original tasks authored for Medha. The bank is not copied from a public benchmark. It spans knowledge/reasoning, code, format following, multi-turn retention, text transformation and refusal behavior.

The JSONL loader accepts a locally downloaded public dataset only when the caller supplies its licence note. Record the exact source URL, revision, licence text, attribution requirements and any non-commercial/share-alike terms next to the downloaded copy. The loader does not download data and does not grant permission to use it.

The task thresholds in `evals/capability/bar.json` are starter bars. The owner may adjust them after reviewing the first actual model run. Only reports from the selected production model on target hardware can support a capability claim.
