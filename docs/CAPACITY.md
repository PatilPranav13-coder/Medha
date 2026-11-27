# Capacity planning

Medha launches with one application worker because SQLite WAL and in-memory token buckets/generation queues are process-local. The owner selects and installs the local model; no weights are bundled.

## Estimate memory before deployment

Use the exact quantized artifact and runtime that will serve beta traffic. A first-order weight estimate is:

`weight bytes ≈ parameter count × quantization bits ÷ 8`

Add runtime buffers, tokenizer/context memory, the operating system, and a key/value cache. For decoder models, a rough KV-cache estimate per active sequence is:

`2 × layers × context tokens × KV heads × head dimension × bytes per value`

Multiply the KV estimate by the number of simultaneously active sequences. Grouped-query attention uses the KV-head count rather than the query-head count. Treat these as estimates; allocator overhead, quantization format and runtime implementation change the measured total.

## Measure the target host

1. Install the chosen model and runtime on the actual staging host.
2. Set `MEDHA_BRAIN_PROVIDER=openai_compat`, `MEDHA_BRAIN_BASE_URL`, `MEDHA_BRAIN_MODEL`, and the API key only if the local service requires one.
3. Warm the service, then run `python scripts/bench_serving.py --model <tag> --out reports/serving.json`.
4. Record peak host RAM/VRAM, time to first token, tokens per second, errors and queue wait at concurrency 1, 2 and 4. Repeat with the longest expected context.
5. Choose `MEDHA_MAX_CONCURRENT_GENERATIONS`, `MEDHA_MAX_QUEUED_GENERATIONS`, queue timeout, output length and user cap from those observations. The beta defaults to one active generation and four waiting requests.

The capability and serving runs are `PENDING OWNER RUN` until performed on the selected hardware. No throughput or host-fit measurement is claimed by this repository build.
