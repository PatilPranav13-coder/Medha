# Medha model card

## Production brain

- **Provider family:** local OpenAI-compatible chat server, intended for Ollama.
- **Model, licence, source URL, quantisation:** **PENDING OWNER RUN**. The owner will select the exact model and quantisation after running `python -m evals.capability --brain openai_compat --model <tag> --out reports/<model>/` and reviewing licence terms on the model's source page.
- **Weights:** supplied and installed by the operator; the application does not download model weights.
- **Intended use:** research and coding assistance for students and engineers, with rule-routed web search and numbered source links when available.
- **Measured results:** **PENDING OWNER RUN**. Run the capability suite and serving benchmark on the selected production brain and target hardware; this repository run uses deterministic fixtures only.
- **Known limits:** no production-model quality or ChatGPT Nov 2022 parity result has been measured. Live provider behavior has not been verified from this build environment. Search can fail, miss sources, or return incomplete pages; the extractive path may be less fluent. Local models may be slow or unavailable on undersized hardware.

## Experimental from-scratch model

- **Architecture:** compact decoder-only Transformer with byte-level BPE, RoPE, RMSNorm, SwiGLU and grouped-query attention.
- **Weights and licence:** no launch checkpoint is supplied; no production licence applies to the architecture implementation itself beyond the repository's project terms.
- **Intended use:** education and model-training experiments only.
- **Known limits:** the previously described 2.7M-parameter toy trained on nine conversations is not suitable as the production assistant. It is not retrained or presented as a capable general-purpose model.
