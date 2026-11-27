# Medha's from-scratch research model

This folder contains an experimental decoder-only Transformer trained from random initialization. Its byte-level BPE tokenizer, RoPE, RMSNorm, SwiGLU, grouped-query attention, KV cache, and versioned checkpoint format are learning and research code. The small model is not Medha's production assistant brain and is not retrained as part of the public-beta work.

## Development

Use Python 3.11+ and install `requirements.txt` from the repository root. To run this folder's tests exactly as CI does:

```powershell
cd medha_local_llm
python -m unittest discover -s tests
```

Train only when intentionally working on the research track:

```powershell
cd ..
python -m medha_local_llm.train --steps 2500
```

The development server loads a checkpoint from `checkpoints/medha.pt` (or `MEDHA_EXPERIMENTAL_CHECKPOINT`) and serves the small demo UI:

```powershell
cd ..
python -m medha_local_llm.app
```

The launch backend uses `medha_local_llm` as a library, but production chat should use the operator-selected local open-weights instruct model through the configured brain adapter.

Only train on text you own, text with a compatible licence, or material you have permission to use. Never commit `.pt` weights.
