# Medha — local LLM from scratch

Medha is a compact, self-contained language-model project. It uses no hosted AI APIs and no pretrained weights. The model is a decoder-only Transformer trained from random initialization; its learned parameters are stored locally in `checkpoints/medha.pt`.

## What this is (and is not)

This project implements the same *kind* of next-token prediction used by modern chat models, including causal self-attention, residual connections and an MLP. Its character tokenizer and weights are entirely Medha's.

A ChatGPT-sized model is not feasible to train on a typical laptop: it needs billions of parameters, massive licensed data, and many GPU-days. This starter trains a small model so you can learn, verify, and extend the full pipeline yourself.

## Quick start

Requires Python 3.10+ and PyTorch.

```powershell
cd outputs/medha-local-llm
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python train.py --steps 2500
python app.py
```

Then open `http://127.0.0.1:8000` in a browser. For a better model, replace or enlarge `data/training.txt` with text you are allowed to train on, then train for more steps. A CUDA GPU is used automatically when PyTorch supports it.

## AWS deployment

After training, build a container which includes the locally generated `checkpoints/medha.pt`:

```powershell
docker build -t medha-ai .
docker run --rm -p 8000:8000 medha-ai
```

Push that image to an Amazon ECR repository, then create an **AWS App Runner** service from the ECR image. Set its service port to `8000`. App Runner gives you a public HTTPS URL to share. Do not commit trained `.pt` files to GitHub; they may become large. Use ECR/S3 or a release asset for the trained checkpoint instead.

## Project map

- `model.py`: the transformer, tokenizer, sampling, and checkpoint format
- `train.py`: train from random initialization and save Medha's parameters
- `app.py`: local FastAPI chat server; no cloud inference calls
- `web/`: browser chat interface
- `data/training.txt`: small starter corpus you can lawfully replace

## Responsible data

Only train on text you own, text with a compatible licence, or material you have permission to use. Do not put private conversations, passwords, or sensitive personal data in the training corpus.
