# Medha — local LLM from scratch

Medha is a self-contained language-model project. It uses no hosted AI APIs and no pretrained weights. Its decoder-only Transformer starts with random weights and trains locally; inference runs inside the app container.

## What this can do

The current model is a learning prototype, not a ChatGPT-sized assistant. It uses a character tokenizer and a small starter corpus (about 1,100 characters). It will not reliably answer broad questions or generate useful code until you provide much more training data and compute. The chat app keeps recent conversation context, but that does not make the underlying model more capable.

Only train on data you own or have permission to use. Avoid secrets and private conversations.

## Train locally

Requires Python 3.10+ and PyTorch.

```powershell
cd medha_local_llm
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --steps 2500
```

This writes `checkpoints/medha.pt`. The checkpoint is intentionally not committed to GitHub.

## Run the app locally

From the repository root:

```powershell
pip install -r backend/requirements.txt
$env:MEDHA_CHECKPOINT = "medha_local_llm/checkpoints/medha.pt"
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Deploy the integrated chat app to AWS App Runner

You need an AWS account, AWS CLI credentials, Docker, and permission to create ECR and App Runner resources. App Runner hosts the container image from ECR and serves it on port 8000. The app does not call an external AI API.

First train the model and build the container from the repository root:

```powershell
cd medha_local_llm
python train.py --steps 2500
cd ..
docker build -t medha-ai .
docker run --rm -p 8000:8000 medha-ai
```

Verify `http://localhost:8000` opens before publishing the image. Then create an ECR repository and push the image. Replace the region placeholder with your AWS region:

```powershell
$Region = "YOUR_AWS_REGION"
$AccountId = aws sts get-caller-identity --query Account --output text
$Registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
aws ecr create-repository --repository-name medha --region $Region
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry
docker tag medha-ai "$Registry/medha:latest"
docker push "$Registry/medha:latest"
```

In the AWS Console, open **App Runner → Create service**, choose **Container registry → Amazon ECR**, select `medha:latest`, and set the service port to `8000`. For a private ECR repository, configure the ECR access role requested by App Runner. Choose an instance size with enough memory for PyTorch (start with at least 2 GB), review the service and expected charges, then create it. App Runner will provide a public HTTPS URL when the service is running.

To publish a newly trained model, build and push a new image containing the new checkpoint, then redeploy the App Runner service. Keep the checkpoint private; do not commit it to GitHub.

## Project map

- `model.py`: transformer, tokenizer, sampling, checkpoint format
- `train.py`: training from random initialization
- `backend/main.py`: local FastAPI inference API
- `frontend/`: browser chat interface
- `data/training.txt`: starter corpus you can replace with permitted training material
- `Dockerfile`: deployable image for the integrated app
