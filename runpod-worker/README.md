# Face LoRA RunPod Worker

This worker runs inside the RunPod persistent Pod on port `4000`.

It provides:

- `GET /health` for Cloudflare and manual checks
- `POST /jobs/dataset` downloads raw images, asks OpenRouter Gemini for face bbox + caption, crops fixed-size face images, and uploads processed faces back to R2
- `POST /jobs/train` downloads dataset images, writes captions, generates an AI Toolkit YAML config, runs `python run.py`, and installs the resulting `.safetensors` into ComfyUI
- `POST /jobs/generate` placeholder for ComfyUI generation
- `GET /jobs/{job_id}` for job status
- `POST /models/z-image/prepare` downloads and pins Z-Image Base into the RunPod workspace
- `GET /models/z-image/status` checks whether the local Z-Image Base model is ready

The service is intentionally separate from ComfyUI. Cloudflare should call this worker, and this worker should call local ComfyUI at `http://127.0.0.1:8188`.

Training defaults are set for Z-Image Base:

- Architecture: `z-image`
- Base model repo: `Tongyi-MAI/Z-Image`
- Rank: `32`
- Learning rate: `1e-4`
- Resolution: `1024`
- Steps: `3000`
- Preview/generation baseline: `40` steps, CFG `5`

## Run on Pod

```bash
cd /workspace
git clone https://github.com/xellse/face-lora-studio.git /workspace/face-lora-studio
cd /workspace/face-lora-studio/runpod-worker
python -m pip install -r requirements.txt

export RUNPOD_WORKSPACE=/workspace
export COMFY_BASE_URL=http://127.0.0.1:8188
export WORKER_TOKEN="change-this-to-a-long-random-value"
export HF_TOKEN="your-huggingface-token-if-required"
export MODEL_CACHE_DIR=/workspace/models
export Z_IMAGE_MODEL_DIR=/workspace/models/z-image-base
export Z_IMAGE_ARCH=zimage
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENROUTER_MODEL="google/gemini-3-flash-preview"
export R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"
export R2_BUCKET="face-lora-assets"
export R2_ACCESS_KEY_ID="your-r2-access-key-id"
export R2_SECRET_ACCESS_KEY="your-r2-secret-access-key"
export PUBLIC_STORAGE_BASE_URL="https://img.xellsun.com"

tmux new -s face-worker
python -m uvicorn app.main:app --host 0.0.0.0 --port 4000
```

Detach from tmux with `Ctrl+B`, then `D`.

Check locally in the Pod:

```bash
curl http://127.0.0.1:4000/health
```

## Prepare Z-Image Base for AI Toolkit

Before starting LoRA training, deploy the base model into the persistent workspace:

```bash
curl -X POST http://127.0.0.1:4000/models/z-image/prepare \
  -H "X-Worker-Token: $WORKER_TOKEN"
```

The response contains a RunPod worker job id. Poll it until completed:

```bash
curl http://127.0.0.1:4000/jobs/<job_id> \
  -H "X-Worker-Token: $WORKER_TOKEN"
```

Check the final local model status:

```bash
curl http://127.0.0.1:4000/models/z-image/status \
  -H "X-Worker-Token: $WORKER_TOKEN"
```

The training config will reference the local path, usually `/workspace/models/z-image-base`, as AI Toolkit's `model.name_or_path`.

Check through RunPod proxy:

```txt
https://<POD_ID>-4000.proxy.runpod.net/health
```

## Training Output

For a training job `rp_xxx`, the worker writes:

- Dataset images and captions: `/workspace/jobs/rp_xxx/dataset/images`
- AI Toolkit config: `/workspace/jobs/rp_xxx/z_image_base_ai_toolkit_config.yaml`
- AI Toolkit output: `/workspace/jobs/rp_xxx/ai_toolkit_output`
- Installed LoRA: `/workspace/ComfyUI/models/loras/local-user/{loraId}.safetensors`

Dataset processing writes:

- Raw downloaded images: `/workspace/jobs/rp_xxx/raw`
- Cropped face images and captions: `/workspace/jobs/rp_xxx/faces`
- R2 public images: `https://img.xellsun.com/datasets/{datasetId}/faces/{faceId}.jpg`
