# Face LoRA RunPod Worker

This worker runs inside the RunPod persistent Pod on port `4000`.

For the first deployment it provides:

- `GET /health` for Cloudflare and manual checks
- `POST /jobs/dataset` placeholder for face crop + caption
- `POST /jobs/train` placeholder for AI Toolkit training
- `POST /jobs/generate` placeholder for ComfyUI generation
- `GET /jobs/{job_id}` for job status

The service is intentionally separate from ComfyUI. Cloudflare should call this worker, and this worker should call local ComfyUI at `http://127.0.0.1:8188`.

## Run on Pod

```bash
cd /workspace
git clone https://github.com/xellse/face-lora-studio.git /workspace/face-lora-studio
cd /workspace/face-lora-studio/runpod-worker
python -m pip install -r requirements.txt

export RUNPOD_WORKSPACE=/workspace
export COMFY_BASE_URL=http://127.0.0.1:8188
export WORKER_TOKEN="change-this-to-a-long-random-value"

tmux new -s face-worker
python -m uvicorn app.main:app --host 0.0.0.0 --port 4000
```

Detach from tmux with `Ctrl+B`, then `D`.

Check locally in the Pod:

```bash
curl http://127.0.0.1:4000/health
```

Check through RunPod proxy:

```txt
https://<POD_ID>-4000.proxy.runpod.net/health
```

