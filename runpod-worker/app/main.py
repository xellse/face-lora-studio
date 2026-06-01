import json
import os
import re
import shutil
import subprocess
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field


WORKSPACE = Path(os.environ.get("RUNPOD_WORKSPACE", "/workspace"))
JOBS_DIR = WORKSPACE / "jobs"
COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
Z_IMAGE_BASE_MODEL = "z-image-base"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image"
AI_TOOLKIT_DIR = WORKSPACE / "ai-toolkit"

APP_VERSION = "0.1.0"

app = FastAPI(title="Face LoRA RunPod Worker", version=APP_VERSION)


class DatasetJobRequest(BaseModel):
    dataset_id: str = Field(alias="datasetId")
    user_id: str = Field(default="local-user", alias="userId")
    raw_keys: list[str] = Field(default_factory=list, alias="rawKeys")
    trigger_word: str = Field(default="person_lora", alias="triggerWord")
    crop_size: int = Field(default=1024, alias="cropSize")


class TrainingJobRequest(BaseModel):
    dataset_id: str = Field(alias="datasetId")
    lora_id: str = Field(alias="loraId")
    user_id: str = Field(default="local-user", alias="userId")
    lora_name: str = Field(alias="loraName")
    trigger_word: str = Field(alias="triggerWord")
    base_model: str = Field(default=Z_IMAGE_BASE_MODEL, alias="baseModel")
    parameters: dict[str, Any] = Field(default_factory=dict)
    dataset_images: list["TrainingImage"] = Field(default_factory=list, alias="datasetImages")


class TrainingImage(BaseModel):
    id: str
    index: int = 0
    url: str
    caption: str
    crop_size: int | None = Field(default=None, alias="cropSize")


class GenerationJobRequest(BaseModel):
    task_id: str = Field(alias="taskId")
    user_id: str = Field(default="local-user", alias="userId")
    lora_id: str = Field(alias="loraId")
    prompt: str
    negative_prompt: str = Field(default="", alias="negativePrompt")
    settings: dict[str, Any] = Field(default_factory=dict)


def require_token(x_worker_token: str | None = Header(default=None)):
    if WORKER_TOKEN and x_worker_token != WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid worker token")


@app.on_event("startup")
def startup():
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    comfy = check_comfy()
    return {
        "ok": True,
        "version": APP_VERSION,
        "defaultModel": {
            "id": Z_IMAGE_BASE_MODEL,
            "architecture": "z-image",
            "repo": Z_IMAGE_REPO,
        },
        "workspace": str(WORKSPACE),
        "paths": {
            "jobs": str(JOBS_DIR),
            "comfy": str(WORKSPACE / "ComfyUI"),
            "aiToolkit": str(WORKSPACE / "ai-toolkit"),
            "loras": str(WORKSPACE / "ComfyUI/models/loras/local-user"),
        },
        "comfy": comfy,
    }


@app.post("/jobs/dataset", dependencies=[Depends(require_token)])
def create_dataset_job(payload: DatasetJobRequest, background_tasks: BackgroundTasks):
    job = create_job("dataset_processing", payload.model_dump(by_alias=True))
    background_tasks.add_task(simulate_dataset_processing, job["id"])
    return job


@app.post("/jobs/train", dependencies=[Depends(require_token)])
def create_training_job(payload: TrainingJobRequest, background_tasks: BackgroundTasks):
    job = create_job("lora_training", payload.model_dump(by_alias=True))
    background_tasks.add_task(simulate_training, job["id"], payload)
    return job


@app.post("/jobs/generate", dependencies=[Depends(require_token)])
def create_generation_job(payload: GenerationJobRequest, background_tasks: BackgroundTasks):
    job = create_job("generation", payload.model_dump(by_alias=True))
    background_tasks.add_task(simulate_generation, job["id"])
    return job


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str):
    job = read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/debug/request")
async def debug_request(request: Request):
    return {
        "url": str(request.url),
        "headers": {
            "host": request.headers.get("host"),
            "x-forwarded-for": request.headers.get("x-forwarded-for"),
            "x-forwarded-proto": request.headers.get("x-forwarded-proto"),
        },
    }


def check_comfy() -> dict[str, Any]:
    try:
        response = requests.get(f"{COMFY_BASE_URL}/system_stats", timeout=3)
        return {
            "baseUrl": COMFY_BASE_URL,
            "reachable": response.ok,
            "statusCode": response.status_code,
        }
    except Exception as exc:
        return {
            "baseUrl": COMFY_BASE_URL,
            "reachable": False,
            "error": str(exc),
        }


def create_job(job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = {
        "id": f"rp_{uuid.uuid4().hex[:10]}",
        "type": job_type,
        "status": "queued",
        "progress": 0,
        "message": "Queued on RunPod worker",
        "payload": payload,
        "result": None,
        "logs": [],
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
    }
    write_job(job)
    return job


def simulate_dataset_processing(job_id: str):
    update_job(job_id, 20, "running", "Preparing raw photos")
    time.sleep(1)
    update_job(job_id, 55, "running", "Face crop/caption step placeholder")
    time.sleep(1)
    update_job(job_id, 100, "completed", "Dataset processing placeholder completed", {
        "note": "Replace this step with face detection, crop, caption, and R2 upload.",
    })


def simulate_training(job_id: str, payload: TrainingJobRequest):
    try:
        if not payload.dataset_images:
            raise RuntimeError("No datasetImages were provided for training")

        update_job(job_id, 10, "running", "Downloading training images and captions")
        dataset_dir = prepare_training_dataset(job_id, payload)

        update_job(job_id, 18, "running", "Writing Z-Image Base AI Toolkit config")
        config_path = write_z_image_training_config(job_id, payload, dataset_dir)

        update_job(job_id, 25, "running", "Starting AI Toolkit training", {
            "aiToolkitConfig": str(config_path),
            "datasetDir": str(dataset_dir),
        })
        output_dir = run_ai_toolkit(job_id, config_path)

        update_job(job_id, 96, "running", "Copying trained LoRA into ComfyUI")
        model_path = install_trained_lora(payload, output_dir)
        update_job(job_id, 100, "completed", "Z-Image Base LoRA training completed", {
            "modelPath": str(model_path),
            "baseModel": Z_IMAGE_BASE_MODEL,
            "modelRepo": Z_IMAGE_REPO,
            "aiToolkitOutput": str(output_dir),
        })
    except Exception as exc:
        update_job(job_id, 100, "failed", f"Training failed: {exc}", {
            "error": str(exc),
            "baseModel": Z_IMAGE_BASE_MODEL,
            "modelRepo": Z_IMAGE_REPO,
        })


def prepare_training_dataset(job_id: str, payload: TrainingJobRequest) -> Path:
    job_dir = JOBS_DIR / job_id
    dataset_dir = job_dir / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(payload.dataset_images, start=1):
        image_path = images_dir / f"{index:04d}.jpg"
        caption_path = images_dir / f"{index:04d}.txt"
        download_training_image(image.url, image_path, image.caption)
        caption_path.write_text(ensure_trigger_word(image.caption, payload.trigger_word))

    return images_dir


def download_training_image(url: str, image_path: Path, caption: str):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        image = Image.new("RGB", (1024, 1024), color=(38, 98, 108))
        draw = ImageDraw.Draw(image)
        draw.ellipse((332, 210, 692, 570), fill=(245, 245, 245))
        draw.rounded_rectangle((300, 650, 724, 880), radius=90, fill=(245, 245, 245))
        draw.text((48, 940), f"placeholder for unsupported {content_type}", fill=(255, 255, 255))
        draw.text((48, 970), caption[:80], fill=(255, 255, 255))
    image.save(image_path, "JPEG", quality=95)


def ensure_trigger_word(caption: str, trigger_word: str) -> str:
    caption = (caption or "").strip()
    if trigger_word and trigger_word not in caption:
        return f"{trigger_word}, {caption}" if caption else trigger_word
    return caption


def write_z_image_training_config(job_id: str, payload: TrainingJobRequest, images_dir: Path) -> Path:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    parameters = z_image_training_parameters(payload.parameters)
    config_path = job_dir / "z_image_base_ai_toolkit_config.yaml"
    output_dir = job_dir / "ai_toolkit_output"
    safe_name = safe_job_name(payload.lora_id)
    config_path.write_text(f"""---
job: extension
config:
  name: "{safe_name}"
  process:
    - type: "sd_trainer"
      training_folder: "{output_dir}"
      device: cuda:0
      trigger_word: "{yaml_escape(payload.trigger_word)}"
      network:
        type: "lora"
        linear: {parameters["rank"]}
        linear_alpha: {parameters["rank"]}
      save:
        dtype: float16
        save_every: {parameters["saveEvery"]}
        max_step_saves_to_keep: 4
        push_to_hub: false
      datasets:
        - folder_path: "{images_dir}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ {parameters["resolution"]} ]
      train:
        batch_size: 1
        cache_text_embeddings: true
        steps: {parameters["steps"]}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: {parameters["learningRate"]}
        dtype: bf16
      model:
        name_or_path: "{Z_IMAGE_REPO}"
        arch: "z_image"
        quantize: true
        low_vram: false
      sample:
        sampler: "flowmatch"
        sample_every: {parameters["saveEvery"]}
        width: {parameters["resolution"]}
        height: {parameters["resolution"]}
        prompts:
          - "[trigger], close-up portrait, natural light, sharp facial detail"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 5
        sample_steps: 40
meta:
  name: "[name]"
  version: "1.0"
""")
    return config_path


def run_ai_toolkit(job_id: str, config_path: Path) -> Path:
    if not (AI_TOOLKIT_DIR / "run.py").exists():
        raise RuntimeError(f"AI Toolkit run.py not found at {AI_TOOLKIT_DIR}")

    command = ["python", "run.py", str(config_path)]
    process = subprocess.Popen(
        command,
        cwd=AI_TOOLKIT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    progress = 25
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        if not clean:
            continue
        progress = max(progress, infer_training_progress(clean, progress))
        append_job_log(job_id, clean, progress=progress, status="running")

    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"AI Toolkit exited with code {exit_code}")

    job = read_job(job_id)
    config_payload = job.get("result") if job else None
    output_hint = None
    if isinstance(config_payload, dict):
        output_hint = config_payload.get("aiToolkitConfig")
    return config_path.parent / "ai_toolkit_output"


def infer_training_progress(line: str, current: int) -> int:
    match = re.search(r"(\d+)\s*/\s*(\d+)", line)
    if match:
        step = int(match.group(1))
        total = max(1, int(match.group(2)))
        return min(95, max(current, 25 + int((step / total) * 70)))
    if "saving" in line.lower():
        return max(current, 90)
    return current


def append_job_log(job_id: str, message: str, progress: int | None = None, status: str | None = None):
    job = read_job(job_id)
    if not job:
        return
    job["logs"].append({"at": now_iso(), "message": message[-1000:]})
    job["logs"] = job["logs"][-200:]
    if progress is not None:
        job["progress"] = progress
    if status is not None:
        job["status"] = status
    job["message"] = message[-240:]
    job["updatedAt"] = now_iso()
    write_job(job)


def install_trained_lora(payload: TrainingJobRequest, output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No safetensors output found under {output_dir}")
    destination = WORKSPACE / "ComfyUI/models/loras" / payload.user_id / f"{payload.lora_id}.safetensors"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], destination)
    return destination


def z_image_training_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": int(parameters.get("steps", 3000)),
        "learningRate": parameters.get("learningRate", "1e-4"),
        "rank": int(parameters.get("rank", 32)),
        "repeats": int(parameters.get("repeats", 10)),
        "resolution": int(parameters.get("resolution", 1024)),
        "saveEvery": int(parameters.get("saveEvery", 250)),
    }


def safe_job_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or f"job_{uuid.uuid4().hex[:8]}"


def yaml_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def simulate_generation(job_id: str):
    update_job(job_id, 25, "running", "Preparing ComfyUI workflow")
    time.sleep(1)
    update_job(job_id, 65, "running", "ComfyUI generation placeholder")
    time.sleep(1)
    update_job(job_id, 100, "completed", "Generation placeholder completed", {
        "note": "Replace this step with ComfyUI /prompt and output upload.",
    })


def update_job(job_id: str, progress: int, status: str, message: str, result: dict[str, Any] | None = None):
    job = read_job(job_id)
    if not job:
        return
    job["progress"] = progress
    job["status"] = status
    job["message"] = message
    job["updatedAt"] = now_iso()
    job["logs"].append({"at": now_iso(), "message": message})
    if result is not None:
        job["result"] = result
    write_job(job)


def read_job(job_id: str) -> dict[str, Any] | None:
    path = job_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_job(job: dict[str, Any]):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_path(job["id"]).write_text(json.dumps(job, indent=2))


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
