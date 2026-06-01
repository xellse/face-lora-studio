import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field


WORKSPACE = Path(os.environ.get("RUNPOD_WORKSPACE", "/workspace"))
JOBS_DIR = WORKSPACE / "jobs"
COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
Z_IMAGE_BASE_MODEL = "z-image-base"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image"

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
    update_job(job_id, 15, "running", "Preparing AI Toolkit config")
    time.sleep(1)
    config_path = write_z_image_training_config(job_id, payload)
    update_job(job_id, 45, "running", "Z-Image Base AI Toolkit training placeholder", {
        "aiToolkitConfig": str(config_path),
    })
    time.sleep(1)
    model_path = WORKSPACE / "ComfyUI/models/loras" / payload.user_id / f"{payload.lora_id}.safetensors"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("placeholder safetensors file; replace with AI Toolkit output\n")
    update_job(job_id, 100, "completed", "Training placeholder completed", {
        "modelPath": str(model_path),
        "baseModel": Z_IMAGE_BASE_MODEL,
        "modelRepo": Z_IMAGE_REPO,
    })


def write_z_image_training_config(job_id: str, payload: TrainingJobRequest) -> Path:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    parameters = z_image_training_parameters(payload.parameters)
    config = {
        "job": f"train_{payload.lora_id}",
        "model": {
            "architecture": "z-image",
            "name_or_path": Z_IMAGE_REPO,
            "base_model": Z_IMAGE_BASE_MODEL,
            "low_vram": False,
        },
        "target": {
            "type": "lora",
            "linear_rank": parameters["rank"],
            "linear_alpha": parameters["rank"],
        },
        "save": {
            "dtype": "bf16",
            "save_every": parameters["saveEvery"],
            "max_step_saves_to_keep": 4,
            "output_path": str(WORKSPACE / "ComfyUI/models/loras" / payload.user_id / f"{payload.lora_id}.safetensors"),
        },
        "training": {
            "steps": parameters["steps"],
            "learning_rate": parameters["learningRate"],
            "batch_size": 1,
            "gradient_accumulation": 1,
            "optimizer": "AdamW8Bit",
            "weight_decay": 0.0001,
            "precision": "bf16",
            "timestep_type": "weighted",
            "timestep_bias": "balanced",
            "loss_type": "mse",
        },
        "dataset": {
            "dataset_id": payload.dataset_id,
            "resolution": parameters["resolution"],
            "repeats": parameters["repeats"],
            "caption_ext": "txt",
            "trigger_word": payload.trigger_word,
        },
        "sample": {
            "steps": 40,
            "cfg": 5,
            "lora_scale": 0.85,
        },
    }
    config_path = job_dir / "z_image_base_ai_toolkit_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def z_image_training_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": int(parameters.get("steps", 3000)),
        "learningRate": parameters.get("learningRate", "1e-4"),
        "rank": int(parameters.get("rank", 32)),
        "repeats": int(parameters.get("repeats", 10)),
        "resolution": int(parameters.get("resolution", 1024)),
        "saveEvery": int(parameters.get("saveEvery", 250)),
    }


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
