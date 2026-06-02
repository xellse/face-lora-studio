import json
import os
import re
import shutil
import subprocess
import time
import uuid
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field


WORKSPACE = Path(os.environ.get("RUNPOD_WORKSPACE", "/workspace"))
JOBS_DIR = WORKSPACE / "jobs"
COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_WORKFLOW_PATH = Path(os.environ.get("COMFY_WORKFLOW_PATH", WORKSPACE / "ComfyUI/workflows/z_image_Lora.json"))
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
Z_IMAGE_BASE_MODEL = "z-image-base"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image"
Z_IMAGE_ARCH = os.environ.get("Z_IMAGE_ARCH", "zimage").strip()
Z_IMAGE_ARCH = {"z_image": "zimage", "z-image": "zimage"}.get(Z_IMAGE_ARCH, Z_IMAGE_ARCH)
AI_TOOLKIT_DIR = WORKSPACE / "ai-toolkit"
MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", WORKSPACE / "models"))
Z_IMAGE_MODEL_DIR = Path(os.environ.get("Z_IMAGE_MODEL_DIR", MODEL_CACHE_DIR / "z-image-base"))
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "face-lora-assets")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
PUBLIC_STORAGE_BASE_URL = os.environ.get("PUBLIC_STORAGE_BASE_URL", "https://img.xellsun.com").rstrip("/")

APP_VERSION = "0.1.9"

app = FastAPI(title="Face LoRA RunPod Worker", version=APP_VERSION)


class DatasetJobRequest(BaseModel):
    dataset_id: str = Field(alias="datasetId")
    user_id: str = Field(default="local-user", alias="userId")
    raw_keys: list[str] = Field(default_factory=list, alias="rawKeys")
    raw_images: list["RawImage"] = Field(default_factory=list, alias="rawImages")
    trigger_word: str = Field(default="person_lora", alias="triggerWord")
    crop_size: int = Field(default=1024, alias="cropSize")


class RawImage(BaseModel):
    id: str
    index: int = 0
    name: str | None = None
    type: str | None = None
    size: int | None = None
    key: str | None = None
    url: str


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
    lora_name: str | None = Field(default=None, alias="loraName")
    lora_file: str | None = Field(default=None, alias="loraFile")
    model_path: str | None = Field(default=None, alias="modelPath")
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
            "architecture": Z_IMAGE_ARCH,
            "repo": Z_IMAGE_REPO,
            "localPath": str(Z_IMAGE_MODEL_DIR),
            "deployed": z_image_model_is_ready(),
        },
        "datasetProcessing": {
            "visionProvider": "openrouter",
            "visionModel": OPENROUTER_MODEL,
            "cropPolicy": "face_center_square_crop_resize_only",
            "openRouterConfigured": bool(OPENROUTER_API_KEY),
            "r2Configured": bool(R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY),
        },
        "workspace": str(WORKSPACE),
        "paths": {
            "jobs": str(JOBS_DIR),
            "comfy": str(WORKSPACE / "ComfyUI"),
            "aiToolkit": str(WORKSPACE / "ai-toolkit"),
            "models": str(MODEL_CACHE_DIR),
            "zImageBase": str(Z_IMAGE_MODEL_DIR),
            "loras": str(WORKSPACE / "ComfyUI/models/loras/local-user"),
            "comfyWorkflow": str(COMFY_WORKFLOW_PATH),
        },
        "comfy": comfy,
    }


@app.post("/jobs/dataset", dependencies=[Depends(require_token)])
def create_dataset_job(payload: DatasetJobRequest, background_tasks: BackgroundTasks):
    job = create_job("dataset_processing", payload.model_dump(by_alias=True))
    background_tasks.add_task(process_dataset, job["id"], payload)
    return job


@app.post("/jobs/train", dependencies=[Depends(require_token)])
def create_training_job(payload: TrainingJobRequest, background_tasks: BackgroundTasks):
    job = create_job("lora_training", payload.model_dump(by_alias=True))
    background_tasks.add_task(simulate_training, job["id"], payload)
    return job


@app.post("/jobs/generate", dependencies=[Depends(require_token)])
def create_generation_job(payload: GenerationJobRequest, background_tasks: BackgroundTasks):
    job = create_job("generation", payload.model_dump(by_alias=True))
    background_tasks.add_task(process_generation, job["id"], payload)
    return job


@app.get("/models/z-image/status", dependencies=[Depends(require_token)])
def get_z_image_model_status():
    return z_image_model_status()


@app.post("/models/z-image/prepare", dependencies=[Depends(require_token)])
def prepare_z_image_model(background_tasks: BackgroundTasks):
    job = create_job("model_prepare", {
        "model": Z_IMAGE_BASE_MODEL,
        "repo": Z_IMAGE_REPO,
        "targetPath": str(Z_IMAGE_MODEL_DIR),
    })
    background_tasks.add_task(prepare_z_image_model_job, job["id"])
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


def process_dataset(job_id: str, payload: DatasetJobRequest):
    try:
        if not payload.raw_images:
            raise RuntimeError("No rawImages were provided")
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if not (R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
            raise RuntimeError("R2 upload environment variables are not configured")

        update_job(job_id, 8, "running", "Preparing Gemini crop/caption pipeline")
        job_dir = JOBS_DIR / job_id
        raw_dir = job_dir / "raw"
        faces_dir = job_dir / "faces"
        raw_dir.mkdir(parents=True, exist_ok=True)
        faces_dir.mkdir(parents=True, exist_ok=True)

        faces = []
        review_items = []
        for index, raw in enumerate(payload.raw_images, start=1):
            update_job(job_id, min(90, 8 + int((index - 1) / max(1, len(payload.raw_images)) * 80)), "running", f"Processing image {index} of {len(payload.raw_images)}")
            raw_path = raw_dir / f"{index:04d}_{safe_job_name(raw.name or raw.id)}.jpg"
            source_image = download_image(raw.url)
            source_image.save(raw_path, "JPEG", quality=95)

            analysis = analyze_face_with_openrouter(raw.url, payload.trigger_word)
            if not analysis.get("usable", True):
                review_items.append({
                    "id": raw.id,
                    "reason": analysis.get("reason", "Gemini marked this image unusable"),
                    "sourceUrl": raw.url,
                })
                continue

            try:
                crop = crop_face(source_image, analysis, payload.crop_size)
            except Exception as exc:
                review_items.append({
                    "id": raw.id,
                    "reason": f"Face crop rejected: {exc}",
                    "sourceUrl": raw.url,
                    "analysis": analysis,
                })
                continue
            face_id = f"face_{uuid.uuid4().hex[:8]}"
            face_path = faces_dir / f"{face_id}.jpg"
            crop.save(face_path, "JPEG", quality=95)

            caption = ensure_trigger_word(analysis.get("caption", ""), payload.trigger_word)
            (faces_dir / f"{face_id}.txt").write_text(caption)
            object_key = f"datasets/{payload.dataset_id}/faces/{face_id}.jpg"
            https_url = upload_file_to_r2(face_path, object_key, "image/jpeg")
            faces.append({
                "id": face_id,
                "status": "approved",
                "caption": caption,
                "objectKey": object_key,
                "httpsUrl": https_url,
                "cropSize": payload.crop_size,
                "sourceUrl": raw.url,
            })

        update_job(job_id, 100, "completed", "Dataset crop and caption completed", {
            "faces": faces,
            "reviewItems": review_items,
            "cropSize": payload.crop_size,
            "visionModel": OPENROUTER_MODEL,
        })
    except Exception as exc:
        update_job(job_id, 100, "failed", f"Dataset processing failed: {exc}", {
            "error": str(exc),
            "visionModel": OPENROUTER_MODEL,
        })


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return ImageOps.exif_transpose(image).convert("RGB")


def analyze_face_with_openrouter(image_url: str, trigger_word: str) -> dict[str, Any]:
    prompt = f"""Analyze this image for FACE LoRA training preparation.
Return JSON only with this shape:
{{
  "usable": true,
  "reason": "",
  "faceTarget": "face_head_only",
  "containsBodyInFaceBox": false,
  "face": {{"centerX": 0.50, "centerY": 0.34, "width": 0.34, "height": 0.42}},
  "bbox": {{"x": 0.33, "y": 0.13, "width": 0.34, "height": 0.42}},
  "caption": "{trigger_word}, close-up portrait, ..."
}}
Rules:
- The goal is a face LoRA dataset, not a full-body or half-body dataset.
- face is the tight normalized geometry of the main person's visible face/head only.
- face must cover the face oval, chin, forehead, ears, and hair mass only.
- face and bbox must exclude shoulders, torso, arms, hands, clothing below the neck, watermarks, captions, and background text.
- If the original photo is full-body or half-body, still return face geometry around the face/head only.
- Set containsBodyInFaceBox=true if face/bbox includes shoulders, torso, hands, or significant clothing below the neck.
- If the face is too small, too blurry, occluded, or there is no clear single main face, set usable=false and explain reason.
- Caption should be concise visual training text and must not include names or identity claims.
- Include the trigger word exactly once in the caption: {trigger_word}
"""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://app.xellsun.com",
            "X-Title": "Face LoRA Studio",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    return parse_json_object(content)


def parse_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def crop_face(image: Image.Image, analysis: dict[str, Any], crop_size: int) -> Image.Image:
    width, height = image.size
    if analysis.get("containsBodyInFaceBox") is True:
        raise RuntimeError("Gemini face geometry includes body/shoulders instead of face only")

    face = normalized_face_geometry(analysis)
    validate_face_geometry(face)

    cx = face["centerX"] * width
    cy = face["centerY"] * height
    face_width = face["width"] * width
    face_height = face["height"] * height

    # Crop a square directly from the original image, centered on the detected face.
    # Resizing this square to crop_size is proportional scaling, not distortion.
    desired_side = max(face_width * 1.35, face_height * 1.25)
    max_centered_side = max_square_side_centered_in_image(cx, cy, width, height)
    side_px = int(round(min(desired_side, max_centered_side, min(width, height))))
    side_px = max(2, side_px)

    cropped = square_crop_centered_on_face(image, cx, cy, side_px)
    if cropped.width != cropped.height:
        raise RuntimeError(f"Internal crop was not square: {cropped.width}x{cropped.height}")
    if cropped.width == crop_size:
        return cropped
    return cropped.resize((crop_size, crop_size), Image.Resampling.LANCZOS)


def normalized_face_geometry(analysis: dict[str, Any]) -> dict[str, float]:
    face = analysis.get("face")
    if isinstance(face, dict):
        return {
            "centerX": clamp(float(face.get("centerX", 0.5)), 0, 1),
            "centerY": clamp(float(face.get("centerY", 0.5)), 0, 1),
            "width": clamp(float(face.get("width", 0.2)), 0.03, 1),
            "height": clamp(float(face.get("height", 0.25)), 0.03, 1),
        }

    bbox = analysis.get("bbox")
    if not isinstance(bbox, dict):
        raise RuntimeError("Gemini did not return valid face geometry")
    x = clamp(float(bbox.get("x", 0)), 0, 1)
    y = clamp(float(bbox.get("y", 0)), 0, 1)
    width = clamp(float(bbox.get("width", 1)), 0.03, 1)
    height = clamp(float(bbox.get("height", 1)), 0.03, 1)
    return {
        "centerX": clamp(x + width / 2, 0, 1),
        "centerY": clamp(y + height / 2, 0, 1),
        "width": width,
        "height": height,
    }


def validate_face_geometry(face: dict[str, float]):
    width = face["width"]
    height = face["height"]
    area = width * height
    if width > 0.68 or height > 0.68:
        raise RuntimeError("face geometry is too large and likely includes body framing")
    if area > 0.34:
        raise RuntimeError("face geometry area is too large for face-only training")
    if width < 0.04 or height < 0.04:
        raise RuntimeError("face geometry is too small for reliable training crop")


def max_square_side_centered_in_image(center_x: float, center_y: float, width: int, height: int) -> int:
    return max(2, int(2 * min(center_x, width - center_x, center_y, height - center_y)))


def square_crop_centered_on_face(image: Image.Image, center_x: float, center_y: float, side_px: int) -> Image.Image:
    width, height = image.size
    side_px = min(side_px, width, height)
    half = side_px / 2
    left = int(round(center_x - half))
    top = int(round(center_y - half))
    left = clamp_int(left, 0, width - side_px)
    top = clamp_int(top, 0, height - side_px)
    right = left + side_px
    bottom = top + side_px
    return image.crop((left, top, right, bottom))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        return minimum
    return max(minimum, min(maximum, value))


def upload_file_to_r2(path: Path, key: str, content_type: str) -> str:
    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )
    client.upload_file(
        str(path),
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"{PUBLIC_STORAGE_BASE_URL}/{key}"


def upload_bytes_to_r2(content: bytes, key: str, content_type: str) -> str:
    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )
    client.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=content,
        ContentType=content_type,
    )
    return f"{PUBLIC_STORAGE_BASE_URL}/{key}"


def prepare_z_image_model_job(job_id: str):
    try:
        update_job(job_id, 5, "running", "Preparing local Z-Image Base model directory")
        model_path = ensure_z_image_base_model(job_id)
        update_job(job_id, 100, "completed", "Z-Image Base model deployed locally", {
            **z_image_model_status(),
            "localPath": str(model_path),
        })
    except Exception as exc:
        update_job(job_id, 100, "failed", f"Z-Image Base model deployment failed: {exc}", {
            "error": str(exc),
            **z_image_model_status(),
        })


def ensure_z_image_base_model(job_id: str | None = None) -> Path:
    Z_IMAGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if z_image_model_is_ready():
        if job_id:
            append_job_log(job_id, f"Z-Image Base already deployed at {Z_IMAGE_MODEL_DIR}", progress=10, status="running")
        return Z_IMAGE_MODEL_DIR

    if job_id:
        append_job_log(job_id, f"Downloading {Z_IMAGE_REPO} to {Z_IMAGE_MODEL_DIR}", progress=7, status="running")
    snapshot_download(
        repo_id=Z_IMAGE_REPO,
        local_dir=str(Z_IMAGE_MODEL_DIR),
        token=HF_TOKEN,
    )
    marker_path = Z_IMAGE_MODEL_DIR / ".face_lora_model_ready.json"
    marker_path.write_text(json.dumps({
        "repo": Z_IMAGE_REPO,
        "architecture": Z_IMAGE_ARCH,
        "downloadedAt": now_iso(),
    }, indent=2))
    if job_id:
        append_job_log(job_id, f"Z-Image Base deployed at {Z_IMAGE_MODEL_DIR}", progress=10, status="running")
    return Z_IMAGE_MODEL_DIR


def z_image_model_status() -> dict[str, Any]:
    return {
        "model": Z_IMAGE_BASE_MODEL,
        "repo": Z_IMAGE_REPO,
        "architecture": Z_IMAGE_ARCH,
        "localPath": str(Z_IMAGE_MODEL_DIR),
        "deployed": z_image_model_is_ready(),
        "hfTokenConfigured": bool(HF_TOKEN),
        "sizeBytes": directory_size(Z_IMAGE_MODEL_DIR) if Z_IMAGE_MODEL_DIR.exists() else 0,
    }


def z_image_model_is_ready() -> bool:
    if not Z_IMAGE_MODEL_DIR.exists():
        return False
    if (Z_IMAGE_MODEL_DIR / ".face_lora_model_ready.json").exists():
        return True
    if (Z_IMAGE_MODEL_DIR / "model_index.json").exists():
        return True
    return any(Z_IMAGE_MODEL_DIR.rglob("*.safetensors"))


def directory_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def simulate_training(job_id: str, payload: TrainingJobRequest):
    try:
        if not payload.dataset_images:
            raise RuntimeError("No datasetImages were provided for training")

        update_job(job_id, 6, "running", "Ensuring Z-Image Base model is deployed locally")
        base_model_path = ensure_z_image_base_model(job_id)

        update_job(job_id, 10, "running", "Downloading training images and captions")
        dataset_dir = prepare_training_dataset(job_id, payload)

        update_job(job_id, 18, "running", "Writing Z-Image Base AI Toolkit config")
        config_path = write_z_image_training_config(job_id, payload, dataset_dir, base_model_path)

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
            "baseModelLocalPath": str(base_model_path),
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


def write_z_image_training_config(job_id: str, payload: TrainingJobRequest, images_dir: Path, model_path: Path) -> Path:
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
        name_or_path: "{yaml_escape(str(model_path))}"
        arch: "{yaml_escape(Z_IMAGE_ARCH)}"
        quantize: true
        qtype: "qfloat8"
        quantize_te: true
        qtype_te: "qfloat8"
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


def process_generation(job_id: str, payload: GenerationJobRequest):
    try:
        if not COMFY_WORKFLOW_PATH.exists():
            raise RuntimeError(f"ComfyUI workflow not found at {COMFY_WORKFLOW_PATH}")
        if not (R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
            raise RuntimeError("R2 upload environment variables are not configured")

        settings = generation_settings(payload.settings)
        images: list[dict[str, Any]] = []
        update_job(job_id, 8, "running", "Loading ComfyUI workflow")

        for index in range(settings["count"]):
            seed = settings["seed"] + index
            update_job(job_id, min(92, 10 + int(index / max(1, settings["count"]) * 80)), "running", f"Generating image {index + 1} of {settings['count']}")
            workflow = load_generation_workflow()
            patch_generation_workflow(workflow, payload, settings, seed, index)
            comfy_prompt_id = queue_comfy_prompt(workflow, job_id)
            history = wait_for_comfy_history(comfy_prompt_id, job_id)
            output_images = download_comfy_history_images(history)
            if not output_images:
                raise RuntimeError(f"ComfyUI returned no images for prompt {comfy_prompt_id}")
            for image_bytes, content_type in output_images:
                if len(images) >= settings["count"]:
                    break
                image_index = len(images)
                extension = "png" if content_type == "image/png" else "jpg"
                object_key = f"generated/{payload.user_id}/{payload.task_id}/image_{image_index + 1}.{extension}"
                https_url = upload_bytes_to_r2(image_bytes, object_key, content_type)
                images.append({
                    "id": f"image_{uuid.uuid4().hex[:8]}",
                    "index": image_index,
                    "localUrl": https_url,
                    "httpsUrl": https_url,
                    "width": settings["width"],
                    "height": settings["height"],
                    "seed": seed,
                    "comfyPromptId": comfy_prompt_id,
                })

        update_job(job_id, 100, "completed", "ComfyUI generation completed", {
            "taskId": payload.task_id,
            "images": images,
            "settings": settings,
            "workflowPath": str(COMFY_WORKFLOW_PATH),
        })
    except Exception as exc:
        update_job(job_id, 100, "failed", f"Generation failed: {exc}", {
            "error": str(exc),
            "taskId": payload.task_id,
            "workflowPath": str(COMFY_WORKFLOW_PATH),
        })


def generation_settings(settings: dict[str, Any]) -> dict[str, Any]:
    width = max(512, min(2048, int(settings.get("width", 1024))))
    height = max(512, min(width, int(settings.get("height", 1024))))
    return {
        "count": max(1, min(12, int(settings.get("count", 1)))),
        "width": width,
        "height": height,
        "seed": int(settings.get("seed", random_seed())),
        "steps": max(15, min(150, int(settings.get("steps", 40)))),
        "cfg": float(settings.get("cfg", 5)),
        "sampler": str(settings.get("sampler", "euler")),
        "scheduler": str(settings.get("scheduler", "normal")),
        "loraWeight": float(settings.get("loraWeight", 0.85)),
    }


def random_seed() -> int:
    return int(time.time() * 1000) % 2_000_000_000


def load_generation_workflow() -> dict[str, Any]:
    workflow = json.loads(COMFY_WORKFLOW_PATH.read_text())
    if not isinstance(workflow, dict):
        raise RuntimeError("ComfyUI workflow JSON must be an API workflow object")
    return workflow


def patch_generation_workflow(workflow: dict[str, Any], payload: GenerationJobRequest, settings: dict[str, Any], seed: int, index: int):
    lora_name = payload.lora_file or comfy_lora_name(payload)
    replacements = {
        "__PROMPT__": payload.prompt,
        "__NEGATIVE_PROMPT__": payload.negative_prompt,
        "__LORA_NAME__": lora_name,
        "__LORA_FILE__": lora_name,
        "__LORA_WEIGHT__": settings["loraWeight"],
        "__WIDTH__": settings["width"],
        "__HEIGHT__": settings["height"],
        "__SEED__": seed,
        "__STEPS__": settings["steps"],
        "__CFG__": settings["cfg"],
        "__SAMPLER__": settings["sampler"],
        "__SCHEDULER__": settings["scheduler"],
        "__TASK_ID__": payload.task_id,
        "__INDEX__": index + 1,
    }
    replace_placeholders(workflow, replacements)

    clip_nodes = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get("class_type", ""))
        lower_class = class_type.lower()

        if "lora" in lower_class:
            set_existing(inputs, ["lora_name", "lora", "name"], lora_name)
            set_existing(inputs, ["strength_model", "strength_clip", "strength"], settings["loraWeight"])

        if "ksampler" in lower_class or "sampler" in lower_class:
            set_existing(inputs, ["seed"], seed)
            set_existing(inputs, ["steps"], settings["steps"])
            set_existing(inputs, ["cfg"], settings["cfg"])
            set_existing(inputs, ["sampler_name", "sampler"], settings["sampler"])
            set_existing(inputs, ["scheduler"], settings["scheduler"])

        if "emptylatentimage" in lower_class or "latent" in lower_class:
            set_existing(inputs, ["width"], settings["width"])
            set_existing(inputs, ["height"], settings["height"])
            set_existing(inputs, ["batch_size"], 1)

        if "saveimage" in lower_class:
            set_existing(inputs, ["filename_prefix"], f"face-lora/{payload.task_id}/{index + 1:03d}")

        if "cliptextencode" in lower_class and "text" in inputs:
            clip_nodes.append((str(node_id), inputs))

    clip_nodes.sort(key=lambda item: item[0])
    if clip_nodes:
        clip_nodes[0][1]["text"] = payload.prompt
    if len(clip_nodes) > 1:
        clip_nodes[1][1]["text"] = payload.negative_prompt


def replace_placeholders(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = replace_placeholders(item, replacements)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = replace_placeholders(item, replacements)
        return value
    if isinstance(value, str):
        for token, replacement in replacements.items():
            if value == token:
                return replacement
            value = value.replace(token, str(replacement))
    return value


def set_existing(inputs: dict[str, Any], keys: list[str], value: Any):
    for key in keys:
        if key in inputs:
            inputs[key] = value


def comfy_lora_name(payload: GenerationJobRequest) -> str:
    if payload.model_path:
        marker = "/models/loras/"
        if marker in payload.model_path:
            return payload.model_path.split(marker, 1)[1]
        path = Path(payload.model_path)
        if path.name:
            return f"{payload.user_id}/{path.name}"
    return f"{payload.user_id}/{payload.lora_id}.safetensors"


def queue_comfy_prompt(workflow: dict[str, Any], job_id: str) -> str:
    debug_dir = JOBS_DIR / job_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "comfy_prompt.json").write_text(json.dumps(workflow, indent=2, ensure_ascii=False))
    response = requests.post(f"{COMFY_BASE_URL}/prompt", json={
        "prompt": workflow,
        "client_id": job_id,
    }, timeout=30)
    if not response.ok:
        error_text = response.text[:4000]
        append_job_log(job_id, f"ComfyUI /prompt error {response.status_code}: {error_text}", status="running")
        raise RuntimeError(f"ComfyUI /prompt failed with {response.status_code}: {error_text}")
    payload = response.json()
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI /prompt did not return prompt_id: {payload}")
    return prompt_id


def wait_for_comfy_history(prompt_id: str, job_id: str) -> dict[str, Any]:
    deadline = time.time() + 1800
    while time.time() < deadline:
        response = requests.get(f"{COMFY_BASE_URL}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        history = response.json()
        if prompt_id in history:
            return history[prompt_id]
        append_job_log(job_id, f"Waiting for ComfyUI prompt {prompt_id}", status="running")
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for ComfyUI prompt {prompt_id}")


def download_comfy_history_images(history: dict[str, Any]) -> list[tuple[bytes, str]]:
    images: list[tuple[bytes, str]] = []
    outputs = history.get("outputs", {})
    if not isinstance(outputs, dict):
        return images
    for output in outputs.values():
        for image in output.get("images", []) if isinstance(output, dict) else []:
            filename = image.get("filename")
            if not filename:
                continue
            response = requests.get(f"{COMFY_BASE_URL}/view", params={
                "filename": filename,
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }, timeout=120)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "image/png").split(";")[0]
            images.append((response.content, content_type if content_type.startswith("image/") else "image/png"))
    return images


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
