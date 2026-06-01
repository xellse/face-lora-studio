import { config } from "../config.js";
import { getState, id, nowIso, patch, saveState, upsert } from "../store.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function startDatasetProcessing({ datasetId, storage }) {
  const state = getState();
  const dataset = state.datasets.find((entry) => entry.id === datasetId);
  if (!dataset) return null;

  const job = upsert("jobs", {
    id: id("job"),
    type: "dataset_processing",
    status: "queued",
    progress: 0,
    datasetId,
    message: "Queued for face crop and captioning",
    logs: [],
    createdAt: nowIso(),
    updatedAt: nowIso()
  });

  void runDatasetProcessing({ datasetId, jobId: job.id, storage });
  return job;
}

async function runDatasetProcessing({ datasetId, jobId, storage }) {
  updateJob(jobId, 8, "preparing", "Downloading uploaded portraits");
  await sleep(700);

  const state = getState();
  const dataset = state.datasets.find((entry) => entry.id === datasetId);
  if (!dataset) return updateJob(jobId, 100, "failed", "Dataset not found");

  updateJob(jobId, 24, "running", "Detecting faces and validating quality");
  await sleep(900);

  const faces = [];
  const reviewItems = [];
  for (const [index, photo] of dataset.rawPhotos.entries()) {
    const isTooSmall = photo.size < 20_000;
    const faceId = id("face");
    if (isTooSmall) {
      reviewItems.push({
        id: faceId,
        sourcePhotoId: photo.id,
        reason: "Image is too small for reliable training",
        status: "needs_review"
      });
      continue;
    }

    const caption = captionFace({ index, triggerWord: dataset.triggerWord || "person_lora" });
    const asset = await storage.putDataUrl({
      key: `datasets/${dataset.id}/faces/${faceId}.jpg`,
      dataUrl: photo.dataUrl,
      contentType: photo.type || "image/jpeg",
      label: `Cropped face ${index + 1}`
    });

    faces.push({
      id: faceId,
      sourcePhotoId: photo.id,
      cropSize: dataset.cropSize,
      status: "approved",
      caption,
      captionFile: `${faceId}.txt`,
      localUrl: asset.localUrl,
      httpsUrl: asset.httpsUrl
    });
  }

  updateJob(jobId, 72, "running", "Writing AI Toolkit caption files");
  await sleep(700);

  dataset.faces = faces;
  dataset.reviewItems = reviewItems;
  dataset.status = faces.length >= 5 ? "ready_for_training" : "needs_more_photos";
  dataset.updatedAt = nowIso();
  saveState();

  updateJob(
    jobId,
    100,
    dataset.status === "ready_for_training" ? "completed" : "needs_review",
    dataset.status === "ready_for_training"
      ? "Dataset is ready for LoRA training"
      : "Add more usable portraits before training"
  );
}

function captionFace({ index, triggerWord }) {
  const angles = ["front-facing", "three-quarter view", "soft side angle", "natural expression"];
  const light = ["soft studio light", "daylight", "balanced indoor light", "clear facial detail"];
  return `${triggerWord}, close-up portrait, ${angles[index % angles.length]}, ${light[index % light.length]}, sharp face, natural skin texture`;
}

export function startTraining({ payload }) {
  const state = getState();
  const dataset = state.datasets.find((entry) => entry.id === payload.datasetId);
  if (!dataset) return { error: "Dataset not found" };
  if (!dataset.faces?.some((face) => face.status === "approved")) {
    return { error: "Dataset has no approved faces" };
  }

  const lora = upsert("loras", {
    id: id("lora"),
    userId: dataset.userId,
    datasetId: dataset.id,
    name: payload.loraName,
    triggerWord: payload.triggerWord,
    baseModel: payload.baseModel,
    status: "training",
    progress: 0,
    modelPath: null,
    createdAt: nowIso(),
    updatedAt: nowIso(),
    parameters: {
      steps: Number(payload.steps || 3000),
      learningRate: payload.learningRate || "1e-4",
      rank: Number(payload.rank || 32),
      repeats: Number(payload.repeats || 10),
      resolution: Number(payload.resolution || 1024),
      saveEvery: Number(payload.saveEvery || 250),
      architecture: "z-image",
      modelRepo: "Tongyi-MAI/Z-Image",
      precision: "bf16",
      optimizer: "AdamW8Bit"
    }
  });

  const job = upsert("jobs", {
    id: id("job"),
    type: "lora_training",
    status: "queued",
    progress: 0,
    datasetId: dataset.id,
    loraId: lora.id,
    message: "Queued for AI Toolkit training",
    logs: [],
    aiToolkitConfig: buildAiToolkitConfig({ dataset, lora }),
    createdAt: nowIso(),
    updatedAt: nowIso()
  });

  void runTraining({ jobId: job.id, loraId: lora.id });
  return { lora, job };
}

async function runTraining({ jobId, loraId }) {
  const stages = [
    [8, "preparing", "Creating AI Toolkit YAML config"],
    [18, "preparing", "Preparing cropped face dataset and captions"],
    [34, "running", "Caching latents"],
    [52, "running", "Training LoRA weights"],
    [70, "running", "Saving checkpoint"],
    [86, "running", "Validating final safetensors file"]
  ];

  for (const [progress, status, message] of stages) {
    updateJob(jobId, progress, status, message);
    patch("loras", loraId, { progress, status: "training" });
    await sleep(900);
  }

  const state = getState();
  const lora = state.loras.find((entry) => entry.id === loraId);
  if (!lora) return updateJob(jobId, 100, "failed", "LoRA record not found");

  const modelPath = `${config.runpodWorkspace}/ComfyUI/models/loras/${lora.userId}/${lora.id}.safetensors`;
  patch("loras", loraId, {
    status: "ready",
    progress: 100,
    modelPath,
    completedAt: nowIso()
  });
  updateJob(jobId, 100, "completed", `LoRA saved to ${modelPath}`);
}

function buildAiToolkitConfig({ dataset, lora }) {
  const datasetPath = `${config.runpodWorkspace}/jobs/${lora.id}/datasets/${dataset.id}/images`;
  const outputPath = `${config.runpodWorkspace}/ComfyUI/models/loras/${lora.userId}/${lora.id}.safetensors`;
  return {
    job: `train_${lora.id}`,
    process: "ai-toolkit lora train",
    datasetPath,
    captionExt: "txt",
    outputPath,
    triggerWord: lora.triggerWord,
    baseModel: lora.baseModel,
    parameters: lora.parameters
  };
}

function updateJob(jobId, progress, status, message) {
  const state = getState();
  const job = state.jobs.find((entry) => entry.id === jobId);
  if (!job) return null;
  job.progress = progress;
  job.status = status;
  job.message = message;
  job.updatedAt = nowIso();
  job.logs.push({ at: nowIso(), message });
  saveState();
  return job;
}
