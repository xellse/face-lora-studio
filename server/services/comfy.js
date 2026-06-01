import { id, nowIso, patch, saveState, upsert, getState } from "../store.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function startGeneration({ payload, storage }) {
  const state = getState();
  const lora = state.loras.find((entry) => entry.id === payload.loraId);
  if (!lora || lora.status !== "ready") return { error: "Selected LoRA is not ready" };

  const task = upsert("generationTasks", {
    id: id("gen"),
    userId: lora.userId,
    loraId: lora.id,
    loraName: lora.name,
    status: "queued",
    progress: 0,
    folderName: `${new Date().toISOString().slice(0, 10)} ${lora.name}`,
    prompt: payload.prompt,
    negativePrompt: payload.negativePrompt || "",
    settings: {
      count: clamp(Number(payload.count || 4), 1, 12),
      width: Number(payload.width || 1024),
      height: Number(payload.height || 1024),
      seed: payload.seed ? Number(payload.seed) : Math.floor(Math.random() * 2_000_000_000),
      steps: Number(payload.steps || 40),
      cfg: Number(payload.cfg || 5),
      sampler: payload.sampler || "euler",
      loraWeight: Number(payload.loraWeight || 0.85)
    },
    images: [],
    comfyPromptId: null,
    createdAt: nowIso(),
    updatedAt: nowIso()
  });

  void runGeneration({ taskId: task.id, storage });
  return { task };
}

async function runGeneration({ taskId, storage }) {
  patch("generationTasks", taskId, {
    status: "preparing",
    progress: 10,
    comfyPromptId: id("prompt"),
    message: "Injecting prompt and LoRA into ComfyUI workflow"
  });
  await sleep(800);

  patch("generationTasks", taskId, {
    status: "running",
    progress: 38,
    message: "Queued in ComfyUI"
  });
  await sleep(1000);

  const state = getState();
  const task = state.generationTasks.find((entry) => entry.id === taskId);
  if (!task) return;

  const images = [];
  for (let index = 0; index < task.settings.count; index += 1) {
    patch("generationTasks", taskId, {
      progress: 45 + Math.round(((index + 1) / task.settings.count) * 40),
      message: `Rendering image ${index + 1} of ${task.settings.count}`
    });
    await sleep(500);

    const svg = renderMockImage({ task, index });
    const asset = await storage.putDataUrl({
      key: `generated/${task.userId}/${task.id}/image_${index + 1}.svg`,
      dataUrl: `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`,
      contentType: "image/svg+xml",
      label: `Generated image ${index + 1}`
    });
    images.push({
      id: id("image"),
      index,
      localUrl: asset.localUrl,
      httpsUrl: asset.httpsUrl,
      width: task.settings.width,
      height: task.settings.height,
      seed: task.settings.seed + index
    });
  }

  const finalTask = state.generationTasks.find((entry) => entry.id === taskId);
  finalTask.images = images;
  finalTask.status = "completed";
  finalTask.progress = 100;
  finalTask.message = "Images uploaded to HTTPS storage";
  finalTask.completedAt = nowIso();
  finalTask.updatedAt = nowIso();
  saveState();
}

function renderMockImage({ task, index }) {
  const width = task.settings.width;
  const height = task.settings.height;
  const hue = (task.settings.seed + index * 47) % 360;
  const prompt = escapeXml(task.prompt.slice(0, 90));
  const lora = escapeXml(task.loraName);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <defs>
      <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="hsl(${hue}, 72%, 58%)"/>
        <stop offset="55%" stop-color="hsl(${(hue + 70) % 360}, 62%, 48%)"/>
        <stop offset="100%" stop-color="hsl(${(hue + 150) % 360}, 70%, 36%)"/>
      </linearGradient>
    </defs>
    <rect width="100%" height="100%" fill="url(#bg)"/>
    <circle cx="${width * 0.5}" cy="${height * 0.38}" r="${Math.min(width, height) * 0.18}" fill="rgba(255,255,255,.38)"/>
    <rect x="${width * 0.28}" y="${height * 0.58}" width="${width * 0.44}" height="${height * 0.2}" rx="${Math.min(width, height) * 0.08}" fill="rgba(255,255,255,.28)"/>
    <text x="48" y="${height - 128}" fill="white" font-family="Arial, sans-serif" font-size="34" font-weight="700">Mock ComfyUI Result ${index + 1}</text>
    <text x="48" y="${height - 82}" fill="white" font-family="Arial, sans-serif" font-size="22">LoRA: ${lora} | seed ${task.settings.seed + index}</text>
    <text x="48" y="${height - 44}" fill="white" font-family="Arial, sans-serif" font-size="20">${prompt}</text>
  </svg>`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
