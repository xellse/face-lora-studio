const USER_ID = "local-user";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") return corsResponse(env);

    try {
      if (request.method === "GET" && url.pathname === "/api/state") {
        return json(await getState(env), env);
      }

      if (request.method === "POST" && url.pathname === "/api/uploads") {
        return json(await uploadObject(request, env), env, 201);
      }

      if (request.method === "POST" && url.pathname === "/api/datasets") {
        return json(await createDataset(request, env), env, 201);
      }

      if (request.method === "POST" && url.pathname === "/api/training") {
        return json(await createTrainingJob(request, env), env, 201);
      }

      if (request.method === "POST" && url.pathname === "/api/generation") {
        return json(await createGenerationTask(request, env), env, 201);
      }

      return json({ error: "Route not found" }, env, 404);
    } catch (error) {
      return json({ error: error.message || "Internal error" }, env, 500);
    }
  }
};

async function getState(env) {
  const [datasets, faces, jobs, loras, generationTasks] = await Promise.all([
    all(env, "SELECT * FROM datasets ORDER BY created_at DESC LIMIT 50"),
    all(env, "SELECT * FROM faces ORDER BY created_at DESC LIMIT 300"),
    all(env, "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100"),
    all(env, "SELECT * FROM loras ORDER BY created_at DESC LIMIT 100"),
    all(env, "SELECT * FROM generation_tasks ORDER BY created_at DESC LIMIT 100")
  ]);

  return {
    users: [{ id: USER_ID, name: "Internal User" }],
    datasets: datasets.map((dataset) => ({
      id: dataset.id,
      userId: dataset.user_id,
      name: dataset.name,
      triggerWord: dataset.trigger_word,
      cropSize: dataset.crop_size,
      status: dataset.status,
      rawPhotoCount: dataset.raw_photo_count,
      faces: faces.filter((face) => face.dataset_id === dataset.id).map(mapFace),
      reviewItems: [],
      createdAt: dataset.created_at,
      updatedAt: dataset.updated_at
    })),
    jobs: jobs.map(mapJob),
    loras: loras.map(mapLora),
    generationTasks: generationTasks.map(mapGenerationTask),
    config: {
      storageDriver: "r2",
      workerDriver: "runpod-pending",
      comfyDriver: "comfyui-pending",
      publicStorageBaseUrl: env.PUBLIC_STORAGE_BASE_URL
    }
  };
}

async function uploadObject(request, env) {
  const form = await request.formData();
  const file = form.get("file");
  const key = form.get("key");
  if (!file || !key) throw new Error("file and key are required");

  await env.ASSETS.put(key, file.stream(), {
    httpMetadata: {
      contentType: file.type || "application/octet-stream"
    }
  });

  return {
    key,
    httpsUrl: `${env.PUBLIC_STORAGE_BASE_URL}/${key}`
  };
}

async function createDataset(request, env) {
  const body = await request.json();
  const now = nowIso();
  const datasetId = id("dataset");
  const photos = body.photos || [];

  await env.DB.prepare(
    "INSERT INTO datasets (id, user_id, name, trigger_word, crop_size, status, raw_photo_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
  )
    .bind(datasetId, USER_ID, body.name || "Portrait dataset", body.triggerWord || "person_lora", Number(body.cropSize || 1024), "ready_for_training", photos.length, now, now)
    .run();

  for (let index = 0; index < photos.length; index += 1) {
    const photo = photos[index];
    const faceId = id("face");
    const objectKey = `datasets/${datasetId}/faces/${faceId}.jpg`;
    const data = dataUrlToBytes(photo.dataUrl);
    await env.ASSETS.put(objectKey, data.bytes, {
      httpMetadata: { contentType: data.contentType || photo.type || "image/jpeg" }
    });
    await env.DB.prepare(
      "INSERT INTO faces (id, dataset_id, status, caption, object_key, https_url, crop_size, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
      .bind(
        faceId,
        datasetId,
        "approved",
        `${body.triggerWord || "person_lora"}, close-up portrait, clean face crop, sharp facial detail`,
        objectKey,
        `${env.PUBLIC_STORAGE_BASE_URL}/${objectKey}`,
        Number(body.cropSize || 1024),
        now,
        now
      )
      .run();
  }

  const job = await insertJob(env, {
    type: "dataset_processing",
    status: "completed",
    progress: 100,
    datasetId,
    message: "Dataset imported to R2. Replace this mock step with RunPod face crop/caption worker."
  });

  return { datasetId, job };
}

async function createTrainingJob(request, env) {
  const body = await request.json();
  const now = nowIso();
  const loraId = id("lora");
  const modelPath = `${env.RUNPOD_WORKSPACE || "/workspace"}/ComfyUI/models/loras/${USER_ID}/${loraId}.safetensors`;
  const parameters = {
    steps: Number(body.steps || 1200),
    learningRate: body.learningRate || "1e-4",
    rank: Number(body.rank || 16),
    repeats: Number(body.repeats || 10),
    resolution: Number(body.resolution || 1024),
    saveEvery: Number(body.saveEvery || 250)
  };

  await env.DB.prepare(
    "INSERT INTO loras (id, user_id, dataset_id, name, trigger_word, base_model, status, progress, model_path, parameters, created_at, updated_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
  )
    .bind(loraId, USER_ID, body.datasetId, body.loraName, body.triggerWord, body.baseModel || "sdxl", "ready", 100, modelPath, JSON.stringify(parameters), now, now, now)
    .run();

  const job = await insertJob(env, {
    type: "lora_training",
    status: "completed",
    progress: 100,
    datasetId: body.datasetId,
    loraId,
    message: "LoRA record created. Replace this mock step with RunPod AI Toolkit call."
  });

  return { loraId, job };
}

async function createGenerationTask(request, env) {
  const body = await request.json();
  const lora = await env.DB.prepare("SELECT * FROM loras WHERE id = ?").bind(body.loraId).first();
  if (!lora) throw new Error("Selected LoRA not found");

  const now = nowIso();
  const taskId = id("gen");
  const count = clamp(Number(body.count || 4), 1, 12);
  const settings = {
    count,
    width: Number(body.width || 1024),
    height: Number(body.height || 1024),
    seed: body.seed ? Number(body.seed) : Math.floor(Math.random() * 2_000_000_000),
    steps: Number(body.steps || 28),
    cfg: Number(body.cfg || 4.5),
    sampler: body.sampler || "euler",
    loraWeight: Number(body.loraWeight || 0.85)
  };
  const images = [];

  for (let index = 0; index < count; index += 1) {
    const key = `generated/${USER_ID}/${taskId}/image_${index + 1}.svg`;
    const svg = mockImageSvg({ prompt: body.prompt, loraName: lora.name, settings, index });
    await env.ASSETS.put(key, svg, { httpMetadata: { contentType: "image/svg+xml" } });
    images.push({
      id: id("image"),
      index,
      localUrl: `${env.PUBLIC_STORAGE_BASE_URL}/${key}`,
      httpsUrl: `${env.PUBLIC_STORAGE_BASE_URL}/${key}`,
      width: settings.width,
      height: settings.height,
      seed: settings.seed + index
    });
  }

  await env.DB.prepare(
    "INSERT INTO generation_tasks (id, user_id, lora_id, lora_name, status, progress, folder_name, prompt, negative_prompt, settings, images, message, comfy_prompt_id, created_at, updated_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
  )
    .bind(
      taskId,
      USER_ID,
      body.loraId,
      lora.name,
      "completed",
      100,
      `${new Date().toISOString().slice(0, 10)} ${lora.name}`,
      body.prompt,
      body.negativePrompt || "",
      JSON.stringify(settings),
      JSON.stringify(images),
      "Images created in R2. Replace this mock step with RunPod ComfyUI call.",
      id("prompt"),
      now,
      now,
      now
    )
    .run();

  return { taskId };
}

async function insertJob(env, input) {
  const now = nowIso();
  const job = {
    id: id("job"),
    type: input.type,
    status: input.status,
    progress: input.progress,
    datasetId: input.datasetId || null,
    loraId: input.loraId || null,
    generationTaskId: input.generationTaskId || null,
    message: input.message || "",
    logs: [{ at: now, message: input.message || "" }],
    payload: input.payload || null,
    createdAt: now,
    updatedAt: now
  };

  await env.DB.prepare(
    "INSERT INTO jobs (id, type, status, progress, dataset_id, lora_id, generation_task_id, message, logs, payload, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
  )
    .bind(job.id, job.type, job.status, job.progress, job.datasetId, job.loraId, job.generationTaskId, job.message, JSON.stringify(job.logs), job.payload ? JSON.stringify(job.payload) : null, now, now)
    .run();

  return job;
}

async function all(env, sql) {
  const result = await env.DB.prepare(sql).all();
  return result.results || [];
}

function mapFace(face) {
  return {
    id: face.id,
    sourcePhotoId: face.id,
    cropSize: face.crop_size,
    status: face.status,
    caption: face.caption,
    captionFile: `${face.id}.txt`,
    localUrl: face.https_url,
    httpsUrl: face.https_url
  };
}

function mapJob(job) {
  return {
    id: job.id,
    type: job.type,
    status: job.status,
    progress: job.progress,
    datasetId: job.dataset_id,
    loraId: job.lora_id,
    generationTaskId: job.generation_task_id,
    message: job.message,
    logs: parseJson(job.logs, []),
    createdAt: job.created_at,
    updatedAt: job.updated_at
  };
}

function mapLora(lora) {
  return {
    id: lora.id,
    userId: lora.user_id,
    datasetId: lora.dataset_id,
    name: lora.name,
    triggerWord: lora.trigger_word,
    baseModel: lora.base_model,
    status: lora.status,
    progress: lora.progress,
    modelPath: lora.model_path,
    parameters: parseJson(lora.parameters, {}),
    createdAt: lora.created_at,
    updatedAt: lora.updated_at,
    completedAt: lora.completed_at
  };
}

function mapGenerationTask(task) {
  return {
    id: task.id,
    userId: task.user_id,
    loraId: task.lora_id,
    loraName: task.lora_name,
    status: task.status,
    progress: task.progress,
    folderName: task.folder_name,
    prompt: task.prompt,
    negativePrompt: task.negative_prompt,
    settings: parseJson(task.settings, {}),
    images: parseJson(task.images, []),
    message: task.message,
    comfyPromptId: task.comfy_prompt_id,
    createdAt: task.created_at,
    updatedAt: task.updated_at,
    completedAt: task.completed_at
  };
}

function dataUrlToBytes(dataUrl) {
  const match = /^data:([^;]+);base64,(.*)$/.exec(dataUrl || "");
  if (!match) throw new Error("Invalid data URL");
  const binary = atob(match[2]);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return { contentType: match[1], bytes };
}

function mockImageSvg({ prompt, loraName, settings, index }) {
  const hue = (settings.seed + index * 47) % 360;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${settings.width}" height="${settings.height}" viewBox="0 0 ${settings.width} ${settings.height}">
    <rect width="100%" height="100%" fill="hsl(${hue},70%,48%)"/>
    <circle cx="${settings.width / 2}" cy="${settings.height * 0.38}" r="${Math.min(settings.width, settings.height) * 0.18}" fill="rgba(255,255,255,.42)"/>
    <text x="42" y="${settings.height - 96}" fill="white" font-family="Arial" font-size="30" font-weight="700">${escapeXml(loraName)}</text>
    <text x="42" y="${settings.height - 52}" fill="white" font-family="Arial" font-size="20">${escapeXml(String(prompt || "").slice(0, 80))}</text>
  </svg>`;
}

function json(payload, env, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...corsHeaders(env)
    }
  });
}

function corsResponse(env) {
  return new Response(null, { status: 204, headers: corsHeaders(env) });
}

function corsHeaders(env) {
  return {
    "Access-Control-Allow-Origin": env.APP_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,PATCH,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization"
  };
}

function id(prefix) {
  return `${prefix}_${crypto.randomUUID().slice(0, 8)}`;
}

function nowIso() {
  return new Date().toISOString();
}

function parseJson(value, fallback) {
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
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
