import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "./config.js";
import { getState, id, nowIso, saveState, upsert } from "./store.js";
import { createStorage } from "./services/storage.js";
import { startDatasetProcessing, startTraining } from "./services/workers.js";
import { startGeneration } from "./services/comfy.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, "..", "public");
const storage = createStorage();

const server = http.createServer(async (req, res) => {
  try {
    if (req.url.startsWith("/api/")) {
      await handleApi(req, res);
      return;
    }
    serveStatic(req, res);
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Internal server error" });
  }
});

server.listen(config.port, () => {
  console.log(`Face LoRA PWA running at http://localhost:${config.port}`);
});

async function handleApi(req, res) {
  const url = new URL(req.url, config.appOrigin);
  const method = req.method || "GET";

  if (method === "GET" && url.pathname === "/api/state") {
    return sendJson(res, 200, publicState());
  }

  if (method === "POST" && url.pathname === "/api/datasets") {
    const body = await readJson(req);
    const dataset = upsert("datasets", {
      id: id("dataset"),
      userId: "local-user",
      name: body.name || "Portrait dataset",
      cropSize: Number(body.cropSize || 1024),
      triggerWord: body.triggerWord || "person_lora",
      status: "processing",
      rawPhotos: (body.photos || []).map((photo, index) => ({
        id: id("photo"),
        name: photo.name || `portrait_${index + 1}.jpg`,
        type: photo.type || "image/jpeg",
        size: Number(photo.size || 0),
        dataUrl: photo.dataUrl,
        uploadedAt: nowIso()
      })),
      faces: [],
      reviewItems: [],
      createdAt: nowIso(),
      updatedAt: nowIso()
    });
    const job = startDatasetProcessing({ datasetId: dataset.id, storage });
    return sendJson(res, 201, { dataset, job });
  }

  if (method === "PATCH" && url.pathname.startsWith("/api/datasets/")) {
    const body = await readJson(req);
    const [, , , datasetId, , faceId] = url.pathname.split("/");
    const state = getState();
    const dataset = state.datasets.find((entry) => entry.id === datasetId);
    if (!dataset) return sendJson(res, 404, { error: "Dataset not found" });
    const face = dataset.faces.find((entry) => entry.id === faceId);
    if (!face) return sendJson(res, 404, { error: "Face not found" });
    if (typeof body.caption === "string") face.caption = body.caption;
    if (typeof body.status === "string") face.status = body.status;
    dataset.updatedAt = nowIso();
    saveState();
    return sendJson(res, 200, { dataset });
  }

  if (method === "POST" && url.pathname === "/api/training") {
    const result = startTraining({ payload: await readJson(req) });
    if (result.error) return sendJson(res, 400, result);
    return sendJson(res, 201, result);
  }

  if (method === "POST" && url.pathname === "/api/generation") {
    const result = startGeneration({ payload: await readJson(req), storage });
    if (result.error) return sendJson(res, 400, result);
    return sendJson(res, 201, result);
  }

  if (method === "GET" && url.pathname.startsWith("/api/assets/")) {
    const key = decodeURIComponent(url.pathname.replace("/api/assets/", ""));
    const asset = getState().assets.find((entry) => entry.key === key);
    if (!asset) return sendJson(res, 404, { error: "Asset not found" });
    const [, meta = "", payload = ""] = asset.dataUrl.match(/^data:([^;]+);base64,(.*)$/) || [];
    res.writeHead(200, {
      "Content-Type": meta || asset.contentType || "application/octet-stream",
      "Cache-Control": "public, max-age=31536000"
    });
    res.end(Buffer.from(payload, "base64"));
    return;
  }

  return sendJson(res, 404, { error: "Route not found" });
}

function publicState() {
  const state = getState();
  return {
    users: state.users,
    datasets: state.datasets.map(({ rawPhotos, ...dataset }) => ({
      ...dataset,
      rawPhotoCount: rawPhotos.length
    })),
    jobs: state.jobs,
    loras: state.loras,
    generationTasks: state.generationTasks,
    config: {
      storageDriver: config.storageDriver,
      workerDriver: config.workerDriver,
      comfyDriver: config.comfyDriver,
      publicStorageBaseUrl: config.publicStorageBaseUrl
    }
  };
}

async function readJson(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function sendJson(res, status, payload) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(JSON.stringify(payload));
}

function serveStatic(req, res) {
  const url = new URL(req.url, config.appOrigin);
  const requestPath = url.pathname === "/" ? "/index.html" : url.pathname;
  const filePath = path.normalize(path.join(publicDir, requestPath));
  if (!filePath.startsWith(publicDir)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404);
    res.end("Not found");
    return;
  }
  const ext = path.extname(filePath);
  const types = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml"
  };
  res.writeHead(200, {
    "Content-Type": types[ext] || "application/octet-stream"
  });
  fs.createReadStream(filePath).pipe(res);
}
