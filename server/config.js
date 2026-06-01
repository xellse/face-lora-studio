import fs from "node:fs";
import path from "node:path";

const envPath = path.join(process.cwd(), ".env");

if (fs.existsSync(envPath)) {
  const lines = fs.readFileSync(envPath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const index = trimmed.indexOf("=");
    if (index === -1) continue;
    const key = trimmed.slice(0, index).trim();
    const value = trimmed.slice(index + 1).trim();
    if (!(key in process.env)) process.env[key] = value;
  }
}

export const config = {
  port: Number(process.env.PORT || 4173),
  appOrigin: process.env.APP_ORIGIN || "http://localhost:4173",
  storageDriver: process.env.STORAGE_DRIVER || "mock",
  publicStorageBaseUrl: process.env.PUBLIC_STORAGE_BASE_URL || "https://mock-r2.local",
  workerDriver: process.env.WORKER_DRIVER || "mock",
  comfyDriver: process.env.COMFY_DRIVER || "mock",
  comfyBaseUrl: process.env.COMFY_BASE_URL || "http://127.0.0.1:8188",
  runpodWorkspace: process.env.RUNPOD_WORKSPACE || "/workspace"
};
