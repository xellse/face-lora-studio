const baseUrl = process.env.APP_ORIGIN || "http://localhost:4173";

async function main() {
  const serverReady = await waitForServer();
  if (!serverReady) throw new Error(`Server is not reachable at ${baseUrl}`);

  const photos = Array.from({ length: 6 }, (_, index) => ({
    name: `portrait_${index + 1}.svg`,
    type: "image/svg+xml",
    size: 50_000 + index,
    dataUrl: svgDataUrl(index)
  }));

  const datasetResult = await post("/api/datasets", {
    name: "Smoke test dataset",
    triggerWord: "smoke_person",
    cropSize: 1024,
    photos
  });

  const dataset = await waitFor(
    () => get("/api/state").then((state) => state.datasets.find((entry) => entry.id === datasetResult.dataset.id)),
    (entry) => entry?.status === "ready_for_training",
    "dataset processing"
  );

  const trainingResult = await post("/api/training", {
    datasetId: dataset.id,
    loraName: "Smoke LoRA",
    triggerWord: "smoke_person",
    baseModel: "flux-dev",
    steps: 300,
    learningRate: "1e-4",
    rank: 16,
    repeats: 10,
    resolution: 1024,
    saveEvery: 150
  });

  const lora = await waitFor(
    () => get("/api/state").then((state) => state.loras.find((entry) => entry.id === trainingResult.lora.id)),
    (entry) => entry?.status === "ready",
    "training"
  );

  const generationResult = await post("/api/generation", {
    loraId: lora.id,
    prompt: "editorial portrait, clean background, cinematic light",
    negativePrompt: "blur, low quality",
    count: 2,
    width: 768,
    height: 768,
    steps: 18,
    cfg: 4.5,
    sampler: "euler",
    loraWeight: 0.85
  });

  const task = await waitFor(
    () => get("/api/state").then((state) => state.generationTasks.find((entry) => entry.id === generationResult.task.id)),
    (entry) => entry?.status === "completed" && entry.images.length === 2,
    "generation"
  );

  console.log(`Smoke test passed: ${dataset.faces.length} faces, LoRA ${lora.modelPath}, ${task.images.length} images`);
}

async function waitForServer() {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const res = await fetch(`${baseUrl}/api/state`);
      if (res.ok) return true;
    } catch {}
    await sleep(250);
  }
  return false;
}

async function waitFor(getter, predicate, label) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    const value = await getter();
    if (predicate(value)) return value;
    await sleep(300);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

async function get(path) {
  const res = await fetch(`${baseUrl}${path}`);
  if (!res.ok) throw new Error(`${path} failed with ${res.status}`);
  return res.json();
}

async function post(path, payload) {
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`${path} failed with ${res.status}: ${await res.text()}`);
  return res.json();
}

function svgDataUrl(index) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024"><rect width="100%" height="100%" fill="hsl(${index * 40},70%,60%)"/><circle cx="512" cy="390" r="180" fill="white"/><rect x="320" y="590" width="384" height="210" rx="96" fill="white"/></svg>`;
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
