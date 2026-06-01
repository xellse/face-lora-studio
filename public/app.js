const app = document.querySelector("#app");

const state = {
  tab: "upload",
  server: null,
  selectedFiles: [],
  previews: [],
  toast: ""
};

const tabs = [
  ["upload", "上传"],
  ["review", "样本"],
  ["train", "训练"],
  ["generate", "出图"],
  ["gallery", "画廊"]
];

init();

async function init() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
  await refresh();
  render();
  setInterval(refresh, 1200);
}

async function refresh() {
  try {
    state.server = await api("/api/state");
    render();
  } catch (error) {
    showToast(error.message);
  }
}

function render() {
  const data = state.server || emptyServer();
  app.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <img src="/icon.svg" alt="" />
          <div>
            <h1 class="brand-title">Face LoRA Studio</h1>
            <p class="brand-subtitle">${data.config.workerDriver} worker · ${data.config.comfyDriver} comfy · ${data.config.storageDriver} storage</p>
          </div>
        </div>
        <span class="status-pill">${activeJobs(data.jobs)} active</span>
      </header>
      <section class="content">
        <nav class="nav">${tabs.map(([id, label]) => `<button data-tab="${id}" class="${state.tab === id ? "active" : ""}">${label}</button>`).join("")}</nav>
        ${metrics(data)}
        ${route(data)}
      </section>
      ${state.toast ? `<div class="toast">${escapeHtml(state.toast)}</div>` : ""}
    </main>
  `;

  bindGlobalEvents();
  bindTabEvents();
}

function route(data) {
  if (state.tab === "upload") return uploadView(data);
  if (state.tab === "review") return reviewView(data);
  if (state.tab === "train") return trainView(data);
  if (state.tab === "generate") return generateView(data);
  return galleryView(data);
}

function metrics(data) {
  return `
    <section class="metric-strip" style="margin-bottom:14px">
      <div class="metric"><b>${data.datasets.length}</b><span>datasets</span></div>
      <div class="metric"><b>${data.loras.filter((lora) => lora.status === "ready").length}</b><span>ready LoRAs</span></div>
      <div class="metric"><b>${data.generationTasks.length}</b><span>task folders</span></div>
      <div class="metric"><b>${data.jobs.filter((job) => job.status === "failed").length}</b><span>failed jobs</span></div>
    </section>
  `;
}

function uploadView(data) {
  return `
    <section class="two-col">
      <form id="uploadForm" class="panel">
        <div class="panel-head">
          <h2 class="panel-title">上传人像照片</h2>
          <span class="tag processing">1024 crop</span>
        </div>
        <div class="panel-body">
          <div class="form-grid">
            <label>数据集名称<input name="name" value="Portrait dataset ${data.datasets.length + 1}" required /></label>
            <label>触发词<input name="triggerWord" value="person_lora" required /></label>
            <label>裁切尺寸
              <select name="cropSize">
                <option value="1024">1024 x 1024</option>
                <option value="768">768 x 768</option>
              </select>
            </label>
          </div>
          <div class="dropzone">
            <input id="photoInput" type="file" accept="image/*" multiple />
            <strong>${state.selectedFiles.length ? `${state.selectedFiles.length} photos selected` : "选择 10-30 张人像照片"}</strong>
          </div>
          ${state.previews.length ? `<div class="thumb-grid">${state.previews.map((src) => `<img class="thumb" src="${src}" alt="" />`).join("")}</div>` : ""}
          <button type="submit" ${state.selectedFiles.length ? "" : "disabled"}>创建裁脸任务</button>
        </div>
      </form>
      <section class="panel">
        <div class="panel-head">
          <h2 class="panel-title">任务状态</h2>
          <button class="secondary" data-refresh>刷新</button>
        </div>
        <div class="panel-body">${jobList(data.jobs)}</div>
      </section>
    </section>
  `;
}

function reviewView(data) {
  const latest = [...data.datasets].reverse()[0];
  if (!latest) return emptyPanel("样本预览", "还没有数据集");
  return `
    <section class="grid">
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">${escapeHtml(latest.name)}</h2>
            <span class="tag ${latest.status}">${latest.status}</span>
          </div>
          <span class="tag ready">${latest.faces.length} faces</span>
        </div>
        <div class="panel-body">
          ${latest.reviewItems.length ? `<div class="empty">${latest.reviewItems.length} 张照片需要人工确认或替换</div>` : ""}
          <div class="face-list">
            ${latest.faces.map((face) => faceItem(latest.id, face)).join("") || `<div class="empty">裁脸任务还在处理中</div>`}
          </div>
        </div>
      </section>
    </section>
  `;
}

function faceItem(datasetId, face) {
  return `
    <article class="item face-item">
      <img class="thumb" src="${face.localUrl}" alt="" />
      <form data-face-form="${datasetId}:${face.id}" class="grid">
        <div class="split">
          <span class="tag ${face.status}">${face.status}</span>
          <button class="ghost" type="submit">保存</button>
        </div>
        <textarea name="caption">${escapeHtml(face.caption)}</textarea>
        <select name="status">
          <option value="approved" ${face.status === "approved" ? "selected" : ""}>approved</option>
          <option value="excluded" ${face.status === "excluded" ? "selected" : ""}>excluded</option>
        </select>
      </form>
    </article>
  `;
}

function trainView(data) {
  const readyDatasets = data.datasets.filter((dataset) => dataset.status === "ready_for_training");
  return `
    <section class="two-col">
      <form id="trainingForm" class="panel">
        <div class="panel-head">
          <h2 class="panel-title">LoRA 训练</h2>
          <span class="tag training">AI Toolkit</span>
        </div>
        <div class="panel-body">
          <div class="form-grid">
            <label>数据集
              <select name="datasetId" required>${readyDatasets.map((dataset) => `<option value="${dataset.id}">${escapeHtml(dataset.name)}</option>`).join("")}</select>
            </label>
            <label>LoRA 名称<input name="loraName" value="Portrait LoRA ${data.loras.length + 1}" required /></label>
            <label>触发词<input name="triggerWord" value="${readyDatasets[0]?.triggerWord || "person_lora"}" required /></label>
            <label>基础模型
              <select name="baseModel">
                <option value="z-image-base" selected>Z-Image Base</option>
              </select>
            </label>
            <label>训练步数<input name="steps" type="number" value="3000" min="1000" step="100" /></label>
            <label>学习率<input name="learningRate" value="1e-4" /></label>
            <label>Rank<input name="rank" type="number" value="32" min="16" max="128" /></label>
            <label>图片重复<input name="repeats" type="number" value="10" min="1" max="50" /></label>
            <label>训练分辨率<input name="resolution" type="number" value="1024" min="512" max="1536" step="128" /></label>
            <label>保存间隔<input name="saveEvery" type="number" value="250" min="50" step="50" /></label>
          </div>
          <button type="submit" ${readyDatasets.length ? "" : "disabled"}>启动训练</button>
        </div>
      </form>
      <section class="panel">
        <div class="panel-head">
          <h2 class="panel-title">LoRA 列表</h2>
        </div>
        <div class="panel-body">${loraList(data.loras)}</div>
      </section>
    </section>
  `;
}

function generateView(data) {
  const readyLoras = data.loras.filter((lora) => lora.status === "ready");
  return `
    <section class="two-col">
      <form id="generationForm" class="panel">
        <div class="panel-head">
          <h2 class="panel-title">ComfyUI 出图</h2>
          <span class="tag ready">${readyLoras.length} usable</span>
        </div>
        <div class="panel-body">
          <label>LoRA
            <select name="loraId" required>${readyLoras.map((lora) => `<option value="${lora.id}">${escapeHtml(lora.name)} · ${escapeHtml(lora.triggerWord)}</option>`).join("")}</select>
          </label>
          <label>正向提示词<textarea name="prompt" required>editorial portrait, clean background, cinematic light, sharp details</textarea></label>
          <label>负向提示词<textarea name="negativePrompt">blur, low quality, distorted face, extra fingers</textarea></label>
          <div class="form-grid">
            <label>数量<input name="count" type="number" value="4" min="1" max="12" /></label>
            <label>宽度<input name="width" type="number" value="1024" min="512" max="1536" step="64" /></label>
            <label>高度<input name="height" type="number" value="1024" min="512" max="1536" step="64" /></label>
            <label>Seed<input name="seed" type="number" placeholder="random" /></label>
            <label>Steps<input name="steps" type="number" value="40" min="20" max="80" /></label>
            <label>CFG<input name="cfg" type="number" value="5" min="1" max="12" step="0.1" /></label>
            <label>Sampler
              <select name="sampler">
                <option value="euler">euler</option>
                <option value="dpmpp_2m">dpmpp_2m</option>
                <option value="dpmpp_sde">dpmpp_sde</option>
              </select>
            </label>
            <label>LoRA 权重<input name="loraWeight" type="number" value="0.85" min="0" max="1.5" step="0.05" /></label>
          </div>
          <button type="submit" ${readyLoras.length ? "" : "disabled"}>开始生成</button>
        </div>
      </form>
      <section class="panel">
        <div class="panel-head">
          <h2 class="panel-title">最近任务</h2>
        </div>
        <div class="panel-body">${taskList(data.generationTasks.slice(-4).reverse())}</div>
      </section>
    </section>
  `;
}

function galleryView(data) {
  const tasks = [...data.generationTasks].reverse();
  return `
    <section class="grid">
      ${tasks.map((task) => galleryTask(task)).join("") || emptyPanel("任务文件夹", "还没有生成任务")}
    </section>
  `;
}

function galleryTask(task) {
  return `
    <section class="panel gallery">
      <div class="panel-head">
        <div>
          <h2 class="panel-title">${escapeHtml(task.folderName)}</h2>
          <span class="tag ${task.status}">${task.status} · ${task.progress}%</span>
        </div>
        <button class="secondary" data-repeat="${task.id}" ${task.status === "completed" ? "" : "disabled"}>重跑</button>
      </div>
      <div class="panel-body">
        <p class="url">${escapeHtml(task.prompt || "")}</p>
        ${task.images?.length ? `
          <div class="image-strip">
            ${task.images.map((image) => `
              <figure class="generated">
                <img src="${image.localUrl}" alt="" />
                <figcaption class="row">
                  <span class="tag ready">seed ${image.seed}</span>
                  <button class="ghost" data-copy="${image.httpsUrl}">复制 URL</button>
                </figcaption>
                <p class="url">${escapeHtml(image.httpsUrl)}</p>
              </figure>
            `).join("")}
          </div>
        ` : `<div class="progress" style="--value:${task.progress}%"><span></span></div>`}
      </div>
    </section>
  `;
}

function jobList(jobs) {
  const recent = [...jobs].reverse().slice(0, 8);
  return `<div class="job-list">${recent.map((job) => `
    <article class="item">
      <div class="split">
        <strong>${job.type}</strong>
        <span class="tag ${job.status}">${job.status}</span>
      </div>
      <div class="progress" style="--value:${job.progress}%"><span></span></div>
      <span class="url">${escapeHtml(job.message || "")}</span>
    </article>
  `).join("") || `<div class="empty">暂无任务</div>`}</div>`;
}

function loraList(loras) {
  return `<div class="lora-list">${[...loras].reverse().map((lora) => `
    <article class="item">
      <div class="split">
        <strong>${escapeHtml(lora.name)}</strong>
        <span class="tag ${lora.status}">${lora.status}</span>
      </div>
      <div class="progress" style="--value:${lora.progress || 0}%"><span></span></div>
      <span class="url">${escapeHtml(lora.modelPath || lora.triggerWord || "")}</span>
    </article>
  `).join("") || `<div class="empty">暂无 LoRA</div>`}</div>`;
}

function taskList(tasks) {
  return `<div class="task-list">${tasks.map((task) => `
    <article class="item">
      <div class="split">
        <strong>${escapeHtml(task.folderName)}</strong>
        <span class="tag ${task.status}">${task.status}</span>
      </div>
      <div class="progress" style="--value:${task.progress || 0}%"><span></span></div>
      <span class="url">${escapeHtml(task.prompt || "")}</span>
    </article>
  `).join("") || `<div class="empty">暂无出图任务</div>`}</div>`;
}

function emptyPanel(title, message) {
  return `<section class="panel"><div class="panel-head"><h2 class="panel-title">${title}</h2></div><div class="panel-body"><div class="empty">${message}</div></div></section>`;
}

function bindGlobalEvents() {
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.tab = button.dataset.tab;
      render();
    });
  });

  document.querySelector("[data-refresh]")?.addEventListener("click", refresh);

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      await navigator.clipboard.writeText(button.dataset.copy);
      showToast("URL copied");
    });
  });

  document.querySelectorAll("[data-repeat]").forEach((button) => {
    button.addEventListener("click", async () => {
      const task = state.server.generationTasks.find((entry) => entry.id === button.dataset.repeat);
      if (!task) return;
      await api("/api/generation", {
        method: "POST",
        body: {
          loraId: task.loraId,
          prompt: task.prompt,
          negativePrompt: task.negativePrompt,
          ...task.settings,
          seed: ""
        }
      });
      state.tab = "gallery";
      showToast("Generation queued");
      await refresh();
    });
  });
}

function bindTabEvents() {
  document.querySelector("#photoInput")?.addEventListener("change", async (event) => {
    state.selectedFiles = [...event.target.files];
    state.previews = await Promise.all(state.selectedFiles.slice(0, 10).map(readFile));
    render();
  });

  document.querySelector("#uploadForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const uploadGroup = crypto.randomUUID();
    const uploadedPhotos = [];
    for (const [index, file] of state.selectedFiles.entries()) {
      showToast(`Uploading ${index + 1} of ${state.selectedFiles.length}`);
      uploadedPhotos.push(await uploadPhoto(file, uploadGroup, index));
    }
    await api("/api/datasets", {
      method: "POST",
      body: {
        name: form.get("name"),
        triggerWord: form.get("triggerWord"),
        cropSize: Number(form.get("cropSize")),
        uploadedPhotos
      }
    });
    state.selectedFiles = [];
    state.previews = [];
    state.tab = "review";
    showToast("Dataset processing queued");
    await refresh();
  });

  document.querySelectorAll("[data-face-form]").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const [datasetId, faceId] = form.dataset.faceForm.split(":");
      const values = new FormData(form);
      await api(`/api/datasets/${datasetId}/faces/${faceId}`, {
        method: "PATCH",
        body: {
          caption: values.get("caption"),
          status: values.get("status")
        }
      });
      showToast("Caption saved");
      await refresh();
    });
  });

  document.querySelector("#trainingForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await api("/api/training", { method: "POST", body: Object.fromEntries(new FormData(event.currentTarget)) });
    showToast("Training queued");
    await refresh();
  });

  document.querySelector("#generationForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await api("/api/generation", { method: "POST", body: Object.fromEntries(new FormData(event.currentTarget)) });
    state.tab = "gallery";
    showToast("Generation queued");
    await refresh();
  });
}

async function uploadPhoto(file, uploadGroup, index) {
  const body = new FormData();
  const safeName = file.name.replace(/[^a-zA-Z0-9._-]+/g, "_") || `portrait_${index + 1}.jpg`;
  const key = `raw/local-user/${uploadGroup}/${String(index + 1).padStart(4, "0")}_${safeName}`;
  body.set("key", key);
  body.set("file", file);
  const apiBaseUrl = window.FACE_LORA_CONFIG?.API_BASE_URL || "";
  const response = await fetch(`${apiBaseUrl}/api/uploads`, {
    method: "POST",
    body
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Upload failed");
  return {
    name: file.name,
    type: file.type,
    size: file.size,
    key: payload.key,
    httpsUrl: payload.httpsUrl
  };
}

async function api(path, options = {}) {
  const apiBaseUrl = window.FACE_LORA_CONFIG?.API_BASE_URL || "";
  const init = { method: options.method || "GET", headers: {} };
  if (options.body) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(`${apiBaseUrl}${path}`, init);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function activeJobs(jobs) {
  return jobs.filter((job) => ["queued", "preparing", "running"].includes(job.status)).length;
}

function emptyServer() {
  return {
    datasets: [],
    jobs: [],
    loras: [],
    generationTasks: [],
    config: { workerDriver: "mock", comfyDriver: "mock", storageDriver: "mock" }
  };
}

function showToast(message) {
  state.toast = message;
  render();
  setTimeout(() => {
    state.toast = "";
    render();
  }, 2400);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
