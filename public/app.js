const app = document.querySelector("#app");

const state = {
  tab: "upload",
  server: null,
  selectedFiles: [],
  previews: [],
  uploadDraft: {},
  trainingDraft: {},
  generationDraft: {},
  faceDrafts: {},
  toast: "",
  refreshTimer: null,
  refreshing: false
};

let toastTimer = null;

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
    navigator.serviceWorker.register("/sw.js").then((registration) => registration.update()).catch(() => {});
  }
  await refresh({ force: true });
}

async function refresh(options = {}) {
  const { force = false, auto = false } = options;
  if (state.refreshing) return;
  if (auto && isEditing()) {
    scheduleNextRefresh();
    return;
  }
  state.refreshing = true;
  try {
    state.server = await api("/api/state");
    if (force || !isEditing()) render();
  } catch (error) {
    showToast(error.message);
  } finally {
    state.refreshing = false;
    scheduleNextRefresh();
  }
}

function scheduleNextRefresh() {
  if (state.refreshTimer) clearTimeout(state.refreshTimer);
  const data = state.server || emptyServer();
  const delay = activeJobs(data.jobs) ? 5000 : 30000;
  state.refreshTimer = setTimeout(() => refresh({ auto: true }), delay);
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
    </main>
  `;

  bindGlobalEvents();
  bindTabEvents();
  renderToast();
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
  const draft = {
    name: state.uploadDraft.name ?? `Portrait dataset ${data.datasets.length + 1}`,
    triggerWord: state.uploadDraft.triggerWord ?? "person_lora",
    cropSize: state.uploadDraft.cropSize ?? "1024"
  };
  return `
    <section class="two-col">
      <form id="uploadForm" class="panel">
        <div class="panel-head">
          <h2 class="panel-title">上传人像照片</h2>
          <span class="tag processing">1024 crop</span>
        </div>
        <div class="panel-body">
          <div class="form-grid">
            <label>数据集名称<input name="name" value="${escapeHtml(draft.name)}" required /></label>
            <label>触发词<input name="triggerWord" value="${escapeHtml(draft.triggerWord)}" required /></label>
            <label>裁切尺寸
              <select name="cropSize">
                <option value="1024" ${draft.cropSize === "1024" ? "selected" : ""}>1024 x 1024</option>
                <option value="768" ${draft.cropSize === "768" ? "selected" : ""}>768 x 768</option>
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
  const latest = data.datasets[0];
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
  const draftKey = `${datasetId}:${face.id}`;
  const draft = state.faceDrafts[draftKey] ?? { caption: face.caption, status: face.status };
  return `
    <article class="item face-item">
      <img class="thumb" src="${face.localUrl}" alt="" />
      <form data-face-form="${datasetId}:${face.id}" class="grid">
        <div class="split">
          <span class="tag ${face.status}">${face.status}</span>
          <button class="ghost" type="submit">保存</button>
        </div>
        <textarea name="caption">${escapeHtml(draft.caption)}</textarea>
        <select name="status">
          <option value="approved" ${draft.status === "approved" ? "selected" : ""}>approved</option>
          <option value="excluded" ${draft.status === "excluded" ? "selected" : ""}>excluded</option>
        </select>
      </form>
    </article>
  `;
}

function trainView(data) {
  const readyDatasets = data.datasets.filter((dataset) => dataset.status === "ready_for_training");
  const firstDataset = readyDatasets[0];
  const draft = {
    datasetId: state.trainingDraft.datasetId ?? firstDataset?.id ?? "",
    loraName: state.trainingDraft.loraName ?? `Portrait LoRA ${data.loras.length + 1}`,
    triggerWord: state.trainingDraft.triggerWord ?? firstDataset?.triggerWord ?? "person_lora",
    baseModel: state.trainingDraft.baseModel ?? "z-image-base",
    steps: state.trainingDraft.steps ?? "3000",
    learningRate: state.trainingDraft.learningRate ?? "1e-4",
    rank: state.trainingDraft.rank ?? "32",
    alpha: state.trainingDraft.alpha ?? "32",
    convRank: state.trainingDraft.convRank ?? "0",
    convAlpha: state.trainingDraft.convAlpha ?? "0",
    repeats: state.trainingDraft.repeats ?? "10",
    resolution: state.trainingDraft.resolution ?? "1024",
    saveEvery: state.trainingDraft.saveEvery ?? "250",
    captionDropoutRate: state.trainingDraft.captionDropoutRate ?? "0.05",
    batchSize: state.trainingDraft.batchSize ?? "1",
    gradientAccumulation: state.trainingDraft.gradientAccumulation ?? "1",
    optimizer: state.trainingDraft.optimizer ?? "adamw8bit",
    timestepType: state.trainingDraft.timestepType ?? "weighted",
    contentOrStyle: state.trainingDraft.contentOrStyle ?? "balanced",
    weightDecay: state.trainingDraft.weightDecay ?? "0.0001",
    dtype: state.trainingDraft.dtype ?? "bf16",
    saveDtype: state.trainingDraft.saveDtype ?? "float16",
    qtype: state.trainingDraft.qtype ?? "qfloat8",
    qtypeTe: state.trainingDraft.qtypeTe ?? "qfloat8",
    sampleSteps: state.trainingDraft.sampleSteps ?? "40",
    guidanceScale: state.trainingDraft.guidanceScale ?? "5",
    cacheLatentsToDisk: state.trainingDraft.cacheLatentsToDisk ?? "true",
    cacheTextEmbeddings: state.trainingDraft.cacheTextEmbeddings ?? "true",
    gradientCheckpointing: state.trainingDraft.gradientCheckpointing ?? "true",
    trainTextEncoder: state.trainingDraft.trainTextEncoder ?? "false",
    quantize: state.trainingDraft.quantize ?? "true",
    quantizeTe: state.trainingDraft.quantizeTe ?? "true",
    lowVram: state.trainingDraft.lowVram ?? "false",
    layerOffloading: state.trainingDraft.layerOffloading ?? "false",
    disableSampling: state.trainingDraft.disableSampling ?? "false",
    bypassGuidanceEmbedding: state.trainingDraft.bypassGuidanceEmbedding ?? "false",
    useEma: state.trainingDraft.useEma ?? "false",
    emaDecay: state.trainingDraft.emaDecay ?? "0.99"
  };
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
              <select name="datasetId" required>${readyDatasets.map((dataset) => `<option value="${dataset.id}" ${draft.datasetId === dataset.id ? "selected" : ""}>${escapeHtml(dataset.name)}</option>`).join("")}</select>
            </label>
            <label>LoRA 名称<input name="loraName" value="${escapeHtml(draft.loraName)}" required /></label>
            <label>触发词<input name="triggerWord" value="${escapeHtml(draft.triggerWord)}" required /></label>
            <label>基础模型
              <select name="baseModel">
                <option value="z-image-base" ${draft.baseModel === "z-image-base" ? "selected" : ""}>Z-Image Base</option>
              </select>
            </label>
            <label>训练步数<input name="steps" type="number" value="${escapeHtml(draft.steps)}" min="1000" step="100" /></label>
            <label>学习率<input name="learningRate" value="${escapeHtml(draft.learningRate)}" /></label>
            <label>Rank<input name="rank" type="number" value="${escapeHtml(draft.rank)}" min="16" max="128" /></label>
            <label>Alpha<input name="alpha" type="number" value="${escapeHtml(draft.alpha)}" min="1" max="128" /></label>
            <label>Conv Rank<input name="convRank" type="number" value="${escapeHtml(draft.convRank)}" min="0" max="128" /></label>
            <label>Conv Alpha<input name="convAlpha" type="number" value="${escapeHtml(draft.convAlpha)}" min="0" max="128" /></label>
            <label>图片重复<input name="repeats" type="number" value="${escapeHtml(draft.repeats)}" min="1" max="50" /></label>
            <label>训练分辨率<input name="resolution" type="number" value="${escapeHtml(draft.resolution)}" min="512" max="1536" step="128" /></label>
            <label>保存间隔<input name="saveEvery" type="number" value="${escapeHtml(draft.saveEvery)}" min="50" step="50" /></label>
            <label>Caption Dropout<input name="captionDropoutRate" type="number" value="${escapeHtml(draft.captionDropoutRate)}" min="0" max="0.3" step="0.01" /></label>
            <label>Batch Size<input name="batchSize" type="number" value="${escapeHtml(draft.batchSize)}" min="1" max="4" /></label>
            <label>梯度累积<input name="gradientAccumulation" type="number" value="${escapeHtml(draft.gradientAccumulation)}" min="1" max="8" /></label>
            <label>Optimizer
              <select name="optimizer">
                ${["adamw8bit", "adamw", "adamwfp8", "adafactor", "Prodigy"].map((option) => `<option value="${option}" ${draft.optimizer === option ? "selected" : ""}>${option}</option>`).join("")}
              </select>
            </label>
            <label>Timestep
              <select name="timestepType">
                ${["weighted", "sigmoid", "linear", "uniform"].map((option) => `<option value="${option}" ${draft.timestepType === option ? "selected" : ""}>${option}</option>`).join("")}
              </select>
            </label>
            <label>Content/Style
              <select name="contentOrStyle">
                ${["balanced", "content", "style"].map((option) => `<option value="${option}" ${draft.contentOrStyle === option ? "selected" : ""}>${option}</option>`).join("")}
              </select>
            </label>
            <label>Weight Decay<input name="weightDecay" type="number" value="${escapeHtml(draft.weightDecay)}" min="0" max="0.1" step="0.0001" /></label>
            <label>训练精度
              <select name="dtype">${["bf16", "fp16", "float32"].map((option) => `<option value="${option}" ${draft.dtype === option ? "selected" : ""}>${option}</option>`).join("")}</select>
            </label>
            <label>保存精度
              <select name="saveDtype">${["float16", "bf16", "fp16", "float32"].map((option) => `<option value="${option}" ${draft.saveDtype === option ? "selected" : ""}>${option}</option>`).join("")}</select>
            </label>
            <label>Transformer 量化
              <select name="qtype">${["qfloat8", "float8", "qint8", "none"].map((option) => `<option value="${option}" ${draft.qtype === option ? "selected" : ""}>${option}</option>`).join("")}</select>
            </label>
            <label>Text Encoder 量化
              <select name="qtypeTe">${["qfloat8", "float8", "qint8", "none"].map((option) => `<option value="${option}" ${draft.qtypeTe === option ? "selected" : ""}>${option}</option>`).join("")}</select>
            </label>
            <label>预览步数<input name="sampleSteps" type="number" value="${escapeHtml(draft.sampleSteps)}" min="15" max="80" /></label>
            <label>预览 CFG<input name="guidanceScale" type="number" value="${escapeHtml(draft.guidanceScale)}" min="1" max="12" step="0.1" /></label>
            ${trainingCheckbox("cacheLatentsToDisk", "缓存 Latents", draft.cacheLatentsToDisk)}
            ${trainingCheckbox("cacheTextEmbeddings", "缓存文本 Embeddings", draft.cacheTextEmbeddings)}
            ${trainingCheckbox("gradientCheckpointing", "Gradient Checkpointing", draft.gradientCheckpointing)}
            ${trainingCheckbox("trainTextEncoder", "训练 Text Encoder", draft.trainTextEncoder)}
            ${trainingCheckbox("quantize", "量化 Transformer", draft.quantize)}
            ${trainingCheckbox("quantizeTe", "量化 Text Encoder", draft.quantizeTe)}
            ${trainingCheckbox("lowVram", "Low VRAM", draft.lowVram)}
            ${trainingCheckbox("layerOffloading", "Layer Offloading", draft.layerOffloading)}
            ${trainingCheckbox("disableSampling", "关闭训练预览", draft.disableSampling)}
            ${trainingCheckbox("bypassGuidanceEmbedding", "Bypass Guidance Embedding", draft.bypassGuidanceEmbedding)}
            ${trainingCheckbox("useEma", "启用 EMA", draft.useEma)}
            <label>EMA Decay<input name="emaDecay" type="number" value="${escapeHtml(draft.emaDecay)}" min="0.9" max="0.9999" step="0.001" /></label>
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

function trainingCheckbox(name, label, value) {
  const checked = value === true || value === "true";
  return `<label class="check-label"><input type="hidden" name="${name}" value="false" /><span><input name="${name}" type="checkbox" value="true" ${checked ? "checked" : ""} /> ${label}</span></label>`;
}

function generateView(data) {
  const readyLoras = data.loras.filter((lora) => lora.status === "ready");
  const draft = {
    loraId: state.generationDraft.loraId ?? readyLoras[0]?.id ?? "",
    prompt: state.generationDraft.prompt ?? "editorial portrait, clean background, cinematic light, sharp details",
    negativePrompt: state.generationDraft.negativePrompt ?? "blur, low quality, distorted face, extra fingers",
    count: state.generationDraft.count ?? "4",
    width: state.generationDraft.width ?? "1024",
    height: state.generationDraft.height ?? "1024",
    seed: state.generationDraft.seed ?? "",
    steps: state.generationDraft.steps ?? "40",
    cfg: state.generationDraft.cfg ?? "5",
    sampler: state.generationDraft.sampler ?? "euler",
    scheduler: state.generationDraft.scheduler ?? "normal",
    loraWeight: state.generationDraft.loraWeight ?? "0.85"
  };
  const samplerOptions = [
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "uni_pc",
    "uni_pc_bh2"
  ];
  const schedulerOptions = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "turbo"];
  return `
    <section class="two-col">
      <form id="generationForm" class="panel">
        <div class="panel-head">
          <h2 class="panel-title">ComfyUI 出图</h2>
          <span class="tag ready">${readyLoras.length} usable</span>
        </div>
        <div class="panel-body">
          <label>LoRA
            <select name="loraId" required>${readyLoras.map((lora) => `<option value="${lora.id}" ${draft.loraId === lora.id ? "selected" : ""}>${escapeHtml(lora.name)} · ${escapeHtml(lora.triggerWord)}</option>`).join("")}</select>
          </label>
          <label>正向提示词<textarea name="prompt" required>${escapeHtml(draft.prompt)}</textarea></label>
          <label>负向提示词<textarea name="negativePrompt">${escapeHtml(draft.negativePrompt)}</textarea></label>
          <div class="form-grid">
            <label>数量<input name="count" type="number" value="${escapeHtml(draft.count)}" min="1" max="12" /></label>
            <label>宽度<input name="width" type="number" value="${escapeHtml(draft.width)}" min="512" max="2048" step="64" /></label>
            <label>高度<input name="height" type="number" value="${escapeHtml(draft.height)}" min="512" max="${escapeHtml(draft.width)}" step="64" /></label>
            <label>Seed<input name="seed" type="number" value="${escapeHtml(draft.seed)}" placeholder="random" /></label>
            <label>Steps<input name="steps" type="number" value="${escapeHtml(draft.steps)}" min="15" max="80" /></label>
            <label>CFG<input name="cfg" type="number" value="${escapeHtml(draft.cfg)}" min="1" max="12" step="0.1" /></label>
            <label>Sampler
              <select name="sampler">
                ${samplerOptions.map((option) => `<option value="${option}" ${draft.sampler === option ? "selected" : ""}>${option}</option>`).join("")}
              </select>
            </label>
            <label>Scheduler
              <select name="scheduler">
                ${schedulerOptions.map((option) => `<option value="${option}" ${draft.scheduler === option ? "selected" : ""}>${option}</option>`).join("")}
              </select>
            </label>
            <label>LoRA 权重<input name="loraWeight" type="number" value="${escapeHtml(draft.loraWeight)}" min="0" max="1.5" step="0.05" /></label>
          </div>
          <button type="submit" ${readyLoras.length ? "" : "disabled"}>开始生成</button>
        </div>
      </form>
      <section class="panel">
        <div class="panel-head">
          <h2 class="panel-title">最近任务</h2>
        </div>
        <div class="panel-body">${taskList(data.generationTasks.slice(0, 4))}</div>
      </section>
    </section>
  `;
}

function galleryView(data) {
  const tasks = data.generationTasks;
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
  const recent = jobs.slice(0, 8);
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
  return `<div class="lora-list">${loras.map((lora) => `
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

  document.querySelector("[data-refresh]")?.addEventListener("click", () => refresh({ force: true }));

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
      await refresh({ force: true });
    });
  });
}

function bindTabEvents() {
  bindDraftForm("#uploadForm", "uploadDraft");
  bindDraftForm("#trainingForm", "trainingDraft");
  bindDraftForm("#generationForm", "generationDraft");

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
    state.uploadDraft = {};
    state.tab = "review";
    showToast("Dataset processing queued");
    await refresh({ force: true });
  });

  document.querySelectorAll("[data-face-form]").forEach((form) => {
    form.addEventListener("input", () => {
      const [datasetId, faceId] = form.dataset.faceForm.split(":");
      state.faceDrafts[`${datasetId}:${faceId}`] = Object.fromEntries(new FormData(form));
    });
    form.addEventListener("change", () => {
      const [datasetId, faceId] = form.dataset.faceForm.split(":");
      state.faceDrafts[`${datasetId}:${faceId}`] = Object.fromEntries(new FormData(form));
    });
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
      delete state.faceDrafts[`${datasetId}:${faceId}`];
      showToast("Caption saved");
      await refresh({ force: true });
    });
  });

  document.querySelector("#trainingForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await api("/api/training", { method: "POST", body: Object.fromEntries(new FormData(event.currentTarget)) });
    state.trainingDraft = {};
    showToast("Training queued");
    await refresh({ force: true });
  });

  document.querySelector("#generationForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = normalizeGenerationBody(Object.fromEntries(new FormData(event.currentTarget)));
    if (body.height > body.width) {
      showToast("高度不能大于宽度");
      return;
    }
    await api("/api/generation", { method: "POST", body });
    state.generationDraft = {};
    state.tab = "gallery";
    showToast("Generation queued");
    await refresh({ force: true });
  });
}

function normalizeGenerationBody(body) {
  const width = Number(body.width || 1024);
  const height = Number(body.height || 1024);
  return {
    ...body,
    count: Number(body.count || 1),
    width,
    height,
    steps: Math.max(15, Number(body.steps || 40)),
    cfg: Number(body.cfg || 5),
    loraWeight: Number(body.loraWeight || 0.85)
  };
}

function bindDraftForm(selector, draftKey) {
  const form = document.querySelector(selector);
  if (!form) return;
  const update = () => {
    state[draftKey] = Object.fromEntries(new FormData(form));
  };
  form.addEventListener("input", update);
  form.addEventListener("change", update);
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
  return jobs.filter((job) => ["queued", "preparing", "processing", "running", "training"].includes(job.status)).length;
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
  renderToast();
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    state.toast = "";
    renderToast();
  }, 2400);
}

function renderToast() {
  let root = document.querySelector("#toast-root");
  if (!root) {
    root = document.createElement("div");
    root.id = "toast-root";
    document.body.appendChild(root);
  }
  root.innerHTML = state.toast ? `<div class="toast">${escapeHtml(state.toast)}</div>` : "";
}

function isEditing() {
  const element = document.activeElement;
  if (!element || !app.contains(element)) return false;
  if (element.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
