# Face LoRA Mobile PWA

This is a runnable first-pass implementation of the planned mobile PWA + backend for:

1. portrait photo upload
2. cloud face crop + caption preparation
3. AI Toolkit LoRA training orchestration
4. ComfyUI generation with selectable LoRAs
5. task-folder gallery with generated HTTPS-style image URLs

The default implementation runs fully locally with mock adapters, so you can validate the product flow before wiring real RunPod, AI Toolkit, ComfyUI, and S3/R2 credentials.

## Run locally

```bash
npm run dev
```

Open `http://localhost:4173`.

## Smoke test

```bash
npm test
```

## Real service wiring points

- Storage: replace `server/services/storage.js` mock uploads with S3/R2 signed upload and object upload calls.
- RunPod / AI Toolkit: replace `server/services/workers.js` mock workflow with calls into your RunPod worker API or SSH/container service.
- ComfyUI: replace `server/services/comfy.js` mock generation with `/prompt`, WebSocket/history polling, and output upload.
- Vision captions: replace `captionFace` in `server/services/workers.js` with your configured vision-language model call.

Recommended RunPod paths are encoded in generated jobs:

- `/workspace/ai-toolkit`
- `/workspace/ComfyUI`
- `/workspace/ComfyUI/models/loras/{userId}/{loraId}.safetensors`
- `/workspace/jobs/{jobId}`

## Product defaults

- base model: Z-Image Base (`Tongyi-MAI/Z-Image`)
- crop size: 1024
- storage: S3/R2-style public HTTPS URLs
- user mode: single internal user named `local-user`
