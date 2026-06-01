import { id, nowIso, upsert } from "../store.js";
import { config } from "../config.js";

export function createStorage() {
  return new MockStorage();
}

class MockStorage {
  async putDataUrl({ key, dataUrl, contentType, label }) {
    const asset = {
      id: id("asset"),
      key,
      label,
      contentType,
      dataUrl,
      httpsUrl: `${config.publicStorageBaseUrl.replace(/\/$/, "")}/${key}`,
      localUrl: `/api/assets/${encodeURIComponent(key)}`,
      createdAt: nowIso()
    };
    upsert("assets", asset);
    return asset;
  }

  async createSignedUpload({ key, contentType }) {
    return {
      key,
      contentType,
      method: "POST",
      url: "/api/uploads/mock",
      publicUrl: `${config.publicStorageBaseUrl.replace(/\/$/, "")}/${key}`,
      headers: {}
    };
  }
}
