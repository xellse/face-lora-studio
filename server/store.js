import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

const dataDir = path.join(process.cwd(), ".data");
const dbPath = path.join(dataDir, "db.json");

const initialState = {
  users: [{ id: "local-user", name: "Internal User" }],
  datasets: [],
  jobs: [],
  loras: [],
  generationTasks: [],
  assets: []
};

let state = loadState();

function loadState() {
  fs.mkdirSync(dataDir, { recursive: true });
  if (!fs.existsSync(dbPath)) return structuredClone(initialState);
  try {
    return { ...structuredClone(initialState), ...JSON.parse(fs.readFileSync(dbPath, "utf8")) };
  } catch {
    return structuredClone(initialState);
  }
}

export function saveState() {
  fs.mkdirSync(dataDir, { recursive: true });
  fs.writeFileSync(dbPath, JSON.stringify(state, null, 2));
}

export function getState() {
  return state;
}

export function nowIso() {
  return new Date().toISOString();
}

export function id(prefix) {
  return `${prefix}_${randomUUID().slice(0, 8)}`;
}

export function upsert(collectionName, item) {
  const collection = state[collectionName];
  const index = collection.findIndex((entry) => entry.id === item.id);
  if (index >= 0) collection[index] = item;
  else collection.push(item);
  saveState();
  return item;
}

export function patch(collectionName, idValue, changes) {
  const collection = state[collectionName];
  const item = collection.find((entry) => entry.id === idValue);
  if (!item) return null;
  Object.assign(item, changes, { updatedAt: nowIso() });
  saveState();
  return item;
}
