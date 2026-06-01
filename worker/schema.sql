CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  trigger_word TEXT NOT NULL,
  crop_size INTEGER NOT NULL,
  status TEXT NOT NULL,
  raw_photo_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  status TEXT NOT NULL,
  caption TEXT NOT NULL,
  object_key TEXT NOT NULL,
  https_url TEXT NOT NULL,
  crop_size INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  status TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  dataset_id TEXT,
  lora_id TEXT,
  generation_task_id TEXT,
  message TEXT,
  logs TEXT NOT NULL DEFAULT '[]',
  payload TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loras (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  name TEXT NOT NULL,
  trigger_word TEXT NOT NULL,
  base_model TEXT NOT NULL,
  status TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  model_path TEXT,
  parameters TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS generation_tasks (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  lora_id TEXT NOT NULL,
  lora_name TEXT NOT NULL,
  status TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  folder_name TEXT NOT NULL,
  prompt TEXT NOT NULL,
  negative_prompt TEXT,
  settings TEXT NOT NULL,
  images TEXT NOT NULL DEFAULT '[]',
  message TEXT,
  comfy_prompt_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_faces_dataset ON faces(dataset_id);
CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at);
CREATE INDEX IF NOT EXISTS idx_loras_status ON loras(status);
CREATE INDEX IF NOT EXISTS idx_generation_tasks_created ON generation_tasks(created_at);
