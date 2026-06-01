# Cloudflare 部署下一步

你的域名是 `xellsun.com`，建议使用：

- PWA 前端：`https://app.xellsun.com`
- Worker API：`https://api.xellsun.com`
- R2 图片：`https://img.xellsun.com`

## 0. 先处理密钥

不要把 R2 Secret 写进 GitHub 或前端代码。你已经在聊天里贴过一次，部署成功后建议在 Cloudflare 重新创建一组 R2 凭据并删除旧凭据。

## 1. 先让代码上 GitHub

在 GitHub 创建一个空仓库，例如：

```txt
face-lora-studio
```

然后在本机项目目录执行：

```bash
git init
git add .
git commit -m "Initial Face LoRA Studio prototype"
git branch -M main
git remote add origin git@github.com:YOUR_NAME/face-lora-studio.git
git push -u origin main
```

如果你不想立刻用 GitHub，也可以先用 Wrangler 直接上传 Pages：

```bash
npx wrangler pages project create face-lora-studio --production-branch main
npx wrangler pages deploy public --project-name face-lora-studio
```

## 2. 创建 D1 数据库

```bash
npx wrangler login
npx wrangler d1 create face-lora-db
```

命令输出里会有 `database_id`。复制它，然后：

```bash
cp worker/wrangler.example.toml worker/wrangler.toml
```

把 `worker/wrangler.toml` 里的 `database_id` 替换成真实值。

初始化表：

```bash
npx wrangler d1 execute face-lora-db --file=worker/schema.sql --remote
```

## 3. 部署 Worker API

进入 `worker/wrangler.toml` 所在配置后部署：

```bash
npx wrangler deploy --config worker/wrangler.toml
```

部署完成后，在 Cloudflare Dashboard 给这个 Worker 添加自定义域名：

```txt
api.xellsun.com
```

## 4. 部署 Pages 前端

先把前端配置指向 Worker：

```bash
cp public/config.production.example.js public/config.js
```

然后二选一：

### 方式 A：通过 GitHub 自动部署

Cloudflare Dashboard → Workers & Pages → Create application → Pages → Connect to Git → 选择 `face-lora-studio` 仓库。

构建设置：

```txt
Build command: 留空
Build output directory: public
```

部署后给 Pages 项目绑定：

```txt
app.xellsun.com
```

### 方式 B：本地直接上传

```bash
npx wrangler pages deploy public --project-name face-lora-studio
```

然后在 Pages 项目里绑定：

```txt
app.xellsun.com
```

## 5. 检查 R2

你已经创建：

```txt
bucket: face-lora-assets
public domain: https://img.xellsun.com
```

当前 Worker 使用 R2 binding，不需要把 R2 Secret 写进 Worker。`worker/wrangler.toml` 里的这段会把 bucket 绑定到 `env.ASSETS`：

```toml
[[r2_buckets]]
binding = "ASSETS"
bucket_name = "face-lora-assets"
```

## 6. 验证

打开：

```txt
https://app.xellsun.com
```

确认：

1. 页面能加载。
2. 上传照片后，R2 中出现 `datasets/...` 文件。
3. 训练按钮能创建 LoRA 记录。
4. 出图后，R2 中出现 `generated/...` 文件。
5. 画廊图片 URL 是 `https://img.xellsun.com/...`。

当前 Worker 仍是云端 mock 流程，下一步才接 RunPod 真实裁脸、AI Toolkit 和 ComfyUI。
