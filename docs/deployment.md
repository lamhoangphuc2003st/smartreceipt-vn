# Deployment Guide — SmartReceipt VN

## Overview

| Component | Platform | Notes |
|-----------|----------|-------|
| Backend API | Railway | Auto-deploy from GitHub, free tier |
| Frontend UI | Hugging Face Spaces | Streamlit, free tier |
| CI/CD | GitHub Actions | Test → Build → Deploy on push to main |

---

## 1. Local Development with Docker

### Prerequisites
- Docker Desktop running
- `.env` file with `Gemini_API_Key` (copy from `.env.example`)

### Run both services
```bash
# Start API + Streamlit
docker-compose up --build

# API docs:      http://localhost:8000/docs
# Streamlit UI:  http://localhost:8501
```

### API only
```bash
docker build -t smartreceipt-vn-api .
docker run -p 8000:8000 --env-file .env smartreceipt-vn-api
```

---

## 2. Deploy Backend to Railway

### Step 1: Create Railway account
1. Đăng ký tại https://railway.app (dùng GitHub OAuth)
2. Create new project → **Deploy from GitHub repo**
3. Chọn repo `smartreceipt-vn`

### Step 2: Configure environment variables
Trong Railway dashboard → Variables tab, thêm:

```
Gemini_API_Key    = <your-gemini-api-key>
PHOBERT_MODEL     = models/phobert-expense-v1
DISABLE_PHOBERT   = 0
LOG_LEVEL         = INFO
```

> **Note:** Railway tự detect `Dockerfile` và build từ đó.
> `PORT` được Railway set tự động — không cần cấu hình.

### Step 3: Get public URL
Railway sẽ cấp URL dạng: `https://smartreceipt-vn-production.up.railway.app`

Kiểm tra:
```bash
curl https://<your-app>.up.railway.app/health
```

### Step 4: Setup CI/CD via GitHub Actions
Thêm các secrets vào GitHub repo → Settings → Secrets and variables → Actions:

```
RAILWAY_TOKEN       = <từ Railway dashboard → Account → Tokens>
RAILWAY_SERVICE_ID  = <từ Railway dashboard → Service → Settings>
RAILWAY_PUBLIC_URL  = https://<your-app>.up.railway.app
```

---

## 3. Deploy Frontend to Hugging Face Spaces

### Step 1: Tạo HF account
1. Đăng ký tại https://huggingface.co
2. Create new Space → **Streamlit** SDK
3. Space name: `smartreceipt-vn`

### Step 2: Push hf_spaces/ content
```bash
# Clone Space repo
git clone https://huggingface.co/spaces/<username>/smartreceipt-vn hf-space-temp

# Copy files
cp hf_spaces/app.py           hf-space-temp/app.py
cp hf_spaces/requirements.txt hf-space-temp/requirements.txt
cp hf_spaces/README.md        hf-space-temp/README.md   # overwrite

# Push
cd hf-space-temp
git add .
git commit -m "Deploy SmartReceipt VN frontend"
git push

cd .. && rm -rf hf-space-temp
```

### Step 3: Add secret in HF Spaces
HF Space → Settings → Repository secrets:
```
API_BASE_URL = https://<your-app>.up.railway.app
```

### Step 4: Verify
HF Space URL: `https://huggingface.co/spaces/<username>/smartreceipt-vn`

---

## 4. GitHub Actions CI/CD

Workflow file: `.github/workflows/ci.yml`

| Event | Jobs |
|-------|------|
| PR to main | `test` only |
| Push to main | `test` → `docker-build` → `deploy` |

**Required secrets** (GitHub repo → Settings → Secrets):
```
RAILWAY_TOKEN
RAILWAY_SERVICE_ID
RAILWAY_PUBLIC_URL
```

---

## 5. Monitoring

### Railway logs
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Stream logs
railway logs --tail
```

### Health endpoint
```bash
curl https://<your-app>.up.railway.app/health
# Expected: {"status":"ok","version":"0.9.0","components":{...}}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Gemini_API_Key not found` | Kiểm tra Railway Variables tab |
| PhoBERT fails to load | Set `DISABLE_PHOBERT=1` để dùng keyword fallback |
| Container OOM | Railway free tier có 512MB RAM — PhoBERT cần ~300MB. Nếu OOM, set `DISABLE_PHOBERT=1` |
| 413 on Railway | Railway default max request size là 100MB — OK cho ảnh hoá đơn |
| HF Space can't connect to API | Kiểm tra `API_BASE_URL` secret và CORS settings trong `api/main.py` |
