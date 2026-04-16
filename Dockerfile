# ──────────────────────────────────────────────────────────────────────────────
# SmartReceipt VN — Backend Dockerfile
#
# Build:  docker build -t smartreceipt-vn-api .
# Run:    docker run -p 8000:8000 -e Gemini_API_Key=<key> smartreceipt-vn-api
#
# Railway auto-detects this file and deploys from it.
# ──────────────────────────────────────────────────────────────────────────────

# --- Stage 1: dependency builder ---
FROM python:3.10-slim AS builder

WORKDIR /build

# System libs required by OpenCV (headless) and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix we'll copy to runtime stage
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install ".[standard]" 2>/dev/null || \
    pip install --no-cache-dir --prefix=/install \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "slowapi>=0.1.9" \
        "python-multipart>=0.0.9" \
        "pydantic>=2.0" \
        "google-genai" \
        "opencv-python-headless" \
        "Pillow" \
        "numpy<2" \
        "rapidfuzz" \
        "transformers>=4.40" \
        # CPU-only torch — much smaller image (~800 MB vs ~3 GB with CUDA)
        # torch 2.4+ required by transformers>=4.40, and compatible with numpy<2
        "torch==2.4.0+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu


# --- Stage 2: runtime ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system libs (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY api/       ./api/
COPY src/        ./src/
COPY data/merchants.json ./data/merchants.json

# Copy local PhoBERT model if present (optional — falls back to HF Hub)
# If models/ is large and you want a faster build, skip this and set
# PHOBERT_MODEL=<hf-repo-id> at runtime.
COPY models/    ./models/

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# FastAPI listens on PORT env var (Railway sets this automatically)
ENV PORT=8000
EXPOSE $PORT

# Health check — Railway uses this to verify the container is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
