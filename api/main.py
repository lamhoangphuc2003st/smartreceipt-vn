"""
api/main.py
------------
SmartReceipt VN — FastAPI application entry point.

Endpoints:
  GET  /health               — liveness + component status
  POST /receipts/extract     — upload image → structured receipt JSON

Run locally:
    uvicorn api.main:app --reload --port 8000

Environment variables:
  Gemini_API_Key  — required (Gemini Vision API key)
  PHOBERT_MODEL   — optional (HF Hub repo ID, default: lamhoangphuc/phobert-expense-v1)
  DISABLE_PHOBERT — set to "1" to force keyword-only classification
  LOG_LEVEL       — DEBUG | INFO | WARNING (default: INFO)
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
from api.limiter import limiter  # noqa: E402

# ── App version ───────────────────────────────────────────────────────────────

APP_VERSION = "0.9.0"

# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the pipeline once at startup so it's shared across all requests.
    Avoids re-loading the PhoBERT model on every request.
    """
    log.info("SmartReceipt VN v%s starting up ...", APP_VERSION)

    from src.pipeline import ReceiptPipeline
    from api.routes.receipts import set_pipeline

    phobert_model = os.getenv("PHOBERT_MODEL", "models/phobert-expense-v1")
    disable_phobert = os.getenv("DISABLE_PHOBERT", "0") == "1"

    if disable_phobert:
        phobert_model = "__disabled__"
        log.info("PhoBERT disabled via DISABLE_PHOBERT=1 — keyword fallback only")

    pipeline = ReceiptPipeline(
        classifier_model=phobert_model,
        classifier_device="cpu",
    )
    set_pipeline(pipeline)

    log.info("Pipeline ready.")
    yield

    log.info("SmartReceipt VN shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartReceipt VN",
    description=(
        "Vietnamese receipt extraction and expense classification API.\n\n"
        "Upload a receipt photo → get structured JSON with merchant, date, "
        "line items, total, and expense category."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow all origins for demo purposes — tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

from api.routes.receipts import router as receipts_router  # noqa: E402

app.include_router(receipts_router)


# ── Health check ──────────────────────────────────────────────────────────────

from api.models import HealthResponse  # noqa: E402
from api.routes.receipts import _pipeline  # noqa: E402


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """
    Liveness check.  Returns component status for each pipeline stage.

    status="ok"      → all components healthy
    status="degraded" → some components unavailable (keyword fallback active)
    """
    from api.routes import receipts_router as _r  # re-import to get current _pipeline ref
    import api.routes.receipts as _receipts_mod

    pipeline = _receipts_mod._pipeline
    components: dict[str, str] = {}

    if pipeline is None:
        return HealthResponse(
            status="degraded",
            version=APP_VERSION,
            components={"pipeline": "not_initialized"},
        )

    components["preprocessing"] = "ok" if pipeline._preprocessor is not None else "unavailable"
    components["extractor"] = "ok"   # GeminiExtractor always initialized (may fail on actual calls)
    components["validator"] = "ok" if pipeline._validator is not None else "unavailable"

    if pipeline._classifier is not None:
        components["classifier"] = (
            "phobert" if pipeline._classifier.using_phobert else "keyword_fallback"
        )
    else:
        components["classifier"] = "unavailable"

    overall = "ok" if all(v not in ("unavailable", "not_initialized") for v in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=APP_VERSION,
        components=components,
    )
