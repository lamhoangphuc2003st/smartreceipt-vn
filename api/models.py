"""
api/models.py
--------------
Pydantic request/response models for the FastAPI receipt API.

These are separate from src/extraction/schemas.py (internal VLM schemas).
They define the public API contract — what callers send and receive.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Item ─────────────────────────────────────────────────────────────────────

class ReceiptItemResponse(BaseModel):
    name: str
    quantity: float | None = None
    unit: str | None = None
    unit_price: int | None = None
    total: int | None = None


# ── Correction ───────────────────────────────────────────────────────────────

class CorrectionResponse(BaseModel):
    field: str
    original: Any
    corrected: Any
    reason: str


# ── Classification ────────────────────────────────────────────────────────────

class ClassificationResponse(BaseModel):
    category: str | None = None
    confidence: float | None = None
    method: str | None = None                        # "phobert" | "keyword"
    all_scores: dict[str, float] = Field(default_factory=dict)


# ── Metadata ──────────────────────────────────────────────────────────────────

class PipelineMetaResponse(BaseModel):
    total_time_ms: float
    stage_times_ms: dict[str, float]
    model_versions: dict[str, str]
    fallbacks_triggered: list[str]
    preprocessing_steps: list[str]
    extraction_from_cache: bool
    extraction_attempts: int
    extraction_cost_usd: float
    image_hash: str | None = None


# ── Main response ─────────────────────────────────────────────────────────────

class ReceiptResponse(BaseModel):
    """
    Full response returned by POST /receipts/extract.

    success=True means extraction succeeded; all other failures are graceful.
    """
    success: bool

    # Receipt fields (all nullable — None if not readable)
    merchant_name: str | None = None
    merchant_address: str | None = None
    date: str | None = None
    time: str | None = None
    items: list[ReceiptItemResponse] = Field(default_factory=list)
    subtotal: int | None = None
    discount: int | None = None
    vat_amount: int | None = None
    vat_rate: float | None = None
    total_amount: int | None = None
    payment_method: str | None = None
    receipt_number: str | None = None
    receipt_type: str | None = None

    # Classification
    classification: ClassificationResponse = Field(default_factory=ClassificationResponse)

    # Validation metadata
    corrections: list[CorrectionResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Pipeline metadata
    meta: PipelineMetaResponse | None = None

    # Error info (populated when success=False)
    error_type: str | None = None
    error_message: str | None = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str        # "ok" | "degraded"
    version: str
    components: dict[str, str]   # component → "ok" | "unavailable"
