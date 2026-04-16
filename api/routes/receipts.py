"""
api/routes/receipts.py
-----------------------
Receipt extraction endpoint.

POST /receipts/extract
  - Accepts: multipart/form-data with file=<image>
  - Returns: ReceiptResponse JSON
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from api.models import (
    ClassificationResponse,
    CorrectionResponse,
    PipelineMetaResponse,
    ReceiptItemResponse,
    ReceiptResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/receipts", tags=["receipts"])

from api.limiter import limiter

# Magic bytes for allowed image types
_MAGIC_BYTES: dict[str, bytes] = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png":  b"\x89PNG",
    "image/webp": b"RIFF",      # RIFF....WEBP — checked further below
}


def _validate_magic_bytes(content_type: str, data: bytes) -> bool:
    """Return True if file header matches expected magic bytes for content_type."""
    magic = _MAGIC_BYTES.get(content_type)
    if magic is None:
        return False
    if not data.startswith(magic):
        return False
    # Extra check for WebP: bytes 8-11 must be b"WEBP"
    if content_type == "image/webp" and data[8:12] != b"WEBP":
        return False
    return True

# Pipeline is injected at startup (see api/main.py)
_pipeline = None


def set_pipeline(pipeline: object) -> None:
    global _pipeline
    _pipeline = pipeline


@router.post(
    "/extract",
    response_model=ReceiptResponse,
    summary="Extract structured data from a receipt image",
    description=(
        "Upload a receipt photo (JPEG, PNG, WebP). "
        "Returns structured JSON with merchant, date, items, total, and expense category."
    ),
)
@limiter.limit("10/minute")
async def extract_receipt(
    request: Request,
    file: UploadFile = File(..., description="Receipt image (JPEG/PNG/WebP, max 10 MB)"),
) -> ReceiptResponse:
    # ── Validate content-type header ──────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(image_bytes) > 10 * 1024 * 1024:   # 10 MB hard limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size is 10 MB.",
        )

    # ── Magic bytes check (prevent content-type spoofing) ─────────────────────
    if not _validate_magic_bytes(file.content_type, image_bytes):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File content does not match declared content type.",
        )

    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized. Check server startup logs.",
        )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    log.info("Received receipt image — %d bytes, filename=%s", len(image_bytes), file.filename)

    result = _pipeline.process(image_bytes)

    # ── Map PipelineResult → ReceiptResponse ──────────────────────────────────
    if not result.success:
        return ReceiptResponse(
            success=False,
            error_type=result.error_type,
            error_message=result.error_message,
            meta=_build_meta(result),
        )

    return ReceiptResponse(
        success=True,
        merchant_name=result.merchant_name,
        merchant_address=result.merchant_address,
        date=result.date,
        time=result.time,
        items=[ReceiptItemResponse(**item) for item in (result.items or [])],
        subtotal=result.subtotal,
        discount=result.discount,
        vat_amount=result.vat_amount,
        vat_rate=result.vat_rate,
        total_amount=result.total_amount,
        payment_method=result.payment_method,
        receipt_number=result.receipt_number,
        receipt_type=result.receipt_type,
        classification=ClassificationResponse(
            category=result.category,
            confidence=result.category_confidence,
            method=result.category_method,
            all_scores=result.category_all_scores or {},
        ),
        corrections=[CorrectionResponse(**c) for c in (result.validation_corrections or [])],
        warnings=result.validation_warnings or [],
        meta=_build_meta(result),
    )


def _build_meta(result: object) -> PipelineMetaResponse:
    return PipelineMetaResponse(
        total_time_ms=result.total_time_ms,
        stage_times_ms=result.stage_times_ms or {},
        model_versions=result.model_versions or {},
        fallbacks_triggered=result.fallbacks_triggered or [],
        preprocessing_steps=result.preprocessing_steps or [],
        extraction_from_cache=result.extraction_from_cache,
        extraction_attempts=result.extraction_attempts,
        extraction_cost_usd=result.extraction_cost_usd,
        image_hash=result.image_hash,
    )
