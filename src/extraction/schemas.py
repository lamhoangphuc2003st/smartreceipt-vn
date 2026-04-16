"""
src/extraction/schemas.py
--------------------------
Pydantic models for structured Gemini output and downstream processing.

These schemas serve three purposes:
  1. Structured output enforcement — passed to Gemini via response_schema so
     the model outputs valid JSON matching our schema (no free-form parsing).
  2. Validation layer input — passed to src/validation/ for math and sanity checks.
  3. API response serialization — returned to callers via FastAPI.

Design principles:
  - All monetary values are integer VND (no decimals, no commas).
  - Fields the VLM cannot read confidently are nullable (None), never guessed.
  - ExtractionResult wraps Receipt with metadata — callers only import ExtractionResult.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Item-level model ──────────────────────────────────────────────────────────

class ReceiptItem(BaseModel):
    """A single line item on a receipt."""

    name: str = Field(
        description="Item name in Vietnamese as printed on receipt. Preserve diacritics."
    )
    quantity: float | None = Field(
        default=None,
        description="Quantity sold. Float to handle weight-based items (1.5 kg). None if unclear.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measure as printed: 'cái', 'kg', 'hộp', 'chai', etc. None if absent.",
    )
    unit_price: int | None = Field(
        default=None,
        description="Price per unit in VND (integer). None if not shown.",
    )
    total: int | None = Field(
        default=None,
        description=(
            "Line total in VND (integer). None if not printed (some restaurant/taxi receipts "
            "omit per-item totals). Never guess — use None if unclear."
        ),
    )

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("item name cannot be empty")
        return v

    @field_validator("total", "unit_price", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> Any:
        """Accept strings like '45,000' or '45.000' — strip separators. Empty/null → None."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.replace(",", "").replace(".", "").strip()
            return int(v) if v else None
        return v


# ── Receipt-level model ───────────────────────────────────────────────────────

class Receipt(BaseModel):
    """
    Full structured output of a single receipt.

    All monetary values are integer VND.
    Nullable fields → None means "not readable" (never guessed).
    """

    merchant_name: str | None = Field(
        default=None,
        description=(
            "Merchant / store name as printed. "
            "None if not readable or image is not a receipt."
        ),
    )
    merchant_address: str | None = Field(
        default=None,
        description="Store address as printed. None if absent.",
    )
    merchant_tax_id: str | None = Field(
        default=None,
        description="Vietnamese Tax ID (MST) if present on VAT invoices. None otherwise.",
    )

    date: str | None = Field(
        default=None,
        description=(
            "Transaction date in ISO format YYYY-MM-DD. "
            "If only month/year visible, use YYYY-MM-01. None if unreadable."
        ),
    )
    time: str | None = Field(
        default=None,
        description="Transaction time HH:MM (24h). None if absent.",
    )

    items: list[ReceiptItem] = Field(
        default_factory=list,
        description="Line items. Empty list only if receipt has no itemized section.",
    )

    @field_validator("items", mode="before")
    @classmethod
    def filter_null_items(cls, v: Any) -> Any:
        """Drop items where name is null — Gemini occasionally returns null names.
        Only filters dicts with explicit null name; passes ReceiptItem objects through."""
        if not isinstance(v, list):
            return v
        return [item for item in v if not (isinstance(item, dict) and item.get("name") is None)]

    subtotal: int | None = Field(
        default=None,
        description="Subtotal before tax/discount in VND. None if not shown.",
    )
    discount: int | None = Field(
        default=None,
        description="Total discount amount in VND (positive integer). None if absent.",
    )
    vat_amount: int | None = Field(
        default=None,
        description="VAT/tax amount in VND. None if not shown.",
    )
    vat_rate: float | None = Field(
        default=None,
        description="VAT rate as decimal (0.08 for 8%). None if not shown.",
    )
    total_amount: int | None = Field(
        default=None,
        description=(
            "Grand total paid in VND. This is the most important field — "
            "extract it even if items are unreadable."
        ),
    )

    payment_method: str | None = Field(
        default=None,
        description="Payment method: 'tiền mặt', 'thẻ', 'QR', 'VNPay', etc. None if absent.",
    )
    receipt_number: str | None = Field(
        default=None,
        description="Receipt/invoice number as printed. None if absent.",
    )

    receipt_type: str | None = Field(
        default=None,
        description=(
            "Type of receipt: 'pos', 'vat_invoice', 'handwritten', 'restaurant', 'other'. "
            "None if cannot determine."
        ),
    )

    error: str | None = Field(
        default=None,
        description=(
            "Set to 'not_a_receipt' if image is not a receipt. "
            "Set to 'unreadable' if image is too blurry/dark to extract any fields. "
            "None for normal receipts."
        ),
    )

    @field_validator("total_amount", "subtotal", "discount", "vat_amount", mode="before")
    @classmethod
    def coerce_money(cls, v: Any) -> Any:
        """Accept '1,234,000' style strings or floats — strip separators, round floats."""
        if isinstance(v, float):
            return round(v)
        if isinstance(v, str):
            v = v.replace(",", "").replace(".", "").strip()
            return int(v) if v else None
        return v

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Accept YYYY-MM-DD only (model is instructed to output this format)
        try:
            date.fromisoformat(v)
        except ValueError:
            return None   # Reject malformed dates rather than raising
        return v

    @model_validator(mode="after")
    def check_total_sign(self) -> Receipt:
        """Totals must be non-negative."""
        for field_name in ("total_amount", "subtotal", "vat_amount"):
            val = getattr(self, field_name)
            if val is not None and val < 0:
                setattr(self, field_name, abs(val))
        return self


# ── Extraction result wrapper ─────────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """
    Returned by GeminiExtractor.extract().  Never raises to caller.

    success=False means extraction failed (Gemini error, parse error, etc.).
    receipt is set when success=True; it may still have null fields.
    """

    success: bool
    receipt: Receipt | None = None

    # Provenance / tracing
    model: str = Field(default="gemini-2.5-flash-lite")
    prompt_version: str = Field(default="v1")
    image_hash: str | None = Field(
        default=None,
        description="SHA256 (first 16 hex chars) of the input image bytes.",
    )

    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    # Latency
    latency_ms: float = 0.0

    # Cache hit
    from_cache: bool = False

    # Error info (populated when success=False)
    error_type: str | None = None     # 'api_error' | 'parse_error' | 'timeout' | 'rate_limit'
    error_message: str | None = None

    # Retry info
    attempts: int = 1
