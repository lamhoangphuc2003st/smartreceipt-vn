"""
src/extraction/sliding_window.py
----------------------------------
Sliding-window extractor for tall/long receipt images.

Problem: Gemini Vision has a finite attention window. For very tall receipts
(aspect ratio height/width > 2.5) the model tends to miss items in the middle
or bottom sections when the image is sent as a single frame.

Solution: Split the image into 2–3 overlapping horizontal chunks, extract each
chunk independently, then merge the results into a single Receipt.

Merge strategy:
  - merchant_name / date / time : first non-null chunk result (header info, top)
  - total_amount / subtotal / vat / payment_method : last non-null (footer, bottom)
  - items : union of all chunks, deduplicated by name similarity

Overlap rationale:
  20% overlap ensures items at chunk boundaries appear in two adjacent chunks,
  so the dedup step can confirm them rather than lose them.

When NOT to use:
  - aspect ratio ≤ 2.5  → single-frame extraction is sufficient
  - Image already failed to load  → caller should handle before calling here

Contract: extract() never raises to caller. Returns ExtractionResult.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .gemini_extractor import GeminiExtractor, _elapsed_ms
from .schemas import ExtractionResult, Receipt, ReceiptItem

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

ASPECT_RATIO_THRESHOLD = 2.5   # h/w above this triggers sliding window
OVERLAP_FRACTION       = 0.20  # 20% overlap between adjacent chunks
MAX_CHUNKS             = 3
CHUNKS_FOR_VERY_LONG   = 3     # ratio > 4.0
CHUNKS_FOR_LONG        = 2     # ratio 2.5–4.0

# rapidfuzz threshold for treating two item names as duplicates
_ITEM_DEDUP_THRESHOLD  = 85


# ── Public API ─────────────────────────────────────────────────────────────────

class SlidingWindowExtractor:
    """
    Wraps GeminiExtractor to handle tall receipts via overlapping chunks.

    Usage:
        extractor  = GeminiExtractor()
        sw         = SlidingWindowExtractor(extractor)
        result     = sw.extract(image_path)       # auto-detects if sliding needed
        # or:
        result     = sw.extract(image_array)      # numpy BGR array
    """

    def __init__(self, extractor: GeminiExtractor) -> None:
        self.extractor = extractor

    # ── Detection ─────────────────────────────────────────────────────────────

    @staticmethod
    def needs_sliding_window(image: np.ndarray) -> bool:
        """Return True if the image is tall enough to benefit from chunking."""
        h, w = image.shape[:2]
        return (h / w) > ASPECT_RATIO_THRESHOLD

    # ── Main entry point ──────────────────────────────────────────────────────

    def extract(
        self,
        image: "np.ndarray | bytes | str | Path",
    ) -> ExtractionResult:
        """
        Extract from image, using sliding window if aspect ratio > threshold.

        Falls back to single-frame extraction if:
          - image is not tall enough (ratio ≤ 2.5)
          - image fails to load as numpy array
          - all chunks fail

        Never raises to caller.
        """
        t0 = time.perf_counter()

        try:
            arr = _to_numpy(image)
        except Exception as exc:
            log.warning("SlidingWindowExtractor: could not load image: %s", exc)
            return self.extractor.extract(image)   # fall back to standard path

        if not self.needs_sliding_window(arr):
            return self.extractor.extract(arr)

        h, w = arr.shape[:2]
        ratio = h / w
        n_chunks = CHUNKS_FOR_VERY_LONG if ratio > 4.0 else CHUNKS_FOR_LONG
        log.info(
            "Sliding window: %.2f ratio → %d chunks (overlap=%.0f%%)",
            ratio, n_chunks, OVERLAP_FRACTION * 100,
        )

        chunks = _split_image(arr, n_chunks, OVERLAP_FRACTION)
        chunk_results: list[ExtractionResult] = []

        for i, chunk in enumerate(chunks):
            log.debug("  Extracting chunk %d/%d (%dpx tall)", i + 1, n_chunks, chunk.shape[0])
            result = self.extractor.extract(chunk)
            chunk_results.append(result)
            if not result.success:
                log.warning("  Chunk %d/%d failed: %s", i + 1, n_chunks, result.error_message)

        return _merge_results(chunk_results, t0)


# ── Image splitting ───────────────────────────────────────────────────────────

def _split_image(
    image: np.ndarray,
    n_chunks: int,
    overlap: float,
) -> list[np.ndarray]:
    """
    Split image into n_chunks overlapping horizontal strips.

    Each chunk overlaps the adjacent chunks by `overlap` fraction of chunk height.
    Strips are clipped to image bounds so the last chunk always reaches the bottom.

    Example for 3000px image, 3 chunks, 20% overlap:
      chunk_h = 1000px, overlap_px = 200px
      Chunk 0: rows    0 – 1200
      Chunk 1: rows  800 – 2200
      Chunk 2: rows 1800 – 3000
    """
    h = image.shape[0]
    chunk_h   = h // n_chunks
    overlap_px = int(chunk_h * overlap)

    chunks = []
    for i in range(n_chunks):
        start = max(0,  i * chunk_h - overlap_px)
        end   = min(h, (i + 1) * chunk_h + overlap_px)
        # Last chunk always extends to image bottom
        if i == n_chunks - 1:
            end = h
        chunks.append(image[start:end, :].copy())

    return chunks


# ── Result merging ────────────────────────────────────────────────────────────

def _merge_results(
    results: list[ExtractionResult],
    t0: float,
) -> ExtractionResult:
    """
    Merge chunk ExtractionResults into one unified ExtractionResult.

    Merge rules:
      Header fields (merchant, date, time)       → first non-null value
      Footer fields (total, vat, payment)        → last non-null value
      Items                                       → union, deduplicated by name
      Cost / tokens                               → sum across all chunks
      success                                     → True if ≥1 chunk succeeded
    """
    successful = [r for r in results if r.success and r.receipt is not None]

    if not successful:
        # All chunks failed — return the first result as representative error
        first = results[0] if results else ExtractionResult(
            success=False,
            error_type="sliding_window_error",
            error_message="all chunks failed",
        )
        return ExtractionResult(
            success=False,
            model=first.model,
            prompt_version=first.prompt_version,
            error_type="sliding_window_all_failed",
            error_message=f"all {len(results)} chunks failed to extract",
            cost_usd=sum(r.cost_usd for r in results),
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
            latency_ms=_elapsed_ms(t0),
            attempts=max((r.attempts for r in results), default=1),
        )

    receipts = [r.receipt for r in successful]  # type: ignore[misc]

    # ── Header fields: first non-null ────────────────────────────────────────
    merchant_name    = _first_non_null(r.merchant_name    for r in receipts)
    merchant_address = _first_non_null(r.merchant_address for r in receipts)
    merchant_tax_id  = _first_non_null(r.merchant_tax_id  for r in receipts)
    date_            = _first_non_null(r.date             for r in receipts)
    time_            = _first_non_null(r.time             for r in receipts)
    receipt_type     = _first_non_null(r.receipt_type     for r in receipts)
    receipt_number   = _first_non_null(r.receipt_number   for r in receipts)

    # ── Footer fields: last non-null ─────────────────────────────────────────
    subtotal         = _last_non_null(r.subtotal         for r in receipts)
    discount         = _last_non_null(r.discount         for r in receipts)
    vat_amount       = _last_non_null(r.vat_amount       for r in receipts)
    vat_rate         = _last_non_null(r.vat_rate         for r in receipts)
    total_amount     = _last_non_null(r.total_amount     for r in receipts)
    payment_method   = _last_non_null(r.payment_method   for r in receipts)

    # ── Items: union with dedup ───────────────────────────────────────────────
    all_items: list[ReceiptItem] = []
    for r in receipts:
        all_items.extend(r.items or [])

    merged_items = _dedup_items(all_items)
    log.debug(
        "Merged items: %d raw → %d after dedup",
        len(all_items), len(merged_items),
    )

    # ── Build merged receipt ──────────────────────────────────────────────────
    merged_receipt = Receipt(
        merchant_name=merchant_name,
        merchant_address=merchant_address,
        merchant_tax_id=merchant_tax_id,
        date=date_,
        time=time_,
        items=merged_items,
        subtotal=subtotal,
        discount=discount,
        vat_amount=vat_amount,
        vat_rate=vat_rate,
        total_amount=total_amount,
        payment_method=payment_method,
        receipt_number=receipt_number,
        receipt_type=receipt_type,
    )

    first_success = successful[0]
    return ExtractionResult(
        success=True,
        receipt=merged_receipt,
        model=first_success.model,
        prompt_version=first_success.prompt_version,
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        cost_usd=sum(r.cost_usd for r in results),
        latency_ms=_elapsed_ms(t0),
        from_cache=all(r.from_cache for r in results),
        attempts=max(r.attempts for r in results),
    )


def _dedup_items(items: list[ReceiptItem]) -> list[ReceiptItem]:
    """
    Remove duplicate items caused by overlap between chunks.

    Two items are considered duplicates if their names have
    rapidfuzz token_set_ratio ≥ _ITEM_DEDUP_THRESHOLD.

    When duplicates exist, keep the one with more complete data
    (prefers non-null quantity, unit_price, and total).
    """
    if not items:
        return []

    from rapidfuzz import fuzz

    kept: list[ReceiptItem] = []

    for candidate in items:
        cand_name = candidate.name.lower().strip()
        is_dup = False

        for i, existing in enumerate(kept):
            existing_name = existing.name.lower().strip()
            score = fuzz.token_set_ratio(cand_name, existing_name)

            if score >= _ITEM_DEDUP_THRESHOLD:
                # Duplicate found — keep the more complete one
                if _completeness(candidate) > _completeness(existing):
                    kept[i] = candidate
                is_dup = True
                break

        if not is_dup:
            kept.append(candidate)

    return kept


def _completeness(item: ReceiptItem) -> int:
    """Score how complete an item's data is (higher = more fields filled in)."""
    return sum([
        item.quantity is not None,
        item.unit is not None,
        item.unit_price is not None,
        item.total is not None,
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_non_null(values) -> object:
    for v in values:
        if v is not None:
            return v
    return None


def _last_non_null(values) -> object:
    result = None
    for v in values:
        if v is not None:
            result = v
    return result


def _to_numpy(image: "np.ndarray | bytes | str | Path") -> np.ndarray:
    """Convert image input to BGR numpy array."""
    if isinstance(image, np.ndarray):
        return image

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        arr = cv2.imread(str(path))
        if arr is None:
            raise ValueError(f"cv2.imread failed for: {path}")
        return arr

    if isinstance(image, (bytes, bytearray)):
        arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("cv2.imdecode failed for provided bytes")
        return arr

    raise TypeError(f"Unsupported image type: {type(image)}")
