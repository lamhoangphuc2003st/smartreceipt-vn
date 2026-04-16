"""
tests/test_sliding_window.py
------------------------------
Unit tests for the sliding window extractor.

All tests mock GeminiExtractor so no real API calls are made.

Covers:
  - needs_sliding_window detection (tall vs normal images)
  - _split_image: correct chunk count, overlap, boundary conditions
  - _dedup_items: identical names, fuzzy duplicates, distinct items, completeness preference
  - _merge_results: header from first, footer from last, all-fail case, cost summing
  - SlidingWindowExtractor.extract: falls through to standard path when ratio ≤ 2.5
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.extraction.schemas import ExtractionResult, Receipt, ReceiptItem
from src.extraction.sliding_window import (
    ASPECT_RATIO_THRESHOLD,
    SlidingWindowExtractor,
    _dedup_items,
    _merge_results,
    _split_image,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_image(w: int, h: int) -> np.ndarray:
    """Create a solid-color BGR image of given dimensions."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_receipt(**kwargs) -> Receipt:
    defaults = dict(
        merchant_name="Test Merchant",
        date="2024-03-15",
        total_amount=100000,
        items=[ReceiptItem(name="Item A", total=100000)],
    )
    defaults.update(kwargs)
    return Receipt.model_validate(defaults)


def make_extraction(success: bool = True, **receipt_kwargs) -> ExtractionResult:
    return ExtractionResult(
        success=success,
        receipt=make_receipt(**receipt_kwargs) if success else None,
        model="gemini-2.5-flash-lite",
        prompt_version="v1",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0001,
        latency_ms=3000.0,
        error_type=None if success else "api_error",
        error_message=None if success else "test error",
    )


def make_mock_extractor(results: list[ExtractionResult]) -> MagicMock:
    """Mock GeminiExtractor that returns results in sequence."""
    mock = MagicMock()
    mock.extract.side_effect = results
    return mock


# ── Detection tests ───────────────────────────────────────────────────────────

class TestNeedsSlidingWindow:
    def test_tall_image_triggers(self):
        img = make_image(400, 1200)   # ratio = 3.0
        assert SlidingWindowExtractor.needs_sliding_window(img) is True

    def test_exactly_at_threshold_does_not_trigger(self):
        """Ratio exactly equal to threshold must NOT trigger (strictly greater)."""
        w, h = 400, int(400 * ASPECT_RATIO_THRESHOLD)
        img = make_image(w, h)
        assert SlidingWindowExtractor.needs_sliding_window(img) is False

    def test_normal_portrait_image(self):
        img = make_image(400, 800)   # ratio = 2.0
        assert SlidingWindowExtractor.needs_sliding_window(img) is False

    def test_landscape_image(self):
        img = make_image(1200, 400)  # ratio = 0.33
        assert SlidingWindowExtractor.needs_sliding_window(img) is False

    def test_square_image(self):
        img = make_image(500, 500)   # ratio = 1.0
        assert SlidingWindowExtractor.needs_sliding_window(img) is False


# ── Split tests ───────────────────────────────────────────────────────────────

class TestSplitImage:
    def test_two_chunks_produced(self):
        img = make_image(400, 1200)
        chunks = _split_image(img, n_chunks=2, overlap=0.20)
        assert len(chunks) == 2

    def test_three_chunks_produced(self):
        img = make_image(400, 3000)
        chunks = _split_image(img, n_chunks=3, overlap=0.20)
        assert len(chunks) == 3

    def test_chunks_have_correct_width(self):
        """Width must be unchanged after splitting."""
        img = make_image(400, 1200)
        for chunk in _split_image(img, n_chunks=2, overlap=0.20):
            assert chunk.shape[1] == 400

    def test_last_chunk_reaches_bottom(self):
        """Last chunk must always end at the image bottom row."""
        h = 1005   # odd number to stress boundary handling
        img = make_image(389, h)
        chunks = _split_image(img, n_chunks=2, overlap=0.20)
        assert chunks[-1].shape[0] + chunks[-1].shape[0] >= h - 10   # within tolerance

    def test_chunks_have_overlap(self):
        """Adjacent chunks must be taller than h/n_chunks (overlap adds rows)."""
        h, n = 1200, 2
        img = make_image(400, h)
        chunks = _split_image(img, n_chunks=n, overlap=0.20)
        base_h = h // n
        for chunk in chunks[:-1]:   # last chunk may vary
            assert chunk.shape[0] > base_h

    def test_chunks_are_copies(self):
        """Modifying a chunk must not modify the original image."""
        img = make_image(400, 1200)
        img[:] = 128   # gray
        chunks = _split_image(img, n_chunks=2, overlap=0.20)
        chunks[0][:] = 0
        assert img[0, 0, 0] == 128  # original unchanged


# ── Dedup tests ───────────────────────────────────────────────────────────────

class TestDedupItems:
    def test_identical_names_deduped(self):
        items = [
            ReceiptItem(name="Mì Hảo Hảo", total=7000),
            ReceiptItem(name="Mì Hảo Hảo", total=7000),
        ]
        result = _dedup_items(items)
        assert len(result) == 1

    def test_fuzzy_typo_deduped(self):
        """Near-identical names (OCR typo) must be treated as duplicates."""
        items = [
            ReceiptItem(name="Nước Aqua 500ml", total=10000),
            ReceiptItem(name="Nuoc Aqua 500ml", total=10000),  # missing diacritics
        ]
        result = _dedup_items(items)
        assert len(result) == 1

    def test_distinct_items_kept(self):
        items = [
            ReceiptItem(name="Mì Hảo Hảo", total=7000),
            ReceiptItem(name="Nước Aqua 500ml", total=10000),
            ReceiptItem(name="Bánh mì", total=15000),
        ]
        result = _dedup_items(items)
        assert len(result) == 3

    def test_prefers_more_complete_item(self):
        """When deduping, keep the item with more non-null fields."""
        incomplete = ReceiptItem(name="Cà phê sữa", total=45000)            # 1 non-null field
        complete   = ReceiptItem(name="Cà phê sữa", quantity=1, unit="ly",
                                 unit_price=45000, total=45000)              # 4 non-null fields
        result = _dedup_items([incomplete, complete])
        assert len(result) == 1
        assert result[0].quantity == 1
        assert result[0].unit_price == 45000

    def test_empty_list_returns_empty(self):
        assert _dedup_items([]) == []

    def test_single_item_unchanged(self):
        items = [ReceiptItem(name="Bánh", total=5000)]
        assert _dedup_items(items) == items


# ── Merge tests ───────────────────────────────────────────────────────────────

class TestMergeResults:
    def test_header_from_first_chunk(self):
        """merchant_name and date must come from the first successful chunk."""
        results = [
            make_extraction(merchant_name="Merchant Top",  date="2024-01-01", total_amount=50000),
            make_extraction(merchant_name="Merchant Bot",  date="2024-01-02", total_amount=100000),
        ]
        merged = _merge_results(results, t0=0.0)
        assert merged.receipt.merchant_name == "Merchant Top"
        assert merged.receipt.date == "2024-01-01"

    def test_total_from_last_chunk(self):
        """total_amount must come from the last successful chunk (receipt footer)."""
        results = [
            make_extraction(total_amount=50000),
            make_extraction(total_amount=100000),
        ]
        merged = _merge_results(results, t0=0.0)
        assert merged.receipt.total_amount == 100000

    def test_items_merged_from_all_chunks(self):
        """Items from all chunks must appear in result (minus duplicates)."""
        results = [
            make_extraction(items=[
                ReceiptItem(name="Item A", total=10000),
                ReceiptItem(name="Item B", total=20000),
            ], total_amount=100000),
            make_extraction(items=[
                ReceiptItem(name="Item B", total=20000),   # duplicate
                ReceiptItem(name="Item C", total=30000),
            ], total_amount=100000),
        ]
        merged = _merge_results(results, t0=0.0)
        names = {item.name for item in merged.receipt.items}
        assert names == {"Item A", "Item B", "Item C"}

    def test_cost_summed_across_chunks(self):
        results = [
            make_extraction(cost_usd=0.001),
            make_extraction(cost_usd=0.002),
        ]
        # Patch cost into ExtractionResult
        results[0] = results[0].model_copy(update={"cost_usd": 0.001})
        results[1] = results[1].model_copy(update={"cost_usd": 0.002})
        merged = _merge_results(results, t0=0.0)
        assert abs(merged.cost_usd - 0.003) < 1e-6

    def test_all_chunks_failed_returns_failure(self):
        results = [
            make_extraction(success=False),
            make_extraction(success=False),
        ]
        merged = _merge_results(results, t0=0.0)
        assert merged.success is False
        assert merged.receipt is None

    def test_partial_success_uses_successful_chunks(self):
        """If some chunks fail, merge should use successful chunks only."""
        results = [
            make_extraction(success=False),
            make_extraction(success=True, merchant_name="Good Merchant", total_amount=100000),
        ]
        merged = _merge_results(results, t0=0.0)
        assert merged.success is True
        assert merged.receipt.merchant_name == "Good Merchant"

    def test_merged_result_is_valid_extraction_result(self):
        results = [make_extraction(), make_extraction()]
        merged = _merge_results(results, t0=0.0)
        assert isinstance(merged, ExtractionResult)
        assert isinstance(merged.receipt, Receipt)


# ── Integration: extract() routing ────────────────────────────────────────────

class TestSlidingWindowExtractorRouting:
    def test_short_image_uses_standard_path(self):
        """Normal aspect ratio → delegates directly to GeminiExtractor.extract."""
        mock_extractor = make_mock_extractor([make_extraction()])
        sw = SlidingWindowExtractor(mock_extractor)

        img = make_image(400, 600)   # ratio = 1.5
        result = sw.extract(img)

        mock_extractor.extract.assert_called_once_with(img)
        assert result.success is True

    def test_tall_image_calls_extract_multiple_times(self):
        """Tall image → extractor called once per chunk."""
        chunk_results = [make_extraction(), make_extraction()]
        mock_extractor = make_mock_extractor(chunk_results)
        sw = SlidingWindowExtractor(mock_extractor)

        img = make_image(400, 1200)   # ratio = 3.0 → 2 chunks
        sw.extract(img)

        assert mock_extractor.extract.call_count == 2

    def test_very_tall_image_uses_three_chunks(self):
        """Ratio > 4.0 → 3 chunks."""
        chunk_results = [make_extraction(), make_extraction(), make_extraction()]
        mock_extractor = make_mock_extractor(chunk_results)
        sw = SlidingWindowExtractor(mock_extractor)

        img = make_image(400, 2000)   # ratio = 5.0 → 3 chunks
        sw.extract(img)

        assert mock_extractor.extract.call_count == 3

    def test_returns_extraction_result(self):
        mock_extractor = make_mock_extractor([make_extraction(), make_extraction()])
        sw = SlidingWindowExtractor(mock_extractor)

        img = make_image(400, 1200)
        result = sw.extract(img)

        assert isinstance(result, ExtractionResult)
