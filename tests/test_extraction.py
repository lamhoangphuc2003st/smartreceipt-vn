"""
tests/test_extraction.py
-------------------------
Unit tests for the extraction module.

All tests mock the Gemini API — no real API calls are made.
Tests cover: schema validation, prompt building, extractor parsing,
cache behavior, retry logic, and error handling.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.extraction.schemas import ExtractionResult, Receipt, ReceiptItem
from src.extraction.prompts import get_prompt, PROMPT_REGISTRY, DEFAULT_PROMPT_VERSION
from src.extraction.gemini_extractor import GeminiExtractor, _sha256_prefix


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_blank_image(h: int = 300, w: int = 200) -> np.ndarray:
    """White BGR image for testing."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def make_mock_response(text: str, input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Build a mock Gemini response object."""
    response = MagicMock()
    response.text = text
    response.usage_metadata.prompt_token_count = input_tokens
    response.usage_metadata.candidates_token_count = output_tokens
    return response


VALID_RECEIPT_JSON = json.dumps({
    "merchant_name": "Highlands Coffee",
    "date": "2024-03-15",
    "items": [
        {"name": "Cà phê sữa đá", "quantity": 2, "unit": "ly", "unit_price": 45000, "total": 90000},
        {"name": "Bánh mì thịt", "quantity": 1, "unit": None, "unit_price": 30000, "total": 30000},
    ],
    "total_amount": 120000,
    "payment_method": "thẻ",
    "receipt_type": "pos",
})

VAT_RECEIPT_JSON = json.dumps({
    "merchant_name": "CÔNG TY ABC",
    "merchant_tax_id": "0123456789",
    "date": "2024-01-20",
    "items": [
        {"name": "Dịch vụ tư vấn", "quantity": 1, "unit": None, "unit_price": 2000000, "total": 2000000}
    ],
    "subtotal": 2000000,
    "vat_amount": 200000,
    "vat_rate": 0.10,
    "total_amount": 2200000,
    "receipt_type": "vat_invoice",
})


# ── TestReceiptItem ────────────────────────────────────────────────────────────

class TestReceiptItem:
    def test_basic_item(self):
        item = ReceiptItem(name="Cà phê sữa đá", quantity=2, unit="ly", unit_price=45000, total=90000)
        assert item.name == "Cà phê sữa đá"
        assert item.quantity == 2.0
        assert item.total == 90000

    def test_coerce_string_numbers(self):
        """String prices with separators should be coerced to int."""
        item = ReceiptItem(name="Trà sữa", unit_price="45,000", total="90,000")
        assert item.unit_price == 45000
        assert item.total == 90000

    def test_empty_name_raises(self):
        with pytest.raises(Exception):
            ReceiptItem(name="   ", total=10000)

    def test_float_quantity_for_weight_items(self):
        item = ReceiptItem(name="Cà chua", quantity=1.5, unit="kg", unit_price=30000, total=45000)
        assert item.quantity == 1.5

    def test_null_optional_fields(self):
        item = ReceiptItem(name="Rau muống", total=5000)
        assert item.quantity is None
        assert item.unit is None
        assert item.unit_price is None

    def test_null_total_allowed(self):
        """Some restaurant/taxi receipts omit per-item totals — total=None must not raise."""
        item = ReceiptItem(name="Cước xe", quantity=1, unit_price=50000, total=None)
        assert item.total is None

    def test_empty_string_total_becomes_none(self):
        """Gemini sometimes returns empty string for total — should become None."""
        item = ReceiptItem(name="Dịch vụ", total="")
        assert item.total is None


# ── TestReceipt ────────────────────────────────────────────────────────────────

class TestReceipt:
    def test_full_pos_receipt(self):
        receipt = Receipt.model_validate(json.loads(VALID_RECEIPT_JSON))
        assert receipt.merchant_name == "Highlands Coffee"
        assert receipt.date == "2024-03-15"
        assert len(receipt.items) == 2
        assert receipt.total_amount == 120000
        assert receipt.receipt_type == "pos"

    def test_vat_invoice(self):
        receipt = Receipt.model_validate(json.loads(VAT_RECEIPT_JSON))
        assert receipt.merchant_tax_id == "0123456789"
        assert receipt.vat_amount == 200000
        assert receipt.vat_rate == 0.10
        assert receipt.receipt_type == "vat_invoice"

    def test_all_nullable_fields_can_be_none(self):
        """Minimal receipt with only required total_amount."""
        receipt = Receipt(total_amount=50000)
        assert receipt.merchant_name is None
        assert receipt.date is None
        assert receipt.items == []
        assert receipt.total_amount == 50000

    def test_invalid_date_becomes_none(self):
        """Malformed date should be nulled rather than raising."""
        receipt = Receipt(date="15/03/2024", total_amount=50000)  # wrong format
        assert receipt.date is None

    def test_valid_iso_date(self):
        receipt = Receipt(date="2024-03-15", total_amount=50000)
        assert receipt.date == "2024-03-15"

    def test_negative_total_becomes_positive(self):
        """Negative totals (OCR artifact) should be corrected to positive."""
        receipt = Receipt(total_amount=-50000)
        assert receipt.total_amount == 50000

    def test_string_total_coerced(self):
        receipt = Receipt(total_amount="1,234,000")
        assert receipt.total_amount == 1234000

    def test_error_field(self):
        receipt = Receipt(error="not_a_receipt")
        assert receipt.error == "not_a_receipt"
        assert receipt.merchant_name is None


# ── TestExtractionResult ──────────────────────────────────────────────────────

class TestExtractionResult:
    def test_success_result(self):
        receipt = Receipt(merchant_name="Test", total_amount=50000)
        result = ExtractionResult(success=True, receipt=receipt)
        assert result.success is True
        assert result.receipt.merchant_name == "Test"
        assert result.from_cache is False

    def test_failure_result(self):
        result = ExtractionResult(
            success=False,
            error_type="api_error",
            error_message="503 Service Unavailable",
        )
        assert result.success is False
        assert result.receipt is None
        assert result.error_type == "api_error"

    def test_default_fields(self):
        result = ExtractionResult(success=True)
        assert result.input_tokens == 0
        assert result.cost_usd == 0.0
        assert result.attempts == 1
        assert result.model == "gemini-2.5-flash-lite"


# ── TestPrompts ────────────────────────────────────────────────────────────────

class TestPrompts:
    def test_default_prompt_exists(self):
        prompt = get_prompt(DEFAULT_PROMPT_VERSION)
        assert len(prompt) > 200, "Prompt too short"

    def test_prompt_has_required_sections(self):
        prompt = get_prompt("v1")
        assert "Vietnamese" in prompt
        assert "Chain-of-thought" in prompt or "chain" in prompt.lower()
        assert "null" in prompt
        assert "not_a_receipt" in prompt

    def test_prompt_registry_has_v1(self):
        assert "v1" in PROMPT_REGISTRY

    def test_unknown_version_raises(self):
        with pytest.raises(KeyError):
            get_prompt("v99")


# ── TestGeminiExtractor ────────────────────────────────────────────────────────

class TestGeminiExtractor:
    """All Gemini API calls are mocked — no real network requests."""

    def _make_extractor(self, cache_dir: Path | None = None) -> GeminiExtractor:
        return GeminiExtractor(
            api_key="fake-key-for-tests",
            cache_dir=cache_dir or Path(tempfile.mkdtemp()),
            enable_cache=True,
        )

    def _patch_client(self, extractor: GeminiExtractor, response_text: str) -> MagicMock:
        """Inject a mock Gemini client into the extractor."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_mock_response(response_text)
        extractor._client = mock_client
        return mock_client

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_successful_extraction(self):
        extractor = self._make_extractor()
        self._patch_client(extractor, VALID_RECEIPT_JSON)

        result = extractor.extract(make_blank_image())
        assert result.success is True
        assert result.receipt.merchant_name == "Highlands Coffee"
        assert result.receipt.total_amount == 120000
        assert result.from_cache is False

    def test_extraction_from_bytes(self):
        import cv2
        extractor = self._make_extractor()
        self._patch_client(extractor, VALID_RECEIPT_JSON)

        img = make_blank_image()
        ok, buf = cv2.imencode(".jpg", img)
        result = extractor.extract(bytes(buf))
        assert result.success is True

    def test_extraction_preserves_diacritics(self):
        extractor = self._make_extractor()
        self._patch_client(extractor, VALID_RECEIPT_JSON)

        result = extractor.extract(make_blank_image())
        assert result.receipt.items[0].name == "Cà phê sữa đá"

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def test_parse_json_in_markdown_fence(self):
        extractor = self._make_extractor()
        wrapped = f"```json\n{VALID_RECEIPT_JSON}\n```"
        self._patch_client(extractor, wrapped)

        result = extractor.extract(make_blank_image())
        assert result.success is True
        assert result.receipt.merchant_name == "Highlands Coffee"

    def test_parse_json_with_prose(self):
        extractor = self._make_extractor()
        wrapped = f"Here is the extracted data:\n\n{VALID_RECEIPT_JSON}\n\nPlease let me know..."
        self._patch_client(extractor, wrapped)

        result = extractor.extract(make_blank_image())
        assert result.success is True

    def test_invalid_json_returns_failure(self):
        extractor = self._make_extractor()
        self._patch_client(extractor, "This is not JSON at all.")

        result = extractor.extract(make_blank_image())
        assert result.success is False
        assert result.error_type == "parse_error"

    def test_not_a_receipt_error_field(self):
        extractor = self._make_extractor()
        self._patch_client(extractor, json.dumps({"error": "not_a_receipt"}))

        result = extractor.extract(make_blank_image())
        # Parses successfully as a Receipt with error field set
        assert result.success is True
        assert result.receipt.error == "not_a_receipt"

    # ── Cache ─────────────────────────────────────────────────────────────────

    def test_cache_hit_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = self._make_extractor(cache_dir=Path(tmpdir))
            mock_client = self._patch_client(extractor, VALID_RECEIPT_JSON)

            img = make_blank_image()
            result1 = extractor.extract(img)
            result2 = extractor.extract(img)

            assert result1.success is True
            assert result2.success is True
            assert result2.from_cache is True
            # API called exactly once
            assert mock_client.models.generate_content.call_count == 1

    def test_cache_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = GeminiExtractor(
                api_key="fake",
                cache_dir=Path(tmpdir),
                enable_cache=False,
            )
            mock_client = self._patch_client(extractor, VALID_RECEIPT_JSON)

            img = make_blank_image()
            extractor.extract(img)
            extractor.extract(img)

            assert mock_client.models.generate_content.call_count == 2

    # ── Error handling ────────────────────────────────────────────────────────

    def test_api_exception_returns_failure(self):
        extractor = self._make_extractor()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable")
        extractor._client = mock_client

        result = extractor.extract(make_blank_image())
        assert result.success is False
        assert result.error_type is not None

    def test_permanent_error_not_retried(self):
        """A 400 Bad Request (permanent) should not be retried 3 times."""
        extractor = self._make_extractor()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("400 Bad Request — invalid API key")
        extractor._client = mock_client

        result = extractor.extract(make_blank_image())
        # With a permanent error, should stop after 1 attempt
        assert mock_client.models.generate_content.call_count == 1
        assert result.success is False

    def test_transient_error_retried(self):
        """A 503 (transient) error should be retried up to MAX_ATTEMPTS times."""
        extractor = self._make_extractor()
        mock_client = MagicMock()
        # Always fail with transient error
        mock_client.models.generate_content.side_effect = Exception("503 ServiceUnavailable")
        extractor._client = mock_client

        result = extractor.extract(make_blank_image())
        assert mock_client.models.generate_content.call_count == 3  # MAX_ATTEMPTS
        assert result.success is False
        assert result.attempts == 3

    def test_invalid_image_path_returns_failure(self):
        extractor = self._make_extractor()
        result = extractor.extract("/nonexistent/path/to/image.jpg")
        assert result.success is False
        assert result.error_type == "input_error"

    # ── Cost tracking ─────────────────────────────────────────────────────────

    def test_cost_tracked(self):
        extractor = self._make_extractor()
        self._patch_client(extractor, VALID_RECEIPT_JSON)

        result = extractor.extract(make_blank_image())
        # 100 input + 50 output tokens from mock
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost_usd > 0

    # ── Batch extraction ──────────────────────────────────────────────────────

    def test_extract_batch(self):
        extractor = self._make_extractor()
        self._patch_client(extractor, VALID_RECEIPT_JSON)

        images = [make_blank_image() for _ in range(3)]
        results = extractor.extract_batch(images, max_workers=2)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_extract_batch_one_failure(self):
        """Batch should return results for all images even if one fails."""
        extractor = self._make_extractor()
        mock_client = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("400 Bad Request")
            return make_mock_response(VALID_RECEIPT_JSON)

        mock_client.models.generate_content.side_effect = side_effect
        extractor._client = mock_client

        # Use distinct images (different pixel values) so each has a unique cache key
        images = [
            np.full((300, 200, 3), i * 80, dtype=np.uint8)
            for i in range(3)
        ]
        results = extractor.extract_batch(images, max_workers=1)

        assert len(results) == 3
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        assert successes == 2
        assert failures == 1
