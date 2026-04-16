"""
tests/test_pipeline.py
-----------------------
Unit tests for ReceiptPipeline orchestrator.

All external I/O (Gemini API, image files) is mocked.
Tests verify:
  - Correct stage sequencing and data flow
  - Graceful fallbacks when each stage fails
  - PipelineResult fields are populated correctly
  - Pipeline never raises to caller
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.receipt_pipeline import PipelineResult, ReceiptPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_receipt(**kwargs):
    """Build a minimal mock Receipt object."""
    defaults = {
        "merchant_name": "Highlands Coffee",
        "merchant_address": "123 Nguyen Hue",
        "date": "2024-03-15",
        "time": "10:30",
        "items": [],
        "subtotal": None,
        "discount": None,
        "vat_amount": None,
        "vat_rate": None,
        "total_amount": 55000,
        "payment_method": "thẻ",
        "receipt_number": "R001",
        "receipt_type": "pos",
        "error": None,
    }
    defaults.update(kwargs)
    receipt = MagicMock()
    for k, v in defaults.items():
        setattr(receipt, k, v)
    receipt.items = []
    return receipt


def _make_extraction_result(success=True, receipt=None, **kwargs):
    """Build a mock ExtractionResult."""
    result = MagicMock()
    result.success = success
    result.receipt = receipt or (_make_receipt() if success else None)
    result.from_cache = kwargs.get("from_cache", False)
    result.cost_usd = kwargs.get("cost_usd", 0.0003)
    result.attempts = kwargs.get("attempts", 1)
    result.error_type = kwargs.get("error_type", None)
    result.error_message = kwargs.get("error_message", None)
    result.image_hash = kwargs.get("image_hash", "abc123")
    return result


def _make_classification_result(category="Ăn uống", confidence=0.92, method="keyword"):
    result = MagicMock()
    result.category = category
    result.confidence = confidence
    result.method = method
    result.all_scores = {category: confidence}
    return result


def _make_validation_result(receipt=None, corrections=None, warnings=None):
    result = MagicMock()
    result.receipt = receipt or _make_receipt()
    result.corrections = corrections or []
    result.warnings = warnings or []
    result.is_valid = True
    return result


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline_no_components():
    """
    ReceiptPipeline with all external components mocked out.
    The extractor is always replaced — it's the only required component.
    """
    pipeline = ReceiptPipeline.__new__(ReceiptPipeline)
    pipeline._enable_preprocessing = False
    pipeline._enable_validation = False
    pipeline._enable_classification = False
    pipeline._preprocessor = None
    pipeline._validator = None
    pipeline._classifier = None
    pipeline._gemini_model = "gemini-2.5-flash-lite"
    pipeline._classifier_model = "disabled"

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = _make_extraction_result()
    pipeline._extractor = mock_extractor

    return pipeline


# ── PipelineResult ────────────────────────────────────────────────────────────

class TestPipelineResult:
    def test_default_success_false(self):
        r = PipelineResult(success=False)
        assert r.success is False

    def test_items_default_empty(self):
        r = PipelineResult(success=True)
        assert r.items == []

    def test_fallbacks_default_empty(self):
        r = PipelineResult(success=True)
        assert r.fallbacks_triggered == []

    def test_stage_times_default_empty(self):
        r = PipelineResult(success=True)
        assert r.stage_times_ms == {}


# ── Happy path ────────────────────────────────────────────────────────────────

class TestPipelineHappyPath:
    def test_process_returns_pipeline_result(self, pipeline_no_components):
        result = pipeline_no_components.process(b"fake_image_bytes")
        assert isinstance(result, PipelineResult)

    def test_success_true_on_good_extraction(self, pipeline_no_components):
        result = pipeline_no_components.process(b"fake_image_bytes")
        assert result.success is True

    def test_merchant_name_propagated(self, pipeline_no_components):
        receipt = _make_receipt(merchant_name="KFC")
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(receipt=receipt)
        result = pipeline_no_components.process(b"fake")
        assert result.merchant_name == "KFC"

    def test_total_amount_propagated(self, pipeline_no_components):
        receipt = _make_receipt(total_amount=120000)
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(receipt=receipt)
        result = pipeline_no_components.process(b"fake")
        assert result.total_amount == 120000

    def test_date_propagated(self, pipeline_no_components):
        receipt = _make_receipt(date="2024-06-01")
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(receipt=receipt)
        result = pipeline_no_components.process(b"fake")
        assert result.date == "2024-06-01"

    def test_stage_times_populated(self, pipeline_no_components):
        result = pipeline_no_components.process(b"fake")
        assert "extraction" in result.stage_times_ms

    def test_total_time_ms_positive(self, pipeline_no_components):
        result = pipeline_no_components.process(b"fake")
        assert result.total_time_ms >= 0

    def test_model_versions_populated(self, pipeline_no_components):
        result = pipeline_no_components.process(b"fake")
        assert result.model_versions["extractor"] == "gemini-2.5-flash-lite"

    def test_image_hash_propagated(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(image_hash="deadbeef")
        result = pipeline_no_components.process(b"fake")
        assert result.image_hash == "deadbeef"

    def test_extraction_cost_propagated(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(cost_usd=0.0005)
        result = pipeline_no_components.process(b"fake")
        assert result.extraction_cost_usd == 0.0005


# ── Extraction failure ────────────────────────────────────────────────────────

class TestExtractionFailure:
    def test_success_false_on_extraction_failure(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(
            success=False, error_type="api_error", error_message="Quota exceeded"
        )
        result = pipeline_no_components.process(b"fake")
        assert result.success is False

    def test_error_type_propagated(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(
            success=False, error_type="timeout", error_message="Timed out"
        )
        result = pipeline_no_components.process(b"fake")
        assert result.error_type == "timeout"

    def test_error_message_propagated(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.return_value = _make_extraction_result(
            success=False, error_type="api_error", error_message="Bad key"
        )
        result = pipeline_no_components.process(b"fake")
        assert result.error_message == "Bad key"

    def test_extraction_exception_handled(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.side_effect = RuntimeError("unexpected!")
        result = pipeline_no_components.process(b"fake")
        assert result.success is False
        assert "unexpected!" in result.error_message

    def test_fallback_recorded_on_exception(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.side_effect = RuntimeError("boom")
        result = pipeline_no_components.process(b"fake")
        assert any("extraction_exception" in f for f in result.fallbacks_triggered)


# ── Preprocessing fallback ────────────────────────────────────────────────────

class TestPreprocessingFallback:
    def _pipeline_with_preprocessing(self):
        pipeline = ReceiptPipeline.__new__(ReceiptPipeline)
        pipeline._enable_preprocessing = True
        pipeline._enable_validation = False
        pipeline._enable_classification = False
        pipeline._gemini_model = "gemini-2.5-flash-lite"
        pipeline._classifier_model = "disabled"
        pipeline._validator = None
        pipeline._classifier = None

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = _make_extraction_result()
        pipeline._extractor = mock_extractor

        return pipeline

    def test_preprocessing_failure_falls_back_to_raw(self):
        pipeline = self._pipeline_with_preprocessing()
        mock_preprocessor = MagicMock()
        mock_proc_result = MagicMock()
        mock_proc_result.success = False
        mock_proc_result.image = None
        mock_proc_result.is_usable = False
        mock_proc_result.warnings = ["is_empty"]
        mock_proc_result.steps_applied = []
        mock_preprocessor.process.return_value = mock_proc_result
        pipeline._preprocessor = mock_preprocessor

        result = pipeline.process(b"fake")
        # Pipeline should still succeed (raw image fallback)
        assert result.success is True
        assert any("preprocessing_failed" in f for f in result.fallbacks_triggered)

    def test_preprocessing_exception_falls_back(self):
        pipeline = self._pipeline_with_preprocessing()
        mock_preprocessor = MagicMock()
        mock_preprocessor.process.side_effect = Exception("cv2 error")
        pipeline._preprocessor = mock_preprocessor

        result = pipeline.process(b"fake")
        assert result.success is True
        assert any("preprocessing_exception" in f for f in result.fallbacks_triggered)

    def test_preprocessing_steps_recorded_on_success(self):
        pipeline = self._pipeline_with_preprocessing()
        mock_preprocessor = MagicMock()
        mock_proc_result = MagicMock()
        mock_proc_result.success = True
        mock_proc_result.image = b"processed"
        mock_proc_result.steps_applied = ["orientation_90deg", "enhance_clahe"]
        mock_preprocessor.process.return_value = mock_proc_result
        pipeline._preprocessor = mock_preprocessor

        result = pipeline.process(b"fake")
        assert "orientation_90deg" in result.preprocessing_steps
        assert "enhance_clahe" in result.preprocessing_steps


# ── Validation integration ────────────────────────────────────────────────────

class TestValidationIntegration:
    def _pipeline_with_validation(self):
        pipeline = ReceiptPipeline.__new__(ReceiptPipeline)
        pipeline._enable_preprocessing = False
        pipeline._enable_validation = True
        pipeline._enable_classification = False
        pipeline._preprocessor = None
        pipeline._classifier = None
        pipeline._gemini_model = "gemini-2.5-flash-lite"
        pipeline._classifier_model = "disabled"

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = _make_extraction_result()
        pipeline._extractor = mock_extractor

        return pipeline

    def test_corrections_propagated(self):
        pipeline = self._pipeline_with_validation()

        from src.validation.validators import Correction
        correction = Correction(
            field="merchant_name",
            original="Higlands",
            corrected="Highlands Coffee",
            reason="fuzzy match",
        )
        mock_validator = MagicMock()
        mock_validator.validate.return_value = _make_validation_result(
            corrections=[correction],
            warnings=[],
        )
        pipeline._validator = mock_validator

        result = pipeline.process(b"fake")
        assert len(result.validation_corrections) == 1
        assert result.validation_corrections[0]["corrected"] == "Highlands Coffee"

    def test_validation_warnings_propagated(self):
        pipeline = self._pipeline_with_validation()
        mock_validator = MagicMock()
        mock_validator.validate.return_value = _make_validation_result(
            warnings=["total exceeds 1B VND"],
        )
        pipeline._validator = mock_validator

        result = pipeline.process(b"fake")
        assert "total exceeds 1B VND" in result.validation_warnings

    def test_validation_exception_falls_back(self):
        pipeline = self._pipeline_with_validation()
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = RuntimeError("validator crashed")
        pipeline._validator = mock_validator

        result = pipeline.process(b"fake")
        assert result.success is True   # still succeeds with unvalidated receipt
        assert any("validation_exception" in f for f in result.fallbacks_triggered)


# ── Classification integration ────────────────────────────────────────────────

class TestClassificationIntegration:
    def _pipeline_with_classification(self):
        pipeline = ReceiptPipeline.__new__(ReceiptPipeline)
        pipeline._enable_preprocessing = False
        pipeline._enable_validation = False
        pipeline._enable_classification = True
        pipeline._preprocessor = None
        pipeline._validator = None
        pipeline._gemini_model = "gemini-2.5-flash-lite"
        pipeline._classifier_model = "keyword"

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = _make_extraction_result()
        pipeline._extractor = mock_extractor

        return pipeline

    def test_category_propagated(self):
        pipeline = self._pipeline_with_classification()
        mock_clf = MagicMock()
        mock_clf.classify.return_value = _make_classification_result("Ăn uống", 0.92, "keyword")
        mock_clf.using_phobert = False
        pipeline._classifier = mock_clf

        result = pipeline.process(b"fake")
        assert result.category == "Ăn uống"

    def test_confidence_propagated(self):
        pipeline = self._pipeline_with_classification()
        mock_clf = MagicMock()
        mock_clf.classify.return_value = _make_classification_result("Đi lại", 0.88, "phobert")
        mock_clf.using_phobert = True
        pipeline._classifier = mock_clf

        result = pipeline.process(b"fake")
        assert result.category_confidence == 0.88
        assert result.category_method == "phobert"

    def test_classification_exception_handled(self):
        pipeline = self._pipeline_with_classification()
        mock_clf = MagicMock()
        mock_clf.classify.side_effect = RuntimeError("clf error")
        pipeline._classifier = mock_clf

        result = pipeline.process(b"fake")
        assert result.success is True    # still succeeds, category is None
        assert result.category is None
        assert any("classification_exception" in f for f in result.fallbacks_triggered)

    def test_classification_disabled_returns_none_category(self):
        pipeline = ReceiptPipeline.__new__(ReceiptPipeline)
        pipeline._enable_preprocessing = False
        pipeline._enable_validation = False
        pipeline._enable_classification = False
        pipeline._preprocessor = None
        pipeline._validator = None
        pipeline._classifier = None
        pipeline._gemini_model = "gemini-2.5-flash-lite"
        pipeline._classifier_model = "disabled"

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = _make_extraction_result()
        pipeline._extractor = mock_extractor

        result = pipeline.process(b"fake")
        assert result.category is None


# ── Never raises ──────────────────────────────────────────────────────────────

class TestNeverRaises:
    def test_process_never_raises_on_bad_bytes(self, pipeline_no_components):
        """Even completely garbage input should not raise."""
        result = pipeline_no_components.process(b"")
        assert isinstance(result, PipelineResult)

    def test_process_never_raises_when_extractor_explodes(self, pipeline_no_components):
        pipeline_no_components._extractor.extract.side_effect = Exception("nuclear")
        result = pipeline_no_components.process(b"fake")
        assert isinstance(result, PipelineResult)
        assert result.success is False
