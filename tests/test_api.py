"""
tests/test_api.py
------------------
Integration tests for the FastAPI receipt API.

Uses FastAPI TestClient (no live server needed).
All external I/O (Gemini, HF Hub) is mocked at the pipeline level.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.pipeline.receipt_pipeline import PipelineResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _successful_result(**overrides) -> PipelineResult:
    defaults = dict(
        success=True,
        merchant_name="Highlands Coffee",
        merchant_address="123 Nguyen Hue",
        date="2024-03-15",
        time="10:30",
        items=[{"name": "Cà phê sữa", "quantity": 1.0, "unit": "ly", "unit_price": 55000, "total": 55000}],
        subtotal=55000,
        discount=None,
        vat_amount=None,
        vat_rate=None,
        total_amount=55000,
        payment_method="thẻ",
        receipt_number="R001",
        receipt_type="pos",
        category="Ăn uống",
        category_confidence=0.92,
        category_method="keyword",
        category_all_scores={"Ăn uống": 0.92},
        validation_corrections=[],
        validation_warnings=[],
        preprocessing_success=True,
        preprocessing_steps=["enhance_clahe"],
        extraction_success=True,
        extraction_from_cache=False,
        extraction_cost_usd=0.0003,
        extraction_attempts=1,
        total_time_ms=1234.5,
        stage_times_ms={"preprocessing": 100.0, "extraction": 1000.0, "validation": 50.0, "classification": 80.0},
        model_versions={"extractor": "gemini-2.5-flash-lite", "classifier": "keyword"},
        fallbacks_triggered=[],
        image_hash="abc123def",
        error_type=None,
        error_message=None,
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


def _failed_result(**overrides) -> PipelineResult:
    defaults = dict(
        success=False,
        error_type="api_error",
        error_message="Quota exceeded",
        preprocessing_success=False,
        preprocessing_steps=[],
        extraction_success=False,
        extraction_from_cache=False,
        extraction_cost_usd=0.0,
        extraction_attempts=1,
        total_time_ms=123.0,
        stage_times_ms={},
        model_versions={"extractor": "gemini-2.5-flash-lite"},
        fallbacks_triggered=["extraction_failed"],
        image_hash=None,
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


def _minimal_jpeg() -> bytes:
    """Minimal valid JPEG bytes (1x1 white pixel)."""
    # Standard minimal JPEG
    return (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
        b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e\xfe...'
        b'\xff\xd9'
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """A mock ReceiptPipeline that returns a successful result by default."""
    pipeline = MagicMock()
    pipeline._preprocessor = MagicMock()
    pipeline._validator = MagicMock()
    pipeline._classifier = MagicMock()
    pipeline._classifier.using_phobert = False
    pipeline.process.return_value = _successful_result()
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """
    FastAPI TestClient with pipeline injected.

    Patches ReceiptPipeline so the lifespan startup gets our mock,
    not a real pipeline that would try to connect to Gemini / HF Hub.
    Rate limiting is disabled so tests can make multiple requests freely.
    """
    from api.main import app

    with patch("src.pipeline.ReceiptPipeline", return_value=mock_pipeline):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        from api.main import APP_VERSION
        assert data["version"] == APP_VERSION

    def test_health_has_components(self, client):
        data = client.get("/health").json()
        assert "components" in data
        assert isinstance(data["components"], dict)


# ── Extract endpoint — success ────────────────────────────────────────────────

class TestExtractSuccess:
    def test_extract_returns_200(self, client):
        resp = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_extract_success_true(self, client):
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert data["success"] is True

    def test_merchant_name_in_response(self, client):
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert data["merchant_name"] == "Highlands Coffee"

    def test_total_amount_in_response(self, client):
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert data["total_amount"] == 55000

    def test_classification_in_response(self, client):
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert data["classification"]["category"] == "Ăn uống"
        assert data["classification"]["confidence"] == 0.92
        assert data["classification"]["method"] == "keyword"

    def test_items_in_response(self, client):
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Cà phê sữa"
        assert data["items"][0]["total"] == 55000

    def test_meta_in_response(self, client):
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert "meta" in data
        assert data["meta"]["total_time_ms"] == 1234.5

    def test_corrections_in_response(self, client, mock_pipeline):
        mock_pipeline.process.return_value = _successful_result(
            validation_corrections=[{
                "field": "merchant_name",
                "original": "Higlands",
                "corrected": "Highlands Coffee",
                "reason": "fuzzy match",
            }]
        )
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert len(data["corrections"]) == 1
        assert data["corrections"][0]["corrected"] == "Highlands Coffee"

    def test_warnings_in_response(self, client, mock_pipeline):
        mock_pipeline.process.return_value = _successful_result(
            validation_warnings=["total exceeds 1B VND"]
        )
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert "total exceeds 1B VND" in data["warnings"]

    def test_png_accepted(self, client):
        # PNG header (1x1 transparent)
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00'
            b'\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        resp = client.post(
            "/receipts/extract",
            files={"file": ("receipt.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200


# ── Extract endpoint — extraction failure ─────────────────────────────────────

class TestExtractFailure:
    def test_extraction_failure_success_false(self, client, mock_pipeline):
        mock_pipeline.process.return_value = _failed_result()
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert data["success"] is False

    def test_extraction_failure_error_type(self, client, mock_pipeline):
        mock_pipeline.process.return_value = _failed_result(error_type="timeout")
        data = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        ).json()
        assert data["error_type"] == "timeout"

    def test_extraction_failure_still_200(self, client, mock_pipeline):
        """API returns HTTP 200 even on extraction failure — error is in response body."""
        mock_pipeline.process.return_value = _failed_result()
        resp = client.post(
            "/receipts/extract",
            files={"file": ("receipt.jpg", _minimal_jpeg(), "image/jpeg")},
        )
        assert resp.status_code == 200


# ── Extract endpoint — input validation ───────────────────────────────────────

class TestExtractInputValidation:
    def test_unsupported_content_type_415(self, client):
        resp = client.post(
            "/receipts/extract",
            files={"file": ("receipt.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 415

    def test_empty_file_400(self, client):
        resp = client.post(
            "/receipts/extract",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_no_file_422(self, client):
        """Missing file field → FastAPI validation error 422."""
        resp = client.post("/receipts/extract")
        assert resp.status_code == 422

    def test_too_large_file_413(self, client):
        big_bytes = b"x" * (10 * 1024 * 1024 + 1)   # 10 MB + 1 byte
        resp = client.post(
            "/receipts/extract",
            files={"file": ("big.jpg", big_bytes, "image/jpeg")},
        )
        assert resp.status_code == 413


# ── OpenAPI docs ──────────────────────────────────────────────────────────────

class TestOpenAPIDocs:
    def test_openapi_json_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/receipts/extract" in data["paths"]

    def test_docs_endpoint_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
