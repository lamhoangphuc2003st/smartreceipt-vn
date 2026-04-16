"""
src/pipeline/receipt_pipeline.py
----------------------------------
ReceiptPipeline — thin orchestrator that glues all processing stages together.

Stage order:
  1. Preprocessing  → ImagePreprocessor (deskew, enhance, document detect)
  2. Extraction     → GeminiExtractor (VLM → structured JSON)
  3. Validation     → ReceiptValidator (math check, date sanity, merchant correction)
  4. Classification → ExpenseClassifier (PhoBERT or keyword fallback)

Design principles (CLAUDE.md):
  - No business logic here — only flow control and error handling.
  - Every stage wrapped in try/except; failure = graceful fallback, not crash.
  - Log entry/exit of each stage with timing.
  - Return PipelineResult — never raise to the API layer.
  - If a new version can't process every golden image without crashing, it's not ready.

Fallback tiers:
  Preprocessing fails → use raw image bytes for extraction
  Extraction fails   → PipelineResult(success=False)  [no OCR fallback yet — Week 9+]
  Validation fails   → use unvalidated receipt + log warning
  Classification fails → keyword fallback (built into ExpenseClassifier)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Unified output of ReceiptPipeline.process().

    success=True means at minimum the receipt was extracted successfully.
    Validation and classification failures are captured in their respective
    fields but do NOT set success=False (they have graceful fallbacks).
    """

    success: bool

    # ── Extracted data ───────────────────────────────────────────────────────
    # Populated when success=True.
    merchant_name: str | None = None
    merchant_address: str | None = None
    date: str | None = None
    time: str | None = None
    items: list[dict] = field(default_factory=list)
    subtotal: int | None = None
    discount: int | None = None
    vat_amount: int | None = None
    vat_rate: float | None = None
    total_amount: int | None = None
    payment_method: str | None = None
    receipt_number: str | None = None
    receipt_type: str | None = None

    # ── Classification ───────────────────────────────────────────────────────
    category: str | None = None
    category_confidence: float | None = None
    category_method: str | None = None       # "phobert" | "keyword"
    category_all_scores: dict[str, float] = field(default_factory=dict)

    # ── Validation metadata ──────────────────────────────────────────────────
    validation_corrections: list[dict] = field(default_factory=list)   # {field, original, corrected, reason}
    validation_warnings: list[str] = field(default_factory=list)

    # ── Stage-level diagnostics ──────────────────────────────────────────────
    preprocessing_success: bool = False
    preprocessing_steps: list[str] = field(default_factory=list)

    extraction_success: bool = False
    extraction_from_cache: bool = False
    extraction_cost_usd: float = 0.0
    extraction_attempts: int = 1

    # ── Timing ───────────────────────────────────────────────────────────────
    total_time_ms: float = 0.0
    stage_times_ms: dict[str, float] = field(default_factory=dict)

    # ── Model versions ───────────────────────────────────────────────────────
    model_versions: dict[str, str] = field(default_factory=dict)

    # ── Fallbacks triggered ──────────────────────────────────────────────────
    fallbacks_triggered: list[str] = field(default_factory=list)

    # ── Error info (only when success=False) ─────────────────────────────────
    error_type: str | None = None
    error_message: str | None = None

    # ── Image hash (for deduplication / cache key) ────────────────────────────
    image_hash: str | None = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ReceiptPipeline:
    """
    End-to-end Vietnamese receipt processing pipeline.

    Usage:
        pipeline = ReceiptPipeline()
        result = pipeline.process("data/raw/01_pos_clean/receipt.jpg")
        print(result.category, result.total_amount)

        # With explicit API key (overrides env var):
        pipeline = ReceiptPipeline(gemini_api_key="AIza...")

    Thread-safety: ReceiptPipeline is stateless beyond its sub-component instances.
    GeminiExtractor and ImagePreprocessor are both thread-safe — safe to share
    one ReceiptPipeline instance across concurrent requests.
    """

    def __init__(
        self,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-2.5-flash",
        classifier_model: str = "models/phobert-expense-v1",
        classifier_device: str = "cpu",
        enable_preprocessing: bool = True,
        enable_classification: bool = True,
        enable_validation: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        self._enable_preprocessing = enable_preprocessing
        self._enable_classification = enable_classification
        self._enable_validation = enable_validation

        api_key = gemini_api_key or os.getenv("Gemini_API_Key") or os.getenv("GEMINI_API_KEY")

        # ── Stage 1: Preprocessor ─────────────────────────────────────────────
        if enable_preprocessing:
            self._preprocessor = self._init_preprocessor()
        else:
            self._preprocessor = None

        # ── Stage 2: Extractor ────────────────────────────────────────────────
        self._extractor = self._init_extractor(api_key, gemini_model, cache_dir)
        self._gemini_model = gemini_model

        # ── Stage 3: Validator ────────────────────────────────────────────────
        if enable_validation:
            self._validator = self._init_validator()
        else:
            self._validator = None

        # ── Stage 4: Classifier ───────────────────────────────────────────────
        if enable_classification:
            self._classifier = self._init_classifier(classifier_model, classifier_device)
            self._classifier_model = classifier_model
        else:
            self._classifier = None
            self._classifier_model = "disabled"

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        image_input: "str | Path | bytes",
    ) -> PipelineResult:
        """
        Process a receipt image end-to-end.

        Parameters
        ----------
        image_input : file path (str/Path) or raw image bytes.

        Returns
        -------
        PipelineResult — never raises.
        """
        t_total = time.perf_counter()
        stage_times: dict[str, float] = {}
        fallbacks: list[str] = []

        log.info("Pipeline START — input=%s", _describe_input(image_input))

        # ── Stage 1: Preprocessing ────────────────────────────────────────────
        t0 = time.perf_counter()
        processed_image, preprocessing_steps, preprocessing_success = \
            self._run_preprocessing(image_input, fallbacks)
        stage_times["preprocessing"] = _ms(t0)
        log.info("  [1] Preprocessing %.0fms — success=%s steps=%s",
                 stage_times["preprocessing"], preprocessing_success, preprocessing_steps)

        # ── Stage 2: Extraction ───────────────────────────────────────────────
        t0 = time.perf_counter()
        extraction_result = self._run_extraction(processed_image, fallbacks)
        stage_times["extraction"] = _ms(t0)
        log.info("  [2] Extraction %.0fms — success=%s from_cache=%s attempts=%d cost=$%.5f",
                 stage_times["extraction"],
                 extraction_result.success,
                 extraction_result.from_cache,
                 extraction_result.attempts,
                 extraction_result.cost_usd)

        if not extraction_result.success:
            total_ms = _ms(t_total)
            log.warning("Pipeline FAILED at extraction — %s", extraction_result.error_message)
            return PipelineResult(
                success=False,
                preprocessing_success=preprocessing_success,
                preprocessing_steps=preprocessing_steps,
                extraction_success=False,
                error_type=extraction_result.error_type,
                error_message=extraction_result.error_message,
                total_time_ms=total_ms,
                stage_times_ms=stage_times,
                fallbacks_triggered=fallbacks,
                model_versions=self._model_versions(),
            )

        receipt = extraction_result.receipt
        image_hash = extraction_result.image_hash

        # ── Stage 3: Validation ───────────────────────────────────────────────
        t0 = time.perf_counter()
        receipt, corrections, val_warnings = self._run_validation(receipt, fallbacks)
        stage_times["validation"] = _ms(t0)
        log.info("  [3] Validation %.0fms — corrections=%d warnings=%d",
                 stage_times["validation"], len(corrections), len(val_warnings))

        # ── Stage 4: Classification ───────────────────────────────────────────
        t0 = time.perf_counter()
        clf_result = self._run_classification(receipt, fallbacks)
        stage_times["classification"] = _ms(t0)
        if clf_result:
            log.info("  [4] Classification %.0fms — %s (%.2f, %s)",
                     stage_times["classification"],
                     clf_result.category, clf_result.confidence, clf_result.method)

        # ── Assemble result ───────────────────────────────────────────────────
        total_ms = _ms(t_total)
        log.info("Pipeline DONE %.0fms — category=%s total=%s",
                 total_ms, clf_result.category if clf_result else None, receipt.total_amount)

        items_serialized = [
            {
                "name": it.name,
                "quantity": it.quantity,
                "unit": it.unit,
                "unit_price": it.unit_price,
                "total": it.total,
            }
            for it in (receipt.items or [])
        ]

        corrections_serialized = [
            {"field": c.field, "original": c.original, "corrected": c.corrected, "reason": c.reason}
            for c in corrections
        ]

        return PipelineResult(
            success=True,
            # Receipt data
            merchant_name=receipt.merchant_name,
            merchant_address=receipt.merchant_address,
            date=receipt.date,
            time=receipt.time,
            items=items_serialized,
            subtotal=receipt.subtotal,
            discount=receipt.discount,
            vat_amount=receipt.vat_amount,
            vat_rate=receipt.vat_rate,
            total_amount=receipt.total_amount,
            payment_method=receipt.payment_method,
            receipt_number=receipt.receipt_number,
            receipt_type=receipt.receipt_type,
            # Classification
            category=clf_result.category if clf_result else None,
            category_confidence=clf_result.confidence if clf_result else None,
            category_method=clf_result.method if clf_result else None,
            category_all_scores=clf_result.all_scores if clf_result else {},
            # Validation metadata
            validation_corrections=corrections_serialized,
            validation_warnings=val_warnings,
            # Stage diagnostics
            preprocessing_success=preprocessing_success,
            preprocessing_steps=preprocessing_steps,
            extraction_success=True,
            extraction_from_cache=extraction_result.from_cache,
            extraction_cost_usd=extraction_result.cost_usd,
            extraction_attempts=extraction_result.attempts,
            # Timing
            total_time_ms=total_ms,
            stage_times_ms=stage_times,
            # Versions
            model_versions=self._model_versions(),
            fallbacks_triggered=fallbacks,
            image_hash=image_hash,
        )

    # ── Stage runners (each wraps failures gracefully) ─────────────────────────

    def _run_preprocessing(
        self,
        image_input: "str | Path | bytes",
        fallbacks: list[str],
    ) -> "tuple[str | Path | bytes, list[str], bool]":
        """
        Run preprocessing.  On failure, return original input (raw-image fallback).
        """
        if not self._enable_preprocessing or self._preprocessor is None:
            return image_input, [], False

        try:
            result = self._preprocessor.process(image_input)
            if result.success and result.image is not None:
                return result.image, result.steps_applied, True
            else:
                fallbacks.append("preprocessing_failed→raw_image")
                log.warning("Preprocessing failed (is_usable=%s): %s",
                            result.is_usable, result.warnings)
                return image_input, [], False
        except Exception as exc:
            fallbacks.append(f"preprocessing_exception→raw_image: {exc}")
            log.exception("Preprocessing raised unexpectedly: %s", exc)
            return image_input, [], False

    def _run_extraction(
        self,
        image_input: "str | Path | bytes | object",
        fallbacks: list[str],
    ) -> "object":  # ExtractionResult
        """
        Run Gemini extraction.  Returns ExtractionResult (always).
        """
        try:
            return self._extractor.extract(image_input)
        except Exception as exc:
            fallbacks.append(f"extraction_exception: {exc}")
            log.exception("Extractor raised unexpectedly: %s", exc)
            # Import here to avoid circular issues at module level
            from src.extraction.schemas import ExtractionResult
            return ExtractionResult(
                success=False,
                error_type="exception",
                error_message=str(exc),
            )

    def _run_validation(
        self,
        receipt: "object",  # Receipt
        fallbacks: list[str],
    ) -> "tuple[object, list, list[str]]":
        """
        Run validation.  On failure, return original receipt with empty corrections.
        """
        if not self._enable_validation or self._validator is None:
            return receipt, [], []

        try:
            result = self._validator.validate(receipt)
            return result.receipt, result.corrections, result.warnings
        except Exception as exc:
            fallbacks.append(f"validation_exception→unvalidated: {exc}")
            log.exception("Validator raised unexpectedly: %s", exc)
            return receipt, [], [f"Validation error: {exc}"]

    def _run_classification(
        self,
        receipt: "object",  # Receipt
        fallbacks: list[str],
    ) -> "object | None":  # ClassificationResult | None
        """
        Run classification.  Built-in keyword fallback inside ExpenseClassifier.
        Returns None only if classification is disabled.
        """
        if not self._enable_classification or self._classifier is None:
            return None

        try:
            merchant = getattr(receipt, "merchant_name", None)
            items = [it.name for it in (getattr(receipt, "items", None) or [])]
            return self._classifier.classify(merchant, items)
        except Exception as exc:
            fallbacks.append(f"classification_exception: {exc}")
            log.exception("Classifier raised unexpectedly: %s", exc)
            return None

    # ── Component initializers ────────────────────────────────────────────────

    def _init_preprocessor(self) -> "object":
        try:
            from src.preprocessing import ImagePreprocessor
            log.info("ImagePreprocessor loaded")
            return ImagePreprocessor()
        except Exception as exc:
            log.warning("Could not init ImagePreprocessor: %s — preprocessing disabled", exc)
            return None

    def _init_extractor(
        self,
        api_key: str | None,
        model: str,
        cache_dir: str | None,
    ) -> "object":
        from src.extraction import GeminiExtractor
        kwargs: dict[str, Any] = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        if cache_dir:
            from pathlib import Path as _Path
            kwargs["cache_dir"] = _Path(cache_dir)
        extractor = GeminiExtractor(**kwargs)
        log.info("GeminiExtractor loaded (model=%s)", model)
        return extractor

    def _init_validator(self) -> "object":
        try:
            from src.validation import ReceiptValidator
            log.info("ReceiptValidator loaded")
            return ReceiptValidator()
        except Exception as exc:
            log.warning("Could not init ReceiptValidator: %s — validation disabled", exc)
            return None

    def _init_classifier(self, model_name: str, device: str) -> "object":
        try:
            from src.classification import ExpenseClassifier
            clf = ExpenseClassifier.from_pretrained(model_name, device=device)
            log.info(
                "ExpenseClassifier loaded — using_phobert=%s",
                clf.using_phobert,
            )
            return clf
        except Exception as exc:
            log.warning("Could not init ExpenseClassifier: %s — classification disabled", exc)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _model_versions(self) -> dict[str, str]:
        return {
            "extractor": self._gemini_model,
            "classifier": self._classifier_model,
        }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _describe_input(image_input: "str | Path | bytes") -> str:
    if isinstance(image_input, bytes):
        return f"<bytes {len(image_input)} bytes>"
    return str(image_input)
