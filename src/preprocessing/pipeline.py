"""
src/preprocessing/pipeline.py
-------------------------------
ImagePreprocessor — the single public interface for the preprocessing module.

Callers (pipeline orchestrator, benchmark script, API layer) only import this.
Individual modules (orientation, deskew, document_detect, perspective, enhance)
are implementation details.

Pipeline order (each step assumes the previous ran):
  orientation → assess_quality → document_detect → perspective → deskew → enhance

Idempotency rule (CLAUDE.md):
  If image is already clean (bright, sharp, portrait-oriented), skip heavy steps
  and run only enhance.  This ensures group 01 POS receipts are not degraded.

Contract:
  - process() never raises — always returns a ProcessingResult
  - ProcessingResult.success=False means something failed gracefully
  - The original_image is always preserved for comparison/debug
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

from .document_detect import DocumentBounds, QualityFlags, assess_quality, detect_document, crop_to_document
from .deskew import deskew
from .enhance import enhance
from .orientation import fix_orientation
from .perspective import warp_perspective

log = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ProcessingResult:
    """
    Returned by ImagePreprocessor.process().  Never raises to caller.

    success=False means the image could not be loaded or a fatal error occurred.
    Even on partial failures, image is set to the best available output.
    original_image is always the unmodified input.
    """
    success: bool
    image: np.ndarray | None          # Processed image (BGR); None only on fatal failure
    original_image: np.ndarray | None # Always set — for comparison / debugging

    # Per-step metadata dicts (keys documented in each module)
    orientation_meta: dict = field(default_factory=dict)
    document_meta: dict = field(default_factory=dict)
    perspective_meta: dict = field(default_factory=dict)
    deskew_meta: dict = field(default_factory=dict)
    enhance_meta: dict = field(default_factory=dict)

    # Aggregated
    warnings: list[str] = field(default_factory=list)
    steps_applied: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    # Quality assessment (from document_detect.assess_quality)
    quality_flags: QualityFlags | None = None

    # is_usable=False only when the image is completely unusable
    is_usable: bool = True


# ── Idempotency threshold ─────────────────────────────────────────────────────

_CLEAN_BRIGHTNESS_THRESHOLD = 150   # mean brightness above → skip heavy steps


# ── ImagePreprocessor ─────────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    Orchestrates the full preprocessing pipeline for receipt images.

    Usage:
        preprocessor = ImagePreprocessor()
        result = preprocessor.process("data/raw/01_pos_clean/pos_clean_0000.jpg")
        if result.success:
            # result.image is a BGR numpy array ready for Gemini
            pass

    The preprocessor is thread-safe — process() and process_batch() can be
    called from multiple threads simultaneously.
    """

    def __init__(
        self,
        target_size: int = 1600,
        enable_orientation: bool = True,
        enable_deskew: bool = True,
        enable_document_detect: bool = True,
        enable_perspective: bool = True,
        enable_enhance: bool = True,
        skip_if_already_clean: bool = True,
    ) -> None:
        self.target_size = target_size
        self.enable_orientation = enable_orientation
        self.enable_deskew = enable_deskew
        self.enable_document_detect = enable_document_detect
        self.enable_perspective = enable_perspective
        self.enable_enhance = enable_enhance
        self.skip_if_already_clean = skip_if_already_clean

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        image_input: "str | Path | np.ndarray | bytes",
    ) -> ProcessingResult:
        """
        Run the full preprocessing pipeline on a single image.

        Accepts:
          - str / Path → file path (JPEG, PNG, etc.)
          - np.ndarray → BGR image array (as returned by cv2.imread)
          - bytes       → raw image bytes

        Returns ProcessingResult.  Never raises.
        """
        t0 = time.perf_counter()
        warnings: list[str] = []

        try:
            image, exif_data = self._load_image(image_input)
        except Exception as exc:
            return ProcessingResult(
                success=False,
                image=None,
                original_image=None,
                warnings=[f"failed to load image: {exc}"],
                is_usable=False,
                processing_time_ms=_elapsed_ms(t0),
            )

        if image is None or image.size == 0:
            return ProcessingResult(
                success=False,
                image=None,
                original_image=image,
                warnings=["loaded image is empty or invalid"],
                is_usable=False,
                processing_time_ms=_elapsed_ms(t0),
            )

        original = image.copy()
        result = self._run_pipeline(image, exif_data)
        result.original_image = original
        result.processing_time_ms = _elapsed_ms(t0)
        return result

    def process_batch(
        self,
        image_inputs: "list[str | Path | np.ndarray]",
        max_workers: int = 4,
    ) -> list[ProcessingResult]:
        """
        Process multiple images in parallel using a thread pool.

        Each image is processed independently — one failure does not block others.
        Returns results in the same order as image_inputs.
        """
        results: list[ProcessingResult | None] = [None] * len(image_inputs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.process, inp): i
                for i, inp in enumerate(image_inputs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = ProcessingResult(
                        success=False,
                        image=None,
                        original_image=None,
                        warnings=[f"process_batch worker error: {exc}"],
                        is_usable=False,
                    )

        return [r if r is not None else ProcessingResult(
            success=False, image=None, original_image=None,
            warnings=["unknown batch error"], is_usable=False,
        ) for r in results]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_image(
        self,
        image_input: "str | Path | np.ndarray | bytes",
    ) -> tuple[np.ndarray | None, dict | None]:
        """
        Load image and EXIF data from various input types.

        Returns (bgr_array, exif_dict_or_None).
        Raises on unrecoverable errors (caller wraps in try/except).
        """
        exif_data: dict | None = None

        if isinstance(image_input, np.ndarray):
            return image_input.copy(), None

        if isinstance(image_input, bytes):
            arr = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img, None

        # Path-based loading — use PIL to get EXIF, then convert to BGR
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")

        pil_img = Image.open(path)
        try:
            raw_exif = pil_img.getexif()
            if raw_exif:
                exif_data = dict(raw_exif)
        except Exception:
            pass   # EXIF read failure is non-fatal

        # Convert PIL → OpenCV BGR
        pil_rgb = pil_img.convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
        return bgr, exif_data

    def _run_pipeline(
        self,
        image: np.ndarray,
        exif_data: dict | None,
    ) -> ProcessingResult:
        """
        Internal: run all enabled steps in order, collecting metadata.
        """
        warnings: list[str] = []
        steps_applied: list[str] = []

        orientation_meta: dict = {}
        document_meta: dict = {}
        perspective_meta: dict = {}
        deskew_meta: dict = {}
        enhance_meta: dict = {}
        quality_flags: QualityFlags | None = None

        current = image

        # ── Step 1: Orientation fix ───────────────────────────────────────────
        # use_hough=False: EXIF-only. Hough caused 14% false-positive rotation
        # rate in exp_002, reducing E2E by 18pp (especially group 03 long invoices).
        if self.enable_orientation:
            current, orientation_meta = fix_orientation(current, exif_data=exif_data, use_hough=False)
            if orientation_meta.get("rotation_applied", 0) != 0:
                steps_applied.append(f"orientation_{orientation_meta['rotation_applied']}deg")
            if orientation_meta.get("warning"):
                warnings.append(orientation_meta["warning"])

        # ── Step 2: Quality assessment ────────────────────────────────────────
        quality_flags = assess_quality(current)
        document_meta["quality"] = {
            "blur_score": quality_flags.blur_score,
            "mean_brightness": quality_flags.mean_brightness,
            "is_blurry": quality_flags.is_blurry,
            "is_dark": quality_flags.is_dark,
            "is_empty": quality_flags.is_empty,
        }
        warnings.extend(quality_flags.warnings)

        if quality_flags.is_empty:
            return ProcessingResult(
                success=False,
                image=current,
                original_image=current,
                orientation_meta=orientation_meta,
                document_meta=document_meta,
                warnings=warnings,
                steps_applied=steps_applied,
                quality_flags=quality_flags,
                is_usable=False,
            )

        # ── Idempotency gate: clean images skip heavy geometric steps ─────────
        is_clean = (
            self.skip_if_already_clean
            and not quality_flags.is_dark
            and not quality_flags.is_blurry
            and quality_flags.mean_brightness > _CLEAN_BRIGHTNESS_THRESHOLD
        )

        if not is_clean:
            # ── Step 3: Document detection + crop ────────────────────────────
            if self.enable_document_detect:
                bounds = detect_document(current)
                document_meta["bounds"] = {
                    "method": bounds.method,
                    "confidence": bounds.confidence,
                    "crop_recommended": bounds.crop_recommended,
                    "warning": bounds.warning,
                }
                if bounds.warning:
                    warnings.append(bounds.warning)
                if bounds.crop_recommended:
                    h0, w0 = current.shape[:2]
                    candidate = crop_to_document(current, bounds)
                    h1, w1 = candidate.shape[:2]
                    area_ratio = (h1 * w1) / (h0 * w0)
                    # All four guards must pass before we accept the crop:
                    #   top_ratio   — quad starts in upper 25% (header not cut off)
                    #   left/right  — quad spans ≥76% of image width (no edge cut off)
                    #   confidence  — ≥0.70 (area × coverage; rejects shifted quads)
                    top_y  = float(np.min(bounds.corners[:, 1]))
                    x_left = float(np.min(bounds.corners[:, 0]))
                    x_right = float(np.max(bounds.corners[:, 0]))
                    top_ratio   = top_y   / h0
                    left_ratio  = x_left  / w0
                    right_ratio = x_right / w0
                    failures = []
                    if area_ratio < 0.4:
                        failures.append(f"area_ratio={area_ratio:.2f}<0.40")
                    if top_ratio > 0.25:
                        failures.append(f"top_ratio={top_ratio:.2f}>0.25")
                    if left_ratio > 0.12:
                        failures.append(f"left_ratio={left_ratio:.2f}>0.12 (left edge cut off)")
                    if right_ratio < 0.88:
                        failures.append(f"right_ratio={right_ratio:.2f}<0.88 (right edge cut off)")
                    if bounds.confidence < 0.70:
                        failures.append(f"confidence={bounds.confidence:.2f}<0.70")
                    crop_ok = len(failures) == 0
                    if crop_ok:
                        current = candidate
                        steps_applied.append("document_crop")
                    else:
                        msg = (f"document_crop rejected ({'; '.join(failures)}): "
                               f"result {w1}x{h1} — using full image")
                        warnings.append(msg)
                        log.warning(msg)
                        bounds = DocumentBounds(
                            corners=None, confidence=bounds.confidence,
                            method=bounds.method, crop_recommended=False,
                        )
            else:
                bounds = DocumentBounds(
                    corners=None, confidence=0.0,
                    method="disabled", crop_recommended=False,
                )

            # ── Step 4: Perspective warp ──────────────────────────────────────
            if self.enable_perspective and getattr(bounds, "corners", None) is not None:
                current, perspective_meta = warp_perspective(current, bounds.corners)
                if perspective_meta.get("applied"):
                    steps_applied.append("perspective_warp")
                if perspective_meta.get("reason_skipped"):
                    warnings.append(f"perspective warp skipped: {perspective_meta['reason_skipped']}")
            else:
                perspective_meta = {"applied": False, "reason_skipped": "no corners or disabled"}

            # ── Step 5: Deskew ────────────────────────────────────────────────
            if self.enable_deskew:
                current, deskew_meta = deskew(current)
                if deskew_meta.get("method") == "projection":
                    steps_applied.append(f"deskew_{deskew_meta['angle_applied']}deg")
                if deskew_meta.get("warning"):
                    warnings.append(deskew_meta["warning"])
        else:
            # Clean image — record that we skipped heavy steps
            document_meta["bounds"] = {"method": "skipped_clean", "crop_recommended": False}
            perspective_meta = {"applied": False, "reason_skipped": "clean image gate"}
            deskew_meta = {"method": "skipped", "angle_applied": 0.0, "warning": "clean image gate"}
            steps_applied.append("clean_image_fast_path")

        # ── Step 6: Enhancement (always) ─────────────────────────────────────
        if self.enable_enhance:
            current, enhance_meta = enhance(
                current,
                quality_flags=quality_flags,
                target_long_side=self.target_size,
            )
            steps_applied.extend(
                f"enhance_{s}" for s in enhance_meta.get("steps_applied", [])
            )
            warnings.extend(enhance_meta.get("warnings", []))

        return ProcessingResult(
            success=True,
            image=current,
            original_image=image,
            orientation_meta=orientation_meta,
            document_meta=document_meta,
            perspective_meta=perspective_meta,
            deskew_meta=deskew_meta,
            enhance_meta=enhance_meta,
            warnings=warnings,
            steps_applied=steps_applied,
            quality_flags=quality_flags,
            is_usable=True,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
